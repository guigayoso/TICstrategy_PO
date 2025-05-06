import data_models as dm
import data_utils as utils
from datetime import datetime as dt
import pandas as pd
import yahoofinancials as yf
import numpy as np
import portfolio_weights as pw
import tc_optimization as tc
import data_analysis as analysis
import data_utils as utils
from sklearn.model_selection import train_test_split

def calculate_cumulative_returns(weights, returns_df):
    """
    Calculate cumulative returns for a portfolio based on weights and asset returns.

    Parameters:
        weights (np.array): Weights of assets in the portfolio.
        returns_df (pd.DataFrame): DataFrame with returns of assets.

    Returns:
        float: Cumulative portfolio returns.
    """
    # Calculate cumulative returns for each asset
    cumulative_returns = (1+returns_df).prod() - 1

    #print(f"cumulative_returns: {cumulative_returns}")
    #print(f"weights: {weights}")
    # Calculate portfolio returns by weighting the asset returns
    portfolio_returns = cumulative_returns.dot(weights)

    return portfolio_returns

'''
Example: Here the code will need to be alterated because the rebalancing period
will be implemented as a dm.Rebalancing_Period
So: def get_portfolio_performance(weights,returns_df,
rebalancing_periods= dm.Rebalancing_Period.WEEKLY): 

'''

def get_portfolio_performance(trendy_assets, mean_reverting_assets, buy_and_sell_signals_dict, close_df, rebalancing_periods=7, shorting_value=0.0, transaction_cost=0.01, distance_method="Manhattan", gamma=0, delta_range=(0, 3), step=1, verbose=True):
    """
    Calculate the portfolio performance based in buy and sell signalling considering the rebalancing periods.

    Parameters:
    trendy_assets (list): The assets that are trending in each period
    mean_reverting_assets (list): The assets that are mean reverting in each period
    buy_and_sell_signals_dict (dict): The buy and sell signals for each asset
    close_df (pd.DataFrame): The close price of the assets
    rebalancing_periods (int): Number of days to rebalance the portfolio
    shorting_value (float): The shorting value
    transaction_cost (float): The transaction cost
    distance_method (str): The distance method to use for the no-trade zone
    gamma (float): The factor of risk aversion for the CRRA function (0 for risk neutral)
    delta_range (tuple): The delta range for the optimization
    step (float): The step for the delta optimization
    verbose (bool): The verbose flag to print delta optimization 

    Returns:
    dict: The portfolio performance
    dict: The portfolio performance with costs
    pd.DataFrame: The target weights DataFrame
    pd.DataFrame: The evolution DataFrame
    pd.DataFrame: The new contributions DataFrame
    pd.DataFrame: The new contributions normalized DataFrame
    """
    dates_to_iterate = list(buy_and_sell_signals_dict.keys())
    rebalances_dates = [close_df.index[i] for i in dates_to_iterate]
    print(f" Close prices in the beginning of the period: \n {close_df.loc[rebalances_dates[0]]}")
    print(f" Close prices in the end of the period: \n {close_df.loc[rebalances_dates[-1]]}")
    print(f" Close prices in the pre end of the period: \n {close_df.loc[rebalances_dates[-2]]}")


    all_assets = close_df.columns
    returns_df = close_df.pct_change()

    weights_df = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # Target weights defined by the strategy

    evolution_df = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # Evolution of the assets during the previous rebalancing period
    new_contributions = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # New asset contributions based on the evolution of the assets
    new_contributions_normalized = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # New asset contributions normalized
    unbalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # The weights at the end of the rebalancing period. It is equal to the new contributions normalized
    rebalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # A weighted sum between the actual weights and the target weights of the next period. It is the optimal allocation of the portfolio in the beginning of the period

    target_new_contributions = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # New asset contributions based on the evolution of the target
    target_new_contributions_normalized = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # New asset contributions normalized based on the evolution of the target
    target_unbalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # The weights at the end of the rebalancing period based on the target weights

    target_performance = {} # The performance of the portfolio in the end of the period with the target weights
    performance_w_optimal_costs = {} # The performance of the portfolio in the end of the period with the optimal weights considering the transaction costs

    previous_assets_to_buy = None

    total_target_cost_of_transaction = 0
    total_optimal_cost_of_transaction = 0

    for i, date in enumerate(dates_to_iterate):
        print(f"\n ---------------------------------------Date: {date} or {rebalances_dates[i]}---------------------------------------------")

        assets_to_buy, _ = analysis.extract_assets(buy_and_sell_signals_dict, date)

        if len(assets_to_buy) > 0:

            # From 96 to 102: "Beginning of the period, we calculate the target weights based on the signals generated"
            buy_array, _, _, assets_to_buy = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value)
            
            print(f"Signals were generated to buy the following assts: {assets_to_buy}")
            print(f"With the following weights: {buy_array}")

            pw.update_weights(weights_df, date, assets_to_buy, buy_array)
            print(f"\n Updating the weights_df: \n {weights_df}")

            if i == 0:
                # In the beginning of a period i>0 we start with the optimal weights, which are the rebalanced
                # weights from the previous period, but in the first period we start with the target weights
                # é como se fosse um rebalanced(i == -1)
                rebalanced_weights.loc[date] = weights_df.loc[date]
                print(f"\n Initial rebalanced weights at {date}: \n {rebalanced_weights.loc[date]}")

                previous_weights = weights_df.loc[date]
                previous_assets_to_buy = assets_to_buy
                continue
            
            # From 116 to 144: "End" of the period, we calculate the evolution of the assets and the new contributions
            previous_date = dates_to_iterate[i - 1]
            _, previous_dataframe_assets, _ = analysis.filter_assets_data(close_df, previous_date, rebalancing_periods, previous_assets_to_buy, all_assets)

            cumulative_returns = (1 + previous_dataframe_assets).cumprod() - 1
            evolution_df.loc[previous_date] = cumulative_returns.iloc[-1]
            print(f"\n Evolution of assets from {previous_date} to {date}: \n {evolution_df.loc[previous_date]}")
            print(f"Previous dataframe assets: {previous_dataframe_assets}")
            
            print("\n------------------Optimal Weights Evolution------------------")
            # Grab on the rebalanced weights form the beginning of the last period and evolve the assets until the end of the last period (for i == 0 the rebalanced weights are the target weights)
            new_contributions, new_contributions_normalized = pw.calculate_new_contributions(rebalanced_weights, evolution_df, date, previous_date)
            print(f"\n New contributions before normalization in the end of the period {previous_date}: \n {new_contributions}")
            print(f"\n Normalized new contributions in the end of the period {previous_date}: \n {new_contributions_normalized}")

            unbalanced_weights.loc[previous_date] = new_contributions_normalized
            print(f"\n Unbalanced weights in the end of the period {previous_date}: \n {unbalanced_weights.loc[previous_date]}")

            print("\n------------------Target Weights Evolution------------------")
            if weights_df.loc[previous_date].isnull().all():
                previous_weights = target_new_contributions_normalized # If the last weights were NaN, we use the target_new_contributions_normalized
                target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(target_new_contributions_normalized, evolution_df, date, previous_date, target=True)
            else:
                previous_weights = weights_df.loc[previous_date] # If we had signals in the last period, we use the weights_df
                target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(weights_df, evolution_df, date, previous_date)
            print(f"\n New contributions of the target before normalization in the end of the period {previous_date}: \n {target_new_contributions}")
            print(f"\n Normalized new contributions of the target in the end of the period {previous_date}: \n {target_new_contributions_normalized}")

            target_unbalanced_weights.loc[previous_date] = target_new_contributions_normalized
            print(f"\n Unbalanced weights of the target in the end of the period {previous_date}: \n {target_unbalanced_weights.loc[previous_date]}")
                
            # From 147 to 198: Prepare the beginning of the next period. Since the last period is over, we calculate the performance of the portfolio in the last period
            print("\n------------------Performances------------------")
            target_performance[previous_date] = calculate_cumulative_returns(previous_weights, previous_dataframe_assets)
            print(f"\n Target performance at {previous_date} before transaction costs: {target_performance[previous_date]} using the previous weights: {previous_weights}")

            performance_w_optimal_costs[previous_date] = calculate_cumulative_returns(rebalanced_weights.loc[previous_date].values, previous_dataframe_assets)
            print(f"\n Performance of the portfolio in the end of the period {previous_date} before transaction costs was {performance_w_optimal_costs[previous_date]} with the optimal weights: {rebalanced_weights.loc[previous_date]}")

            if i > 1:
                valid_weights = weights_df.dropna().loc[:dates_to_iterate[i - 1]]
                #print(f"\n Valid weights: \n {valid_weights}")

                if len(valid_weights) >= 2:
                    last_date = valid_weights.index[-1]
                    previous_target_unbalanced_weights = target_unbalanced_weights.loc[last_date - rebalancing_periods]
                    print(f"\n Due to the change in weights from {last_date - rebalancing_periods} to {last_date} we have:")

                    target_performance[previous_date], target_cost_of_transaction = tc.apply_transaction_cost(
                        target_performance[previous_date],
                        transaction_cost,
                        weights_df.loc[last_date].values,
                        previous_target_unbalanced_weights.values
                    )

                    total_target_cost_of_transaction += target_cost_of_transaction
                    print(f"\n Target performance at {previous_date} after transaction costs: {target_performance[previous_date]}")

                    previous_unbalanced_weights = unbalanced_weights.loc[last_date - rebalancing_periods]

                    performance_w_optimal_costs[previous_date], optimal_cost_of_transaction = tc.apply_transaction_cost(
                        performance_w_optimal_costs[previous_date],
                        transaction_cost,
                        rebalanced_weights.loc[last_date].values,
                        previous_unbalanced_weights.values
                    )

                    total_optimal_cost_of_transaction += optimal_cost_of_transaction
                    print(f"\n Performance of the portfolio in the end of the period {previous_date} after transaction costs: {performance_w_optimal_costs[previous_date]}")
                else:
                    print(f"\n Not enough valid dates to apply transaction costs")
            else:
                print(f"\n No transaction costs to apply in the first period")

            # In this part, we calculate the optimal delta (alpha) to prepare the rebalanced weights for the next period
            # i == 0 is the first period, so we don't have previous weights to calculate the transaction costs
            delta, crra, alpha = tc.optimize_delta_refined(
            weights_df.loc[date].values, unbalanced_weights.loc[previous_date].values, 
            trendy_assets, mean_reverting_assets, previous_dataframe_assets, 
            previous_date, date,
            distance_method=distance_method, gamma=gamma, delta_range=delta_range, initial_step=step, 
            transaction_cost=transaction_cost, verbose=verbose
            ) 

            """ alpha, _ = tc.optimize_alpha(weights_df.loc[date].values, unbalanced_weights.loc[previous_date].values, 
            trendy_assets, mean_reverting_assets, previous_dataframe_assets,
            previous_date, date, gamma=gamma, alpha_range=(0, 1), step=0.01,
            transaction_cost=transaction_cost, verbose=verbose) """
            
            rebalanced_weights.loc[date] = pw.calculate_rebalanced_weights(alpha, weights_df.loc[date], unbalanced_weights.loc[previous_date])
            print(f"\n Combining the target weights at {date} and the unbalanced weights at {previous_date} with an alpha of {alpha} we have the rebalanced weights for the beginning of {date}: \n {rebalanced_weights.loc[date]}")

            #previous_weights = weights_df.loc[date]
            previous_assets_to_buy = assets_to_buy
        
        else:
            print(f"\n No signals to buy for {date}. Continuing with previous assets.")
            if i > 0:
                previous_date = dates_to_iterate[i - 1]
                _, previous_dataframe_assets, _ = analysis.filter_assets_data(close_df, previous_date, rebalancing_periods, previous_assets_to_buy, all_assets)

                # No signals
                weights_df.loc[date] = np.nan
                print(f"\n No signals generated: \n {weights_df}")
                print(f"\n Since we don't have signals, we keep the previous assets.")
                #print(f"\n To keep evolving the unbalanced weights {unbalanced_weights.loc[previous_date]} we need to calculate the evolution of the assets from {previous_date} to {date}")

                # Calculate the evolution of the assets from the previous date to the current date
                cumulative_returns = (1 + previous_dataframe_assets).cumprod() - 1
                evolution_df.loc[previous_date] = cumulative_returns.iloc[-1]
                print(f"\n Using the previous data frame assets: {previous_dataframe_assets}")
                print(f"\n The evolution of these assets from {previous_date} to {date} is: \n {evolution_df.loc[previous_date]}")


                print(f"\n------------------Optimal Weights Evolution------------------")
                # Calculate the new contributions in the end of the rebalancing period based on the evolution of the assets, even if there are signals or not
                new_contributions, new_contributions_normalized = pw.calculate_new_contributions(rebalanced_weights, evolution_df, date, previous_date)
                print(f"\n New contributions before normalization in the end of the period {previous_date}: \n {new_contributions}")
                print(f"\n Normalized new contributions in the end of the period {previous_date}: \n {new_contributions_normalized}")

                unbalanced_weights.loc[previous_date] = new_contributions_normalized
                print(f"\n Unbalanced weights in the end of the period {previous_date}: \n {unbalanced_weights.loc[previous_date]}")

                print(f"\n------------------Target Weights Evolution------------------")
                # condicao para ver se o weights df era nan na iteracao pssada, se for nan, usar o target new normalized, se nao for nan usar o proprio weight df
                if weights_df.loc[previous_date].isnull().all():
                    previous_weights = target_new_contributions_normalized # If the last weights were NaN, we use the target_new_contributions_normalized
                    target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(target_new_contributions_normalized, evolution_df, date, previous_date, target=True)
                else:
                    previous_weights = weights_df.loc[previous_date] # If we had signals in the last period, we use the weights_df
                    target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(weights_df, evolution_df, date, previous_date)
                print(f"\n New contributions of the target before normalization in the end of the period {previous_date}: \n {target_new_contributions}")
                print(f"\n Normalized new contributions of the target in the end of the period {previous_date}: \n {target_new_contributions_normalized}")

                target_unbalanced_weights.loc[previous_date] = target_new_contributions_normalized

                print(f"\n------------------Performances------------------")
                target_performance[previous_date] = calculate_cumulative_returns(previous_weights, previous_dataframe_assets)
                print(f"\n Target performance at {previous_date} before transaction costs: {target_performance[previous_date]} using the previous weights: {previous_weights}")

                performance_w_optimal_costs[previous_date] = calculate_cumulative_returns(rebalanced_weights.loc[previous_date].values, previous_dataframe_assets)
                print(f"\n Performance of the portfolio in the end of the period {previous_date} was {performance_w_optimal_costs[previous_date]} with the optimal weights: {rebalanced_weights.loc[previous_date]}")


                if i > 1:
                    valid_weights = weights_df.dropna().loc[:dates_to_iterate[i - 1]]

                    if len(valid_weights) >= 2 and weights_df.loc[dates_to_iterate[i - 1]].notna().all():
                        last_date = valid_weights.index[-1]
                        previous_target_unbalanced_weights = target_unbalanced_weights.loc[last_date - rebalancing_periods]
                        print(f"\n Due to the change in weights from {last_date - rebalancing_periods} to {last_date} we have:")

                        target_performance[previous_date], target_cost_of_transaction = tc.apply_transaction_cost(
                            target_performance[previous_date],
                            transaction_cost,
                            weights_df.loc[last_date].values,
                            previous_target_unbalanced_weights.values
                        )

                        total_target_cost_of_transaction += target_cost_of_transaction
                        print(f"\n Target performance at {previous_date} after transaction costs: {target_performance[previous_date]}")

                        previous_unbalanced_weights = unbalanced_weights.loc[last_date - rebalancing_periods]

                        performance_w_optimal_costs[previous_date], optimal_cost_of_transaction = tc.apply_transaction_cost(
                            performance_w_optimal_costs[previous_date],
                            transaction_cost,
                            rebalanced_weights.loc[last_date].values,
                            previous_unbalanced_weights.values
                        )

                        total_optimal_cost_of_transaction += optimal_cost_of_transaction
                        print(f"\n Performance of the portfolio in the end of the period {previous_date} after transaction costs: {performance_w_optimal_costs[previous_date]}")
                    else:
                        print(f"\n No transaction costs to apply")
                else:
                    print(f"\n No transaction costs to apply in the first period")

                # Since we don't have signals, we assume that the unbalanced weights are the rebalanced weights for the next period
                rebalanced_weights.loc[date] = unbalanced_weights.loc[previous_date]

                _, previous_dataframe_assets, previous_dataframe_assets_filtered = analysis.filter_assets_data(close_df, date, rebalancing_periods, previous_assets_to_buy, all_assets)
                #print(f"\n Preparing the previous dataframe assets for the next period: {previous_dataframe_assets}")        
        

    return rebalances_dates, target_performance, performance_w_optimal_costs, weights_df, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction

def get_portfolio_performance_delta(trendy_assets, mean_reverting_assets, buy_and_sell_signals_dict, close_df, rebalancing_periods=7, shorting_value=0.0, transaction_cost=0.01, distance_method="Manhattan", gamma=0, delta = 1, verbose=True):
    """
    Calculate the portfolio performance based in buy and sell signalling considering the rebalancing periods.

    Parameters:
    trendy_assets (list): The assets that are trending in each period
    mean_reverting_assets (list): The assets that are mean reverting in each period
    buy_and_sell_signals_dict (dict): The buy and sell signals for each asset
    close_df (pd.DataFrame): The close price of the assets
    rebalancing_periods (int): Number of days to rebalance the portfolio
    shorting_value (float): The shorting value
    transaction_cost (float): The transaction cost
    distance_method (str): The distance method to use for the no-trade zone
    gamma (float): The factor of risk aversion for the CRRA function (0 for risk neutral)
    delta_range (tuple): The delta range for the optimization
    step (float): The step for the delta optimization
    verbose (bool): The verbose flag to print delta optimization 

    Returns:
    dict: The portfolio performance
    dict: The portfolio performance with costs
    pd.DataFrame: The target weights DataFrame
    pd.DataFrame: The evolution DataFrame
    pd.DataFrame: The new contributions DataFrame
    pd.DataFrame: The new contributions normalized DataFrame
    """
    dates_to_iterate = list(buy_and_sell_signals_dict.keys())
    print(f"Dates to iterate: {dates_to_iterate}")
    rebalances_dates = [close_df.index[i] for i in dates_to_iterate]
    #print(f" Close prices in the beginning of the period: \n {close_df.loc[rebalances_dates[0]]}")
    #print(f" Close prices in the end of the period: \n {close_df.loc[rebalances_dates[-1]]}")
    #print(f" Close prices in the pre end of the period: \n {close_df.loc[rebalances_dates[-2]]}")


    all_assets = close_df.columns
    returns_df = close_df.pct_change()

    weights_df = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # Target weights defined by the strategy

    evolution_df = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # Evolution of the assets during the previous rebalancing period
    new_contributions = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # New asset contributions based on the evolution of the assets
    new_contributions_normalized = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # New asset contributions normalized
    unbalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # The weights at the end of the rebalancing period. It is equal to the new contributions normalized
    rebalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets, dtype=float).fillna(0.0) # A weighted sum between the actual weights and the target weights of the next period. It is the optimal allocation of the portfolio in the beginning of the period

    target_new_contributions = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # New asset contributions based on the evolution of the target
    target_new_contributions_normalized = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # New asset contributions normalized based on the evolution of the target
    target_unbalanced_weights = pd.DataFrame(index=dates_to_iterate, columns=all_assets).fillna(0.0) # The weights at the end of the rebalancing period based on the target weights

    target_performance = {} # The performance of the portfolio in the end of the period with the target weights
    performance_w_optimal_costs = {} # The performance of the portfolio in the end of the period with the optimal weights considering the transaction costs

    previous_assets_to_buy = None

    total_target_cost_of_transaction = 0
    total_optimal_cost_of_transaction = 0

    CRRA_total = 0

    for i, date in enumerate(dates_to_iterate):
        print(f"\n ---------------------------------------Date: {date} or {rebalances_dates[i]}---------------------------------------------")

        assets_to_buy, _ = analysis.extract_assets(buy_and_sell_signals_dict, date)

        if len(assets_to_buy) > 0:

            # From 96 to 102: "Beginning of the period, we calculate the target weights based on the signals generated"
            buy_array, _, _, _ = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value)
            
            print(f"Signals were generated to buy the following assts: {assets_to_buy}")
            print(f"With the following weights: {buy_array}")

            pw.update_weights_df(weights_df, date, assets_to_buy, buy_array)
            print(f"\n Updating the weights_df: \n {weights_df}")

            if i == 0:
                # In the beginning of a period i>0 we start with the optimal weights, which are the rebalanced
                # weights from the previous period, but in the first period we start with the target weights
                # é como se fosse um rebalanced(i == -1)
                rebalanced_weights.loc[date] = weights_df.loc[date]
                print(f"\n Initial rebalanced weights at {date}: \n {rebalanced_weights.loc[date]}")

                previous_weights = weights_df.loc[date]
                previous_assets_to_buy = assets_to_buy
                continue
            
            # From 116 to 144: "End" of the period, we calculate the evolution of the assets and the new contributions
            previous_date = dates_to_iterate[i - 1]
            _, previous_dataframe_assets, _ = analysis.filter_assets_data(close_df, previous_date, rebalancing_periods, previous_assets_to_buy, all_assets)

            cumulative_returns = (1 + previous_dataframe_assets).cumprod() - 1
            evolution_df.loc[previous_date] = cumulative_returns.iloc[-1]
            print(f"\n Evolution of assets from {previous_date} to {date}: \n {evolution_df.loc[previous_date]}")
            print(f"Previous dataframe assets: {previous_dataframe_assets}")
            
            print("\n------------------Optimal Weights Evolution------------------")
            # Grab on the rebalanced weights form the beginning of the last period and evolve the assets until the end of the last period (for i == 0 the rebalanced weights are the target weights)
            new_contributions, new_contributions_normalized = pw.calculate_new_contributions(rebalanced_weights, evolution_df, date, previous_date)
            print(f"\n New contributions before normalization in the end of the period {previous_date}: \n {new_contributions}")
            print(f"\n Normalized new contributions in the end of the period {previous_date}: \n {new_contributions_normalized}")

            unbalanced_weights.loc[previous_date] = new_contributions_normalized
            print(f"\n Unbalanced weights in the end of the period {previous_date}: \n {unbalanced_weights.loc[previous_date]}")

            print("\n------------------Target Weights Evolution------------------")
            if weights_df.loc[previous_date].isnull().all():
                previous_weights = target_new_contributions_normalized # If the last weights were NaN, we use the target_new_contributions_normalized
                target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(target_new_contributions_normalized, evolution_df, date, previous_date, target=True)
            else:
                previous_weights = weights_df.loc[previous_date] # If we had signals in the last period, we use the weights_df
                target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(weights_df, evolution_df, date, previous_date)
            print(f"\n New contributions of the target before normalization in the end of the period {previous_date}: \n {target_new_contributions}")
            print(f"\n Normalized new contributions of the target in the end of the period {previous_date}: \n {target_new_contributions_normalized}")

            target_unbalanced_weights.loc[previous_date] = target_new_contributions_normalized
            print(f"\n Unbalanced weights of the target in the end of the period {previous_date}: \n {target_unbalanced_weights.loc[previous_date]}")
                
            # From 147 to 198: Prepare the beginning of the next period. Since the last period is over, we calculate the performance of the portfolio in the last period
            print("\n------------------Performances------------------")
            target_performance[previous_date] = calculate_cumulative_returns(previous_weights, previous_dataframe_assets)
            print(f"\n Target performance at {previous_date} before transaction costs: {target_performance[previous_date]} using the previous weights: {previous_weights}")

            performance_w_optimal_costs[previous_date] = calculate_cumulative_returns(rebalanced_weights.loc[previous_date].values, previous_dataframe_assets)
            print(f"\n Performance of the portfolio in the end of the period {previous_date} before transaction costs was {performance_w_optimal_costs[previous_date]} with the optimal weights: {rebalanced_weights.loc[previous_date]}")

            if i > 1:
                valid_weights = weights_df.dropna().loc[:dates_to_iterate[i - 1]]
                #print(f"\n Valid weights: \n {valid_weights}")

                if len(valid_weights) >= 2:
                    last_date = valid_weights.index[-1]
                    previous_target_unbalanced_weights = target_unbalanced_weights.loc[last_date - rebalancing_periods]
                    print(f"\n Due to the change in weights from {last_date - rebalancing_periods} to {last_date} we have:")

                    target_performance[previous_date], target_cost_of_transaction = tc.apply_transaction_cost(
                        target_performance[previous_date],
                        transaction_cost,
                        weights_df.loc[last_date].values,
                        previous_target_unbalanced_weights.values
                    )

                    total_target_cost_of_transaction += target_cost_of_transaction
                    print(f"\n Target performance at {previous_date} after transaction costs: {target_performance[previous_date]}")

                    previous_unbalanced_weights = unbalanced_weights.loc[last_date - rebalancing_periods]

                    performance_w_optimal_costs[previous_date], optimal_cost_of_transaction = tc.apply_transaction_cost(
                        performance_w_optimal_costs[previous_date],
                        transaction_cost,
                        rebalanced_weights.loc[last_date].values,
                        previous_unbalanced_weights.values
                    )

                    total_optimal_cost_of_transaction += optimal_cost_of_transaction
                    print(f"\n Performance of the portfolio in the end of the period {previous_date} after transaction costs: {performance_w_optimal_costs[previous_date]}")
                else:
                    print(f"\n Not enough valid dates to apply transaction costs")
            else:
                print(f"\n No transaction costs to apply in the first period")

            # In this part, we calculate the optimal delta (alpha) to prepare the rebalanced weights for the next period
            # i == 0 is the first period, so we don't have previous weights to calculate the transaction costs
            """ delta, crra, alpha = tc.optimize_delta_refined(
            weights_df.loc[date].values, unbalanced_weights.loc[previous_date].values, 
            trendy_assets, mean_reverting_assets, previous_dataframe_assets, 
            previous_date, date,
            distance_method=distance_method, gamma=gamma, delta_range=delta_range, initial_step=step, 
            transaction_cost=transaction_cost, verbose=verbose
            )  """
            # Funcao que calcula o CRRA para esse delta utilizando os retornos futuros
            CRRA_t, alpha = tc.calculate_crra(delta, weights_df.loc[date].values, unbalanced_weights.loc[previous_date].values,
            trendy_assets, mean_reverting_assets, previous_date, date, previous_dataframe_assets, distance_method=distance_method, gamma=gamma, transaction_cost=transaction_cost)
            
            CRRA_total += CRRA_t

            rebalanced_weights.loc[date] = pw.calculate_rebalanced_weights(alpha, weights_df.loc[date], unbalanced_weights.loc[previous_date])
            print(f"\n Combining the target weights at {date} and the unbalanced weights at {previous_date} with an alpha of {alpha} we have the rebalanced weights for the beginning of {date}: \n {rebalanced_weights.loc[date]}")

            #previous_weights = weights_df.loc[date]
            previous_assets_to_buy = assets_to_buy
        
        else:
            print(f"\n No signals to buy for {date}. Continuing with previous assets.")
            if i > 0:
                previous_date = dates_to_iterate[i - 1]
                _, previous_dataframe_assets, _ = analysis.filter_assets_data(close_df, previous_date, rebalancing_periods, previous_assets_to_buy, all_assets)

                # No signals
                weights_df.loc[date] = np.nan
                print(f"\n No signals generated: \n {weights_df}")
                print(f"\n Since we don't have signals, we keep the previous assets.")
                #print(f"\n To keep evolving the unbalanced weights {unbalanced_weights.loc[previous_date]} we need to calculate the evolution of the assets from {previous_date} to {date}")

                # Calculate the evolution of the assets from the previous date to the current date
                cumulative_returns = (1 + previous_dataframe_assets).cumprod() - 1
                evolution_df.loc[previous_date] = cumulative_returns.iloc[-1]
                print(f"\n Using the previous data frame assets: {previous_dataframe_assets}")
                print(f"\n The evolution of these assets from {previous_date} to {date} is: \n {evolution_df.loc[previous_date]}")


                print(f"\n------------------Optimal Weights Evolution------------------")
                # Calculate the new contributions in the end of the rebalancing period based on the evolution of the assets, even if there are signals or not
                new_contributions, new_contributions_normalized = pw.calculate_new_contributions(rebalanced_weights, evolution_df, date, previous_date)
                print(f"\n New contributions before normalization in the end of the period {previous_date}: \n {new_contributions}")
                print(f"\n Normalized new contributions in the end of the period {previous_date}: \n {new_contributions_normalized}")

                unbalanced_weights.loc[previous_date] = new_contributions_normalized
                print(f"\n Unbalanced weights in the end of the period {previous_date}: \n {unbalanced_weights.loc[previous_date]}")

                print(f"\n------------------Target Weights Evolution------------------")
                # condicao para ver se o weights df era nan na iteracao pssada, se for nan, usar o target new normalized, se nao for nan usar o proprio weight df
                if weights_df.loc[previous_date].isnull().all():
                    previous_weights = target_new_contributions_normalized # If the last weights were NaN, we use the target_new_contributions_normalized
                    target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(target_new_contributions_normalized, evolution_df, date, previous_date, target=True)
                else:
                    previous_weights = weights_df.loc[previous_date] # If we had signals in the last period, we use the weights_df
                    target_new_contributions, target_new_contributions_normalized = pw.calculate_new_contributions(weights_df, evolution_df, date, previous_date)
                print(f"\n New contributions of the target before normalization in the end of the period {previous_date}: \n {target_new_contributions}")
                print(f"\n Normalized new contributions of the target in the end of the period {previous_date}: \n {target_new_contributions_normalized}")

                target_unbalanced_weights.loc[previous_date] = target_new_contributions_normalized

                print(f"\n------------------Performances------------------")
                target_performance[previous_date] = calculate_cumulative_returns(previous_weights, previous_dataframe_assets)
                print(f"\n Target performance at {previous_date} before transaction costs: {target_performance[previous_date]} using the previous weights: {previous_weights}")

                performance_w_optimal_costs[previous_date] = calculate_cumulative_returns(rebalanced_weights.loc[previous_date].values, previous_dataframe_assets)
                print(f"\n Performance of the portfolio in the end of the period {previous_date} was {performance_w_optimal_costs[previous_date]} with the optimal weights: {rebalanced_weights.loc[previous_date]}")


                if i > 1:
                    valid_weights = weights_df.dropna().loc[:dates_to_iterate[i - 1]]

                    if len(valid_weights) >= 2 and weights_df.loc[dates_to_iterate[i - 1]].notna().all():
                        last_date = valid_weights.index[-1]
                        previous_target_unbalanced_weights = target_unbalanced_weights.loc[last_date - rebalancing_periods]
                        print(f"\n Due to the change in weights from {last_date - rebalancing_periods} to {last_date} we have:")

                        target_performance[previous_date], target_cost_of_transaction = tc.apply_transaction_cost(
                            target_performance[previous_date],
                            transaction_cost,
                            weights_df.loc[last_date].values,
                            previous_target_unbalanced_weights.values
                        )

                        total_target_cost_of_transaction += target_cost_of_transaction
                        print(f"\n Target performance at {previous_date} after transaction costs: {target_performance[previous_date]}")

                        previous_unbalanced_weights = unbalanced_weights.loc[last_date - rebalancing_periods]

                        performance_w_optimal_costs[previous_date], optimal_cost_of_transaction = tc.apply_transaction_cost(
                            performance_w_optimal_costs[previous_date],
                            transaction_cost,
                            rebalanced_weights.loc[last_date].values,
                            previous_unbalanced_weights.values
                        )

                        total_optimal_cost_of_transaction += optimal_cost_of_transaction
                        print(f"\n Performance of the portfolio in the end of the period {previous_date} after transaction costs: {performance_w_optimal_costs[previous_date]}")
                    else:
                        print(f"\n No transaction costs to apply")
                else:
                    print(f"\n No transaction costs to apply in the first period")

                # Since we don't have signals, we assume that the unbalanced weights are the rebalanced weights for the next period
                rebalanced_weights.loc[date] = unbalanced_weights.loc[previous_date]

                _, previous_dataframe_assets, previous_dataframe_assets_filtered = analysis.filter_assets_data(close_df, date, rebalancing_periods, previous_assets_to_buy, all_assets)
                #print(f"\n Preparing the previous dataframe assets for the next period: {previous_dataframe_assets}")        

            elif i == 0:
                #define rebalance weights as 1 for btc
                rebalanced_weights.loc[date] = np.zeros(len(all_assets))
                rebalanced_weights.loc[date, "BTC"] = 1
                weights_df.loc[date] = rebalanced_weights.loc[date]
                print(f"\n Initial rebalanced weights without signal at {date}: \n {rebalanced_weights.loc[date]}")

                previous_assets_to_buy = "BTC"
                continue

    return CRRA_total, rebalances_dates, target_performance, performance_w_optimal_costs, weights_df, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction

def get_portfolio_performance_best_delta(test_size, trendy_assets, mean_reverting_assets, buy_and_sell_signals_dict, close_df, rebalancing_periods=7, shorting_value=0.0, transaction_cost=0.01, distance_method="Manhattan", gamma=0, delta_range=(0, 3), initial_step=1, verbose=True):
    """
    Calculate the portfolio performance based in buy and sell signalling considering the rebalancing periods.

    Parameters:
    trendy_assets (list): The assets that are trending in each period
    mean_reverting_assets (list): The assets that are mean reverting in each period
    buy_and_sell_signals_dict (dict): The buy and sell signals for each asset
    close_df_train (pd.DataFrame): The close price of the assets for the training period
    close_df_test (pd.DataFrame): The close price of the assets for the testing period
    rebalancing_periods (int): Number of days to rebalance the portfolio
    shorting_value (float): The shorting value
    transaction_cost (float): The transaction cost
    distance_method (str): The distance method to use for the no-trade zone
    gamma (float): The factor of risk aversion for the CRRA function (0 for risk neutral)
    delta_range (tuple): The delta range for the optimization
    step (float): The step for the delta optimization
    verbose (bool): The verbose flag to print delta optimization 

    Returns:
    dict: The portfolio performance
    dict: The portfolio performance with costs
    pd.DataFrame: The target weights DataFrame
    pd.DataFrame: The evolution DataFrame
    pd.DataFrame: The new contributions DataFrame
    pd.DataFrame: The new contributions normalized DataFrame
    """

    if delta_range[0] >= delta_range[1]:
        raise ValueError("Invalid delta_range: start must be less than end.")

    train_keys, test_keys = train_test_split(sorted(buy_and_sell_signals_dict.keys()), test_size=test_size, shuffle=False)

    # Apply embargo
    train_limit = test_keys[0] - train_keys[0]

    train_keys = [k for k in train_keys if k <= train_limit]

    buy_and_sell_signals_train = {k: buy_and_sell_signals_dict[k] for k in train_keys}
    print(f"\n Buy and sell signals train: \n {buy_and_sell_signals_train}")
    buy_and_sell_signals_test = {k: buy_and_sell_signals_dict[k] for k in test_keys}
    print(f"\n Buy and sell signals test: \n {buy_and_sell_signals_test}")

    # NAO PRECISA POIS SO VAI ITERAR NAS DATAS DE REBALANCE DEFINIDAS PELO DICIONARIO BUY SELL
    #close_df_train = close_df.loc[:close_df.index[train_keys[-1]]]
    #print(f"\n Close prices train: \n {close_df_train.tail()}")
    #close_df_test = close_df.loc[close_df.index[test_keys[0]:]]
    #print(f"\n Close prices test: \n {close_df_test.head()}")
    
    best_delta = None
    best_crra = -np.inf
    step = initial_step
    previous_best_delta = None

    while step >= initial_step/1000:

        print(f"\n Delta range: {delta_range}")
        print(f"\n Step: {step}")

        deltas = np.arange(delta_range[0], delta_range[1], step)
        CRRA = np.zeros(len(deltas))

        for d, delta in enumerate(deltas):
            
            CRRA[d], *_ = get_portfolio_performance_delta(trendy_assets, mean_reverting_assets, buy_and_sell_signals_train, close_df, rebalancing_periods, shorting_value, transaction_cost, distance_method, gamma, delta, verbose)
            print(f"\n Delta: {delta} CRRA: {CRRA[d]}")
        max_index = np.argmax(CRRA)
        best_delta = deltas[max_index]
        best_crra = CRRA[max_index]

        if previous_best_delta is not None and best_delta == previous_best_delta:
            break

        previous_best_delta = best_delta
        delta_range = (max(delta_range[0], best_delta - step), min(delta_range[1], best_delta + step))
        step /= 10

    print(f"\n Best delta: {best_delta}")
    print(f"\n Best CRRA: {best_crra}")

    print("Train keys: ", train_keys)
    print(f"Test keys: ", test_keys)

    CRRA, rebalances_dates, target_performance_results, performance_w_optimal_costs, target_weights, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction = \
    get_portfolio_performance_delta(trendy_assets, mean_reverting_assets, buy_and_sell_signals_test, close_df, rebalancing_periods, shorting_value, transaction_cost, distance_method, gamma, best_delta, verbose)
        
    return best_delta, CRRA, rebalances_dates, target_performance_results, performance_w_optimal_costs, target_weights, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction
   
def get_benchmark_returns(benchmark_df, rebalance_dates):
    returns = []

    print(f"\n Benchmark returns: \n {benchmark_df}")

    rebalance_dates = pd.to_datetime(rebalance_dates).normalize()

    start_date = rebalance_dates[0]
    
    for i in range(len(rebalance_dates) - 1):
        end_date = rebalance_dates[i + 1] - pd.Timedelta(days=1)

        start_price = benchmark_df.loc[start_date]
        end_price = benchmark_df.loc[end_date]

        ret = (end_price - start_price) / start_price
        returns.append(ret)

        start_date = end_date
    
    return returns

def annualized_return():
    pass

def max_drawdown():
    pass

def sharpe_ratio():
    pass

'''Fill with the rest of the statistics'''

def fill_data_model_strategy_output(self):
    '''
    
    After running the strategy calculate all statistics and fill the data model:
    Strategy_Output to have acess to all info in an organized form
    
    '''