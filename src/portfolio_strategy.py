import sys
import os

# Adiciona o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime as dt
import enum
import numpy as np
import pandas as pd
import typing
import typing_extensions
import yahoofinancials as yf
import src.data_models as dm
import src.indicators as indicators
import src.data_analysis as analysis
import src.signalling as signalling
import src.metrics as metrics
import src.tc_optimization as tc
import src.data_utils as utils
import data.data_collection as dc
import src.visuals as visuals



from sklearn.model_selection import train_test_split

from src.backtesting.backtesting_test import Backtester

class Strategy_Portfolio:
    def __init__(self, 
                 start_date,
                 end_date,
                 Asset_Category: dm.Asset_Category,
                         Momentum_Type: dm.Momentum_Type,
                         Mean_Rev_Type: dm.Mean_Rev_Type,
                         Rebalancing_Period: dm.Rebalancing_Period,
                         Functional_Constraints: dm.Functional_Constraints,
                         Rebalance_Constraints: dm.Rebalance_Constraints,
                         Momentum_Days: int
                         ):
        self.start_date = start_date
        self.end_date = end_date
        self.asset_category = Asset_Category
        self.momentum_type = Momentum_Type
        self.mean_rev_type = Mean_Rev_Type
        self.rebalancing_period = Rebalancing_Period
        self.functional_constraints = Functional_Constraints
        self.rebalance_constraints = Rebalance_Constraints
        self.momentum_days = Momentum_Days

        self.transaction_costs = self.rebalance_constraints.transaction_cost
        self.hurst_exponents_period = self.functional_constraints.hurst_exponents_period
        self.momentum_threshold = self.functional_constraints.momentum_threshold

        self.rebalancing_period_days = analysis.get_rebalancing_period_days(self.rebalancing_period)

        self.portfolio_analysis_results = None

    def run_strategy(self):
        '''
        Function that implements all the logic to run the strategy, using all
        the functions from the other files

        Data Collection - Filtering - Signalling - Rebalancing - Statistics -
        Plots etc
        
        '''

        # Data Collection
        """ if self.asset_category == dm.Asset_Category.Top20CryptoByMarketCap:
            closes= dcollection.get_top20crypto()
        elif self.asset_category == dm.Asset_Category.SP500_Stocks:
            closes= dcollection.get_sp500_returns()
        elif self.asset_category == dm.Asset_Category.Crypto_and_Stocks:
            closes= dcollection.get_crypto_and_stocks_returns()
        else:
            closes= dcollection.get_crypto_prices(tickers= dm.Asset_Category.tickers, 
                                                start_date= self.start_date,
                                                end_date= self.end_date, 
                                                data_frequency= dm.ReturnsType.daily) """
        
        if self.asset_category == dm.Asset_Category.Top20CryptoByMarketCap:
            dcollection = dc.DataProcessor(self.asset_category, "BTC-USD", self.start_date)

        elif self.asset_category == dm.Asset_Category.SP500_Stocks:
            dcollection = dc.DataProcessor(self.asset_category, "^GSPC", self.start_date)
            
        closes = dcollection.get_asset_close_prices()            
        
        adjusted_start_date = self.start_date + pd.to_timedelta(self.hurst_exponents_period, unit='D')    
        # Benchmark
        """ benchmark = dcollection.get_returns_benchmark(benchmark='BTC',
                                                      start_date= adjusted_start_date,
                                                      end_date= self.end_date, 
                                                      data_frequency= dm.ReturnsType.daily) """
        
        benchmark = dcollection.get_benchmark_close_prices(
            start_date=adjusted_start_date
        )

            
        #print("Close prices DataFrame:\n", closes)

        # Run Portfolio Analysis

        portfolio_analysis_results = self.run_portfolio_analysis(
                                        benchmark,
                                        closes, 
                                        rebalancing_period= self.rebalancing_period_days, 
                                        hurst_exponents_period= self.hurst_exponents_period,
                                        transaction_costs= self.transaction_costs,
                                        functional_constraints= self.functional_constraints,
                                        momentum_days= self.momentum_days,
                                        distance_method= self.rebalance_constraints.distance_method,
                                        gamma= self.rebalance_constraints.gamma, 
                                        delta_range= self.rebalance_constraints.delta_range, 
                                        step= self.rebalance_constraints.delta_step,
                                        verbose=True,
                                        hurst_thresholds= self.functional_constraints.hurst_filter,
                                        mean_rev_type= self.mean_rev_type,
                                        momentum_type= self.momentum_type
                                        )
        
        print("Portfolio Analysis Results:\n", portfolio_analysis_results)

        self.portfolio_analysis_results = portfolio_analysis_results

        # Plotting
        visuals.plot_strategy_w_costs(portfolio_analysis_results, transaction_cost=self.transaction_costs)


    @staticmethod
    def run_portfolio_analysis(benchmark, close_df, rebalancing_period: int, hurst_exponents_period: int, transaction_costs: float, functional_constraints: dm.Functional_Constraints, momentum_days: int, distance_method = tc.DistanceMethod.EUCLIDEAN , gamma=0, delta_range=(-0.2, 2), step=0.2, verbose=True, hurst_thresholds=dm.HurstFilter.STANDARD, mean_rev_type=dm.Mean_Rev_Type.RSI, momentum_type=dm.Momentum_Type.MACD):
        """
        Execute the portfolio analysis.

        Parameters:
        benchmark (pd.DataFrame): The benchmark close prices to compare the portfolio
        close_df (pd.DataFrame): The close prices of the assets
        rebalancing_period (int): The rebalancing period
        transaction_costs (float): The transaction costs
        distance_method (str): The distance method to use
        gamma (float): The gamma value for the CRRA function
        delta_range (tuple): The delta range for the optimization
        step (float): The step for the optimization
        verbose (bool): The verbose flag to print delta optimization steps
        
        Returns:
        dict: The portfolio analysis results
        """

        test_size = 0.3

        #filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(close_df_train, rebalancing_period=rebalancing_period, hurst_exponents_period=hurst_exponents_period, mean_rev_type=mean_rev_type, momentum_type=momentum_type, functional_constraints = functional_constraints,
        #momentus_days_period = momentum_days), hurst_thresholds=hurst_thresholds, mean_rev_type= mean_rev_type, momentum_type= momentum_type)

        filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(close_df, rebalancing_period=rebalancing_period, hurst_exponents_period=hurst_exponents_period, mean_rev_type=mean_rev_type, momentum_type=momentum_type, functional_constraints = functional_constraints,
        momentus_days_period = momentum_days), hurst_thresholds=hurst_thresholds, mean_rev_type= mean_rev_type, momentum_type= momentum_type)
        
        buy_and_sells = signalling.buy_and_sell_signalling(filtered_data, mean_rev_type=mean_rev_type, momentum_type=momentum_type, functional_constraints=functional_constraints)
        
        #best_delta, CRRA, rebalances_dates, target_performance_results, performance_w_optimal_costs, target_weights, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction = \
        #metrics.get_portfolio_performance_best_delta(train_set_len, trendy_assets, mean_reverting_assets, buy_and_sells, close_df_train, close_df_test, rebalancing_periods=rebalancing_period, shorting_value=0.0, transaction_cost=transaction_costs, distance_method=distance_method, gamma=gamma, delta_range=delta_range, initial_step=step, verbose=verbose)
        
        best_delta, CRRA, rebalances_dates, target_performance_results, performance_w_optimal_costs, target_weights, evolution_df, new_contributions, new_contributions_normalized, unbalanced_weights, rebalanced_weights, total_optimal_cost_of_transaction, total_target_cost_of_transaction = \
        metrics.get_portfolio_performance_best_delta(test_size, trendy_assets, mean_reverting_assets, buy_and_sells, close_df, rebalancing_periods=rebalancing_period, shorting_value=0.0, transaction_cost=transaction_costs, distance_method=distance_method, gamma=gamma, delta_range=delta_range, initial_step=step, verbose=verbose)
        
        target_performance_results_df = pd.DataFrame(target_performance_results, index=[0])
        target_performance_cumulative_returns = (1 + target_performance_results_df).cumprod(axis=1)
        
        performance_w_optimal_costs_df = pd.DataFrame(performance_w_optimal_costs, index=[0])
        performance_cumulative_returns_w_optimal_costs = (1 + performance_w_optimal_costs_df).cumprod(axis=1)
        
        target_final_cumulative_return = target_performance_cumulative_returns.iloc[0, -1]
        target_expected_value = target_performance_results_df.mean(axis=1).iloc[0]
        
        final_cumulative_return_w_optimal_costs = performance_cumulative_returns_w_optimal_costs.iloc[0, -1]
        expected_value_w_optimal_costs = performance_w_optimal_costs_df.mean(axis=1).iloc[0]

        """ benchmark_returns = metrics.get_benchmark_returns(benchmark, rebalances_dates)
        benchmark_returns = pd.Series(benchmark_returns)
        #print("Benchmark Returns:\n", benchmark_returns)    
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() """
        
        if verbose:
            print('BENFICAAAA')
            utils.print_pretty_dic(target_weights)
            #print("Final target cumulative return:", target_final_cumulative_return)
            #print("Target Expected value:", target_expected_value)
            #print("Final cumulative return with costs:", final_cumulative_return_w_optimal_costs)
            #print("Expected value with costs:", expected_value_w_optimal_costs)
            #print("target_weights DataFrame:\n", target_weights)
            #print("unbalanced_weights DataFrame:\n", unbalanced_weights)
            print("rebalanced_weights DataFrame:\n", rebalanced_weights)
            #print("Evolution DataFrame:\n", evolution_df)
            #print("New Contributions DataFrame:\n", new_contributions)
            #print("New Contributions Normalized DataFrame:\n", new_contributions_normalized)
            print("Best Delta:", best_delta)
        
        return {
            "target_performance_results": target_performance_results,
            "target_performance_cumulative_returns": target_performance_cumulative_returns,
            "performance_w_optimal_costs": performance_w_optimal_costs,
            "performance_cumulative_returns_w_optimal_costs": performance_cumulative_returns_w_optimal_costs,
            "target expected_value": target_expected_value,
            "expected_value_w_optimal_costs": expected_value_w_optimal_costs,
            "target_weights": target_weights,
            "rebalanced_weights": rebalanced_weights,
            "total_optimal_cost_of_transaction": total_optimal_cost_of_transaction,
            "total_target_cost_of_transaction": total_target_cost_of_transaction,
            "rebalance_dates": rebalances_dates,
            #"evolution_df": evolution_df,
            #"new_contributions": new_contributions,
            #"new_contributions_normalized": new_contributions_normalized,
            #"unbalanced_weights": unbalanced_weights,
            "trendy_assets": trendy_assets,
            "mean_reverting_assets": mean_reverting_assets,
            #"benchmark_cumulative_returns": benchmark_cumulative_returns
        }

    def backtest(self):
        print("Backtesting the strategy")
        rebalanced_weights = self.portfolio_analysis_results["rebalanced_weights"]
        print("Rebalanced weights:\n", rebalanced_weights)
        index_ = self.portfolio_analysis_results["rebalance_dates"]
        print("Rebalance dates:\n", index_)

        df_positions = rebalanced_weights
        df_positions.reset_index(
            drop=True, inplace=True
        )  # Convert the existing index to a default range
        df_positions.index = pd.to_datetime(index_)  # Assign the new index
        print(df_positions)

        import ast

        # Load CSV
        if self.asset_category == dm.Asset_Category.Top20CryptoByMarketCap:
            df = pd.read_csv("data/crypto_data_scraping/crypto_prices_rank.csv")
        elif self.asset_category == dm.Asset_Category.SP500_Stocks:
            df = pd.read_csv("data/stock_data_scraping/sp500_data.csv")
        
        # Initialize dictionary
        data = {}

        # Process each cryptocurrency
        for col in df.columns[1:]:  # Skip 'Date' column

            def safe_eval(value, date):
                try:
                    ohlcv = ast.literal_eval(value)  # Convert string to list
                    if len(ohlcv) < 5:
                        raise ValueError("Not enough values")
                    return [date] + ohlcv[:5]  # Ensure exactly 6 values
                except (ValueError, SyntaxError, TypeError):
                    return [date, 0, 0, 0, 0, 0]  # Return a full row with zeros

            # Apply function and expand into separate columns
            expanded = df.apply(
                lambda row: pd.Series(safe_eval(row[col], row["Date"])), axis=1
            )

            # Rename columns
            expanded.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

            # Convert Date to datetime format
            expanded["Date"] = pd.to_datetime(expanded["Date"], errors="coerce")
            expanded.set_index("Date", inplace=True)

            # Store in dictionary
            data[col] = expanded

        bt = Backtester(data)
        bt.positions = df_positions
        bt.run_backtest()
        print(bt.calculate_metrics())
        bt.plot_results()

    def stress_test(self):
        pass

