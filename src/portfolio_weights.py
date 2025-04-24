import data_clients as dc
import data_models as dm
import data_utils as utils
from datetime import datetime as dt
import pandas as pd
import yahoofinancials as yf
import numpy as np
import signalling as signal
import indicators as indicator
import portfolio_strategy as strategy

'''

Currently almost everything is working without data models.
But what you have to do is to use the first only if dm.Weight_Allocation is 
single strenght and develop an alternative to equal-weight

'''

def assign_portfolio_weights(buy_and_sell_signals):
    key = list(buy_and_sell_signals.keys())[0]  # There's only one key
    buy_list = buy_and_sell_signals[key]['buy']
    
    # Calculate the total for normalization
    denominator = sum(item[1] for item in buy_list)

    # Create a list of tuples for the weights
    weight_list = [(item[0], item[1] / denominator) for item in buy_list]
    
    return weight_list


def calculate_uniform_weights(assets_to_buy, assets_to_sell, shorting_value):
    """Calculate the weights for the assets to buy and sell.

    Parameters:
    assets_to_buy (list): The assets to buy
    assets_to_sell (list): The assets to sell
    shorting_value (float): The shorting value

    Returns:
    list: The joined weights array (buy and sell)
    list: The assets selected"""
    no_assets_to_buy = len(assets_to_buy)
    no_assets_to_sell = len(assets_to_sell)

    buy_array = [(1 + shorting_value) / no_assets_to_buy] * no_assets_to_buy if no_assets_to_buy else []
    sell_array = [-shorting_value / no_assets_to_sell] * no_assets_to_sell if no_assets_to_sell else []

    joined_weights_array = buy_array + sell_array
    assets_selected = assets_to_buy + assets_to_sell
    return buy_array, joined_weights_array, assets_selected, assets_to_buy


def calculate_new_contributions(weights, evolution_df, date, previous_date, target = False):
    """
    Calculate the new asset contributions based on the evolution of the assets.

    Parameters:
    weights (pd.DataFrame): The weights in the beggining of the period.
    evolution_df (pd.DataFrame): The evolution of the assets during the previous rebalancing period.
    date (int): The current date.
    previous_date (int): The previous date to calculate the new contributions.
    
    Returns:
    pd.Series: The new contributions.
    pd.Series: The new contributions normalized.
    """
    # Calcular as novas contribuições da linha anterior
    #new_contributions = (rebalanced_weights.loc[previous_date] * (1 + evolution_df.loc[previous_date]))

    if target:
        #print(f" \n Parameters used in calculate_new_contributions: \n{weights}, {evolution_df.loc[previous_date]}")
        new_contributions = (weights * (1 + evolution_df.loc[previous_date]))
    else:
        #print(f"\n Parameters used in calculate_new_contributions: \n{weights.loc[previous_date]}, {evolution_df.loc[previous_date]}")
        new_contributions = (weights.loc[previous_date] * (1 + evolution_df.loc[previous_date]))

    # Normalizar as novas contribuições
    new_contributions_normalized = new_contributions / new_contributions.sum()
    
    return new_contributions.fillna(0), new_contributions_normalized.fillna(0)



def update_weights_df(weights_df, date, assets_to_buy, buy_array):
    """Update the weights DataFrame with the new weights.

    Parameters:
    weights_df (pd.DataFrame): The DataFrame containing the target weights for each asset 
    date (int): The date to update the weights
    assets_to_buy (list): The list of assets to buy
    buy_array (list): The corresponding array of weights for the assets to buy
    """
    
    # If the buy_array is empty, set the weights to 0, otherwise update the weights 
    if len(buy_array) != 0:
        for asset, weight in zip(assets_to_buy, buy_array):
            weights_df.at[date, asset] = weight
    else:
        weights_df.loc[date] = {asset: 0 for asset in weights_df.columns}


def calculate_rebalanced_weights(alpha, w_target, w_unbalanced):
    """Calculate the rebalanced weights based on alpha.

    Parameters:
    alpha (float): The proportion of the target weights for the next rebalancing period
    w_target (np.array): The target weights defined by the strategy
    w_unbalanced (np.array): The contribution of each asset in the end of the period
    
    Returns:
    np.array: The rebalanced weights
    """
    return alpha * w_unbalanced + (1 - alpha) * w_target


def assign_equal_weights():
    pass

def assign_proportional_weights():
    pass

def rebalance_portfolio(inputs=None, ideal_weights=None):
    if inputs is None:
        from portfolio_strategy import Strategy_Portfolio
        inputs = Strategy_Portfolio
    if ideal_weights is None:
        ideal_weights = assign_equal_weights()
    pass