import data_clients as dc
import data_models as dm
import data_utils as utils
from datetime import datetime as dt
import pandas as pd
import yahoofinancials as yf
import numpy as np

import indicators as indicator


def get_sigmoid_momentum_signals(close_prices_df, daily_tr_df, 
                                 asset_ids: list[dm.AssetID],start_date: dt,
                            end_date: dt, data_frequency= dm.ReturnsType.daily,         
                exp_decay= 0.2,prices_window=21, ATR_window=50, decay= 0.2):
    """
    Calculate sigmoid momentum signals based on price moving averages and ATR.

    Parameters:
        close_prices_df (pd.DataFrame): DataFrame with close prices.
        daily_tr_df (pd.DataFrame): DataFrame with daily true range values.
        asset_ids (list[dm.AssetID]): List of asset IDs.
        start_date (dt): Start date for the data.
        end_date (dt): End date for the data.
        data_frequency (dm.ReturnsType): Frequency of the data (daily, weekly, etc.).
        exp_decay (float): Exponential decay for moving averages.
        prices_window (int): Window size for price moving averages.
        ATR_window (int): Window size for ATR calculation.
        decay (float): Decay factor for ATR calculation.

    Returns:
        pd.DataFrame: DataFrame with sigmoid momentum signals.
    """
    # Calculate exponential moving average of prices
    prices_average = indicator.prices_moving_average(
        close_prices_df.iloc[-prices_window:], prices_window, 
        ma_type='exponential', exp_decay=decay)
      
    # Calculate the difference between the latest close price and the moving average
    close_minus_average= close_prices_df.iloc[-1:]- prices_average.iloc[-1:]
   
    # Calculate ATR (Average True Range)
    ATR_df = indicator.get_ATR(asset_ids, start_date,
                               end_date, data_frequency, 
                               daily_tr_df.iloc[-ATR_window:],decay= exp_decay)

    not_sigmoid_momentum= close_minus_average/ATR_df.iloc[-1:]

    sigmoid_momentum= indicator.sigmoid(not_sigmoid_momentum)

    sigmoid_momentum

    return sigmoid_momentum 


'''
Ass you can see this function is far from being completely implemented so lot's 
of work here

The treshold is 0.73 because a non sigmoid signal of 1 is equal to a 0.7311 
sigmoid signal

This also need change because you'll have MACD momentum and sigmoid momentum to choose
'''

def buy_and_sell_signalling(data_filtered, mean_rev_type = dm.Mean_Rev_Type.RSI, momentum_type = dm.Momentum_Type.MACD, functional_constraints = dm.Functional_Constraints): # mudar para 0.1 e colocar como parametro do example e live
    """
    Generates buy and sell signals based on momentum and Bollinger Bands.

    Parameters:
        data_filtered (dict): Filtered data dictionary with trendy and mean-reverting assets.
        momentum_threshold (float): Threshold for generating buy/sell signals for trendy assets.

    Returns:
        dict: Dictionary with buy and sell signals for each rebalance date.
    """
    buy_and_sell_signals = {}

    for i, data in data_filtered.items():
        buy_and_sell_signals[i] = {'buy': [], 'sell': []}
        trendy_assets = data['trendy_assets']
        mean_reverting_assets = data['mean_reverting_assets']
        
        # Process trendy assets
        if momentum_type == dm.Momentum_Type.Cumulative_Returns:
            #por enquanto so cum rets
            for asset, momentum in trendy_assets:
                if momentum > functional_constraints.get_momentum_threshold(): # buy nao significa aumentar o peso da moeda, mas que queremos ter long position, o peso vem depois
                    buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset) 
                elif momentum < -np.inf: #np.inf pra evitar short selling
                #elif momentum < -momentum_threshold: #np.inf pra evitar short selling
                    buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
        
        elif momentum_type == dm.Momentum_Type.MACD:
            for asset, macd_values in trendy_assets:
                close_price = next((tuple[1] for tuple in data['close_price'] if tuple[0] == asset), None)
                macd = macd_values[1]
                histogram = macd_values[3]
                hist_diff = macd_values[4]
                lower_threshold = macd_values[5]
                upper_threshold = macd_values[6]
                hist_trend = macd_values[7]
                if 0 < macd and lower_threshold < histogram < upper_threshold and hist_trend > 0: # nosso
                #if histogram > 0 and hist_diff > 0 and lower_threshold < histogram < upper_threshold: # indicator seasons colby
                    #print("close_price:", close_price, "macd_values[3]==histogram:", macd_values[3], "macd_values[5]==threshold:", macd_values[5])
                    buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset)
                elif macd_values[3] < -np.inf or macd_values[4] < -np.inf:
                    buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
            
        # Process mean-reverting assets
        if mean_rev_type == dm.Mean_Rev_Type.Bollinger_Bands:
            for asset, bollinger_bands in mean_reverting_assets:
                close_price = next((tuple[1] for tuple in data['close_price'] if tuple[0] == asset), None)
                upper_band, _, lower_band = bollinger_bands
                if close_price > upper_band:
                    buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
                elif close_price < lower_band:
                    buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset)
        
        elif mean_rev_type == dm.Mean_Rev_Type.RSI:
            for asset, rsi in mean_reverting_assets:
                #print("mean_reverting_assets", mean_reverting_assets)
                #print("rsi", rsi)
                #print("rsi_overbought", functional_constraints.rsi_overbought)
                if rsi > functional_constraints.rsi_overbought:
                    buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
                elif rsi < functional_constraints.rsi_oversold:
                    buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset)
        
    return buy_and_sell_signals

def sigmoid_buy_and_sell_signalling():
    pass

def macd_buy_and_sell_signalling():
    pass

def RSI_buy_and_sell_signalling(data_filtered):

    buy_and_sell_signals = {}

    for i, data in data_filtered.items():
        buy_and_sell_signals[i] = {'buy': [], 'sell': []}
        mean_reverting_assets = data['mean_reverting_assets']

        # Process mean-reverting assets
        for asset, rsi in mean_reverting_assets:
            if rsi > 70:
                buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
            elif rsi < 30: # Conectar isto ao datamodels
                buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset)


def bollinger_buy_and_sell_signalling(data_filtered):

    buy_and_sell_signals = {}

    for i, data in data_filtered.items():
        buy_and_sell_signals[i] = {'buy': [], 'sell': []}
        mean_reverting_assets = data['mean_reverting_assets']
        
        # Process mean-reverting assets
        for asset, bollinger_bands in mean_reverting_assets:
            close_price = next((tuple[1] for tuple in data['close_price'] if tuple[0] == asset), None)
            upper_band, _, lower_band = bollinger_bands
            if close_price > upper_band:
                buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
            elif close_price < lower_band:
                buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset)
    
    return buy_and_sell_signals

def cumrets_mom_buy_and_sell_signalling(data_filtered, momentum_threshold=0.01): # mudar para 0.1 e colocar como parametro do example e live
    """
    Generates buy and sell signals based on momentum and Bollinger Bands.

    Parameters:
        data_filtered (dict): Filtered data dictionary with trendy and mean-reverting assets.
        momentum_threshold (float): Threshold for generating buy/sell signals for trendy assets.

    Returns:
        dict: Dictionary with buy and sell signals for each rebalance date.
    """
    buy_and_sell_signals = {}

    for i, data in data_filtered.items():
        buy_and_sell_signals[i] = {'buy': [], 'sell': []}
        trendy_assets = data['trendy_assets']
        
        # Process trendy assets
        for asset, momentum in trendy_assets:
            if momentum > momentum_threshold: # buy nao significa aumentar o peso da moeda, mas que queremos ter long position, o peso vem depois
                buy_and_sell_signals.setdefault(i, {}).setdefault('buy', []).append(asset) 
            elif momentum < -np.inf: #np.inf pra evitar short selling
            #elif momentum < -momentum_threshold: #np.inf pra evitar short selling
                buy_and_sell_signals.setdefault(i, {}).setdefault('sell', []).append(asset)
    
    return buy_and_sell_signals