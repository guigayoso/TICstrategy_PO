import data_models as dm
import data_utils as utils
from datetime import datetime as dt
import pandas as pd
import yahoofinancials as yf
import numpy as np
from hurst import compute_Hc

'''

File to calculate all indicators Hurst, ATR, sigmoid, Bollinger Bands, in the 
future MACD and others

'''

def calculate_hurst_exponent(df):
    """
    Calculate the Hurst exponent for each column in the dataframe.
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and adjusted close prices for different assets
    Returns:
    pd.Series: Series containing Hurst exponents for each column in the dataframe
    """
    hurst_values = df.apply(lambda x: compute_Hc(x, kind='price',simplified=False)[0])
    return hurst_values

def get_ATR(asset_ids: list[dm.AssetID],start_date: dt,
                    end_date: dt, data_frequency= dm.ReturnsType.daily,
                    exp_decay= 0.2):
    
    tr_data = dc.BnBClient(
                asset_ids,
                start_date, 
                end_date, 
                data_frequency
            ).get_daily_tr()
    
    tr_df = pd.DataFrame(tr_data)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    tr_df.index = dates

    daily_atr= tr_df.ewm(alpha= exp_decay, adjust=False).mean()
    return daily_atr

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_bollinger_bands(df, num_std=2):
    """
    Calculate Bollinger Bands for each column in the dataframe.
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and adjusted close prices for different assets
    window (int): The period for the moving average and standard deviation calculation
    num_std (int): Number of standard deviations to use for the bands
    Returns:
    pd.DataFrame: DataFrame containing the Bollinger Bands (upper, middle, lower) for each column
    """
    mean = df.mean()
    std = df.std()
    
    bollinger_upper = mean + (std * num_std)
    bollinger_middle = mean
    bollinger_lower = mean - (std * num_std)
    
    bollinger_bands = pd.DataFrame({
        'upper': bollinger_upper,
        'middle': bollinger_middle,
        'lower': bollinger_lower
    })
    
    return bollinger_bands

def calculate_RSI(df, window= 5):
    """
    Calculate the Relative Strength Index (RSI) for each column in the
    dataframe.
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and adjusted close prices for different assets
    window (int): The period for the RSI calculation
    Returns:
    pd.DataFrame: DataFrame containing the RSI for each column
    """

    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    #gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    #loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    
    return RSI

def prices_moving_average(prices_df, ma_window, ma_type= 'exponential', exp_decay= 0.2):
    if ma_type== 'simple':
        prices_moving_average = prices_df.rolling(window=ma_window).mean()
    if ma_type == 'exponential':
        #Currently not moving average, just average
        prices_moving_average= prices_df.ewm(alpha=exp_decay, adjust= False).mean()
    return prices_moving_average

def momentum_n_days(close_df,momentum_n_days=30):
    """
    Calculate cumulative momentum for the last `momentum_n_days` for each asset.

    Parameters:
        close_df (pd.DataFrame): DataFrame with adjusted close prices for different assets.
        momentum_n_days (int): Number of days to consider for momentum calculation.

    Returns:
        list: List of cumulative momentum values for each column in the DataFrame.
    """
    cumulative_returns = [(close_df[col].iloc[-momentum_n_days:].pct_change().add(1).cumprod().iloc[-1] - 1) for col in close_df.columns]
    return cumulative_returns

def calculate_macd(close, short_window=12, long_window=26, signal_window=9):
 
    macd_results = {}

    for col in close.columns: 
        short_ema = close[col].ewm(span=short_window, adjust=False).mean()
        long_ema = close[col].ewm(span=long_window, adjust=False).mean()

        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        histogram = macd - signal_line

        positive_histogram = histogram[histogram > 0]
        sorted_positive_histogram = positive_histogram.sort_values(ascending=True)

        percentile = 0.85
        #threshold_index = int(len(sorted_positive_histogram) * percentile)
        #threshold = sorted_positive_histogram[threshold_index]

        lower_thresold_index = int(len(sorted_positive_histogram) * (percentile-0.8499)) # Resultados do csv usando -0.45 e +0.1499 do 0.85
        lower_threshold = sorted_positive_histogram[lower_thresold_index]

        upper_threshold_index = int(len(sorted_positive_histogram) * (percentile-0.15))
        upper_threshold = sorted_positive_histogram[upper_threshold_index]

        gradient = np.gradient(histogram)
        trend = np.mean(gradient[-3:])

        hist_diff = histogram.diff()

        # Adiciona o ultimo valor de histograma ao dicionario
        macd_results[col] = {
            'macd': macd.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'hist_diff': hist_diff.iloc[-1],
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'trend': trend
        }

    macd_df = pd.DataFrame(macd_results)

    return macd_df
