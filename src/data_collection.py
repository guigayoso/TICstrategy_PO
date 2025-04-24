import data_clients as dc
import data_models as dm
import data_utils as utils
from datetime import datetime as dt
import pandas as pd
import yahoofinancials as yf
import numpy as np
'''

These are example funtions to show you how to store  information in the data
models while collecting returns. You'll have to adapt this function to use the 
method from data model dm.Asset_Category that gives the tickers of the biggest
20 crypto by market cap and the sp500 tickers.

'''

# dt is this format dt(2019,1,1)

def get_stocks_prices(tickers, start_date: dt,
                    end_date: dt, data_frequency= dm.ReturnsType.daily):
    
    asset_ids = [
            dm.AssetID(type=dm.AssetType.stock, ticker=ticker, 
                        name= dc.YahooFinanceClient.get_ticker_name(ticker))
            for ticker in dc.YahooFinanceClient.valid_tickers(tickers)
            ]
    prices_data = dc.YahooFinanceClient(
            asset_ids,
            start_date, 
            end_date, 
            data_frequency
        ).get_close_adjusted_prices()
    
    close_df = pd.DataFrame(prices_data)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    close_df.index = dates
    
    return close_df

def get_crypto_prices(tickers, start_date: dt, end_date: dt,
                       data_frequency= dm.ReturnsType.daily):
    asset_ids = [
                dm.AssetID(type=dm.AssetType.cryptocurrency, ticker=ticker, 
                            name= dc.BnBClient.get_ticker_name(ticker))
                for ticker in dc.BnBClient.valid_tickers(tickers)
                ]
    prices_data = dc.BnBClient(
                asset_ids,
                start_date, 
                end_date, 
                data_frequency
            ).get_close_adjusted_prices()
    
    close_df = pd.DataFrame(prices_data)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    close_df.index = dates
            
    return close_df

def get_top20crypto():
    pass

def get_sp500_returns():
    pass

def get_crypto_and_stocks_returns():
    pass

def get_returns_benchmark(benchmark, start_date, end_date, data_frequency):
    def get_single_benchmark_returns(benchmark):
        if benchmark == 'BTC':
            Benchmark_id = [dm.AssetID(type=dm.AssetType.cryptocurrency, ticker=benchmark, name=dc.BnBClient.get_ticker_name(benchmark))]
            prices_data = dc.BnBClient(
                Benchmark_id,
                start_date,
                end_date,
                data_frequency
            ).get_close_adjusted_prices()

            close_df = pd.DataFrame(prices_data)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            close_df.index = dates

            return close_df
        
        elif benchmark == 'SP500':
            Benchmark_id = [dm.AssetID(type=dm.AssetType.stock, ticker='SPY', name='S&P 500')]
            prices_data = dc.YahooFinanceClient(
                Benchmark_id,
                start_date,
                end_date,
                data_frequency
            ).get_close_adjusted_prices()
            
            close_df = pd.DataFrame(prices_data)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            close_df.index = dates

            return close_df

    if isinstance(benchmark, list):
        all_returns = {}
        for bm in benchmark:
            all_returns[bm] = get_single_benchmark_returns(bm)
        return pd.DataFrame(all_returns)
    else:
        return get_single_benchmark_returns(benchmark)
