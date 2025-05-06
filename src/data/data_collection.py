from data.crypto_data_scraping.get_crypto_data import CryptoDataProcessor
from data.stock_data_scraping.get_stock_data import StockDataProcessor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_models as dm
import pandas as pd
from datetime import datetime as dt
import yfinance as yf
import ast  # Used to evaluate strings as Python literals
import sys
import os

class DataProcessor:
    crypto_csv_path = "data/crypto_data_scraping/crypto_prices_rank.csv"
    #stocks_csv_path = "data/stock_data_scraping/stocks_prices_rank.csv"
    stocks_csv_path = "data/stock_data_scraping/sp500_data.csv"
    stocks_evo_path = "data/stock_data_scraping/sp500_evolution.csv"

    def __init__(
        self, asset_category, benchmark_ticker=None, start_date=dt(2019, 1, 1)
    ):
        self.last_asset_date = None
        self.last_benchmark_date = None
        self.df_asset_existing = None
        self.df_benchmark_existing = None
        self.asset_category = asset_category
        self.start_date = start_date
        self.benchmark_ticker = benchmark_ticker

        self.crypto_data_processor = CryptoDataProcessor()
        self.stocks_data_processor = StockDataProcessor()

        print(asset_category)
        if asset_category == dm.Asset_Category.Top20CryptoByMarketCap:
            self._update_crypto_csv()
        elif asset_category == dm.Asset_Category.SP500_Stocks:
            self._update_stocks_csv()
        elif asset_category == dm.Asset_Category.Crypto_and_Stocks:
            self._update_crypto_csv()
            self._update_stocks_csv()
        else:
            raise ValueError("CUnknown Asset Category")

    def _update_crypto_csv(self):
        self.crypto_data_processor.run()

    def _update_stocks_csv(self):
        self.stocks_data_processor.run()

    def _update_dataframe(self):
        pass

    def _get_asset_data(self):
        if self.last_asset_date == dt.today():
            return self.df_asset_existing

        if self.asset_category == dm.Asset_Category.Top20CryptoByMarketCap:
    
        # Load CSV into a DataFrame and parse dates
            self.df_asset_existing = pd.read_csv(
                DataProcessor.crypto_csv_path,
                parse_dates=["Date"],  # Ensures the Date column is read as datetime
                index_col="Date",  # Sets Date as the DataFrame index
            )

        elif self.asset_category == dm.Asset_Category.SP500_Stocks:
            # Load CSV into a DataFrame and parse dates
            self.df_asset_existing = pd.read_csv(
                DataProcessor.stocks_csv_path,
                parse_dates=["Date"],  # Ensures the Date column is read as datetime
                index_col="Date",  # Sets Date as the DataFrame index
            )

        # Update the last asset date
        self.last_asset_date = self.df_asset_existing.index[-1]

        # Return data from the start date up to today
        return self.df_asset_existing[self.df_asset_existing.index >= self.start_date]

    def _get_benchmark_data(self, start_date=None):
        if self.benchmark_ticker is None:
            raise ValueError(
                "No benchmark provided. Set it using the set_benchmark method"
            )

        if self.last_benchmark_date == dt.today():
            return self.df_benchmark_existing

        if start_date is None:
            start_date = self.start_date

        # Download the benchmark data from Yahoo Finance - TO BE CHANGED LATER!
        self.df_benchmark_existing = yf.download(
            self.benchmark_ticker, start=start_date
        )
        self.last_benchmark_date = dt.today()

        return self.df_benchmark_existing

    def get_asset_close_prices(self):
        """Extracts Close prices (fourth element in each list) for each asset and returns a new DataFrame with Date as index."""
        df = self._get_asset_data()

        # Ensure 'Date' is a column before setting it as an index
        if df.index.name == "Date":
            df = df.reset_index()

        # Initialize new DataFrame with Date as index
        close_prices = pd.DataFrame(index=df["Date"])

        for col in df.columns[1:]:  # Skip 'Date'
            new_col = []
            for row in df[col]:
                try:
                    parsed = ast.literal_eval(row)
                    if isinstance(parsed, list):
                        value = float(parsed[3])
                        new_col.append(value)
                    else:
                        new_col.append(None)
                except (ValueError, SyntaxError):
                        print(f"Error to process: {row}")
                        new_col.append(None)

            close_prices[col] = new_col

        return close_prices.dropna(axis=1, thresh=len(close_prices)).fillna(
            method="ffill"
        )  # Drop columns that dont have 2 datapoints - SEE LATER!

    def get_benchmark_close_prices(self, start_date=None):
        """Extracts the Close prices for the benchmark and returns a new DataFrame."""
        df = self._get_benchmark_data(start_date=start_date)
        return df["Close"]