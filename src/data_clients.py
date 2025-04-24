from datetime import datetime as dt
import pandas as pd
import typing
import ast

import yfinance  # type: ignore
import yfinance as yf  # type: ignore
import warnings

from binance.client import Client
import data_models as dm
import data_utils as utils

"""

This file is used to use the YahooFinance, Binance and other API that you wish 
to use.

"""


class YahooFinanceClient:
    """This class is a wrapper for the Yahoo Financials Module
    methods to create stock data extracts from the ticker or ticker list
    provided when instantiating the class.

    Args:
        :param list_of_tickers: list of tickers in the yahoo finance format
        :param start: start date in YYYY-MM-DD format
        :param end: end date in YYYY-MM-DD format
        :param date_interval: granularity, i.e. 'daily'
        :param outputs: if True returns the output, if False only records\
            data in instance
    """

    def __init__(
        self,
        asset_list: typing.List[dm.AssetID],
        start_date,
        end_date,
        data_frequency: dm.ReturnsType,
    ):
        self.ticker_list = [asset.ticker for asset in asset_list]
        self.parsed_ticker_list: typing.List[str]
        self.raw_data: dict
        self.start_date = utils.datetime_to_yf_format(start_date)
        self.end_date = utils.datetime_to_yf_format(end_date)
        self.data_frequency = data_frequency
        self._gather_raw_data()

    def _gather_raw_data(self) -> None:
        """
        Gather raw historical price data for the specified tickers from Yahoo Finance.

        Returns
        -------
        None
        """
        yahoo_data_handler = yf.YahooFinancials(self.ticker_list)
        data_dict = yahoo_data_handler.get_historical_price_data(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.data_frequency,
        )

        self.parsed_ticker_list = list(data_dict.keys())
        self.raw_data = data_dict

    def _get_raw_close_adjusted_prices(self) -> pd.DataFrame:
        """
        Extract close adjusted prices from the raw data and format them into a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing close adjusted prices for each ticker.
        """
        output = pd.DataFrame()
        for key in self.raw_data.keys():

            # Adjust the reference to correctly access 'prices'
            prices_data = self.raw_data[key].get(
                "prices"
            )  # Using .get to avoid KeyError
            if prices_data is None:
                raise KeyError(f"No 'prices' found for {key}")

            prices_df = pd.DataFrame(prices_data)
            temp = prices_df[["adjclose", "formatted_date"]]

            # Convert formatted_date to datetime
            temp["formatted_date"] = pd.to_datetime(
                temp["formatted_date"], yearfirst=True
            )
            temp.columns = [key, "Date"]

            if output.empty:
                output = temp
            else:
                output = pd.concat([output, temp[key]], axis=1)

        output.set_index("Date", drop=True, inplace=True)
        self.data = output
        return output

    def get_close_adjusted_prices(self) -> typing.List[dm.Returns]:
        """
        Retrieve close adjusted prices for the specified tickers.

        Returns
        -------
        typing.List[dm.Returns]
            A    list of Returns objects, each containing close adjusted prices for one ticker.
        """
        returns_list = self._get_raw_close_adjusted_prices()
        """for line in self._get_raw_close_adjusted_prices():
            
            returns_list.append(
                #data.Returns.from_pandas_dataframe(
                    #type=self.data_frequency,
                    #returns=line))"""
        # print(returns_list)
        return returns_list

    @staticmethod
    def valid_tickers(tickers: typing.List[str]) -> typing.List[str]:
        """
        Check the validity of tickers in a list.

        Parameters
        ----------
        tickers : typing.List[str]
            A list of ticker symbols to be validated.

        Returns
        -------
        typing.List[str]
            A list containing only the tickers that are valid and can be read from the Yahoo Finance API.

        Raises
        ------
        ValueError
            If no valid tickers are provided.

        Warnings
        --------
        UserWarning
            If a ticker is not found in the Yahoo Finance API.

        Notes
        -----
        This function checks all tickers in the provided list. If a ticker can be read from the Yahoo Finance API,
        it is considered valid and included in the returned list. If a ticker cannot be found, a warning is issued,
        and that ticker is excluded from the final list.
        """
        valid_tickers = []
        for t in tickers:
            df = yfinance.Ticker(t).history(period="1mo", interval="1d")
            if not df.empty:
                valid_tickers.append(t)
            else:
                warnings.warn("Ticker %s was not found!" % str(t))
        if len(valid_tickers) == 0:
            raise ValueError("No valid tickers provided!")
        return valid_tickers

    @staticmethod
    def get_ticker_name(ticker: str):
        """
        Retrieve the name of a ticker from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            The ticker symbol for which to get the name.

        Returns
        -------
        str
            The name of the ticker.
        """
        aux = yfinance.Ticker(ticker)
        name = aux.info["shortName"]
        return name


class BnBClient:
    """
    This class is a wrapper for the Binance API methods to create cryptocurrency data extracts from the ticker or ticker list
        provided when instantiating the class.

        Args:
            :param asset_list: list of AssetID objects representing the assets in the portfolio
            :param start_date: start date in YYYY-MM-DD format
            :param end_date: end date in YYYY-MM-DD format
            :param data_frequency: granularity, i.e. 'daily', 'monthly', or 'yearly'
    """

    def __init__(
        self,
        asset_list: typing.List[dm.AssetID],
        start_date,
        end_date,
        data_frequency: dm.ReturnsType,
    ):

        self.ticker_list = [asset.ticker for asset in asset_list]
        # self.parsed_ticker_list: typing.List[str]
        # self.raw_data: dict
        self.start_date = start_date
        self.end_date = end_date
        self.data_frequency = data_frequency

    def get_close_adjusted_prices(self) -> pd.DataFrame:
        """
        Retrieve close adjusted prices for the specified tickers from Binance.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the close adjusted prices for each ticker in the portfolio over the specified date range.
        """
        client = Client(
            "HaaNFyZgjudSuBi8ndlTFgC9wf39RSi9uIRCkLvquAuTxRrgpVxDzuKAstrRjY9l"
        )
        portfolio_returns = pd.DataFrame()

        # print(type(self.start_date), self.start_date[0])
        # print(type(self.end_date), self.end_date)

        start_date = self.start_date.strftime("%d %b, %Y")
        end_date = self.end_date.strftime("%d %b, %Y")

        freq = self.data_frequency
        freq_method_map = {
            dm.ReturnsType.daily: "KLINE_INTERVAL_1DAY",
            # dm.ReturnsType.monthly: 'KLINE_INTERVAL_1MONTH',
            # dm.ReturnsType.yearly: 'KLINE_INTERVAL_1Year'
        }
        interval = freq_method_map.get(freq)

        if interval:
            interval = getattr(client, interval)  # Access the attribute dynamically

        for ticker in self.ticker_list:
            data = client.get_historical_klines(
                f"{ticker}USDT", interval, start_date, end_date
            )
            df = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df.set_index("close_time", inplace=True)
            df[ticker] = pd.to_numeric(df["close"])

            portfolio_returns = pd.concat([portfolio_returns, df[ticker]], axis=1)

        return portfolio_returns

    def get_daily_tr(self):
        """
        Retrieves the daily Average True Range for the specified tickers from Binance.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the ATR for each ticker in the portfolio over the specified date range.
        """
        client = Client(
            "HaaNFyZgjudSuBi8ndlTFgC9wf39RSi9uIRCkLvquAuTxRrgpVxDzuKAstrRjY9l"
        )
        start_date = self.start_date.strftime("%d %b, %Y")
        end_date = self.end_date.strftime("%d %b, %Y")
        data_list = []

        freq = self.data_frequency
        freq_method_map = {
            dm.ReturnsType.daily: "KLINE_INTERVAL_1DAY",
            dm.ReturnsType.monthly: "KLINE_INTERVAL_1MONTH",
            dm.ReturnsType.yearly: "KLINE_INTERVAL_1Year",
        }
        interval = freq_method_map.get(freq)

        if interval:
            interval = getattr(client, interval)  # Access the attribute dynamically

        for ticker in self.ticker_list:
            klines = client.get_historical_klines(
                f"{ticker}USDT", interval, start_date, end_date
            )

            for kline in klines:
                # kline contains: [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
                # number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
                data_list.append(
                    {
                        "Ticker": ticker,
                        "Date": pd.to_datetime(
                            kline[0], unit="ms"
                        ).date(),  # Convert milliseconds to date
                        "Today High (H)": float(kline[2]),  # High price
                        "Today Low (L)": float(kline[3]),  # Low price
                        "Close (C)": float(kline[4]),  # Close price
                    }
                )

        # print(data_list)
        # Convert the data_list into a DataFrame
        df = pd.DataFrame(data_list)

        # Dictionary to store DataFrames for each ticker
        ticker_tr_dict = {}

        for ticker, group_df in df.groupby("Ticker"):
            # Set 'Date' as the index
            group_df.set_index("Date", inplace=True)

            # Calculate Previous Close
            group_df["Previous Close"] = group_df["Close (C)"].shift(1)

            # Calculate True Range (TR)
            group_df["True Range"] = group_df.apply(
                lambda row: max(
                    row["Today High (H)"] - row["Today Low (L)"],
                    abs(row["Today High (H)"] - row["Previous Close"]),
                    abs(row["Today Low (L)"] - row["Previous Close"]),
                ),
                axis=1,
            )

            # Store the daily TR as a Series in the dictionary
            ticker_tr_dict[ticker] = group_df["True Range"]

        # Create a DataFrame with the daily TR data, aligning by Date
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)

        # print(group_df)
        tr_df = pd.DataFrame(ticker_tr_dict)

        return tr_df

    @staticmethod
    def valid_tickers(tickers: typing.List[str]) -> typing.List[str]:
        """
        Validate a list of ticker symbols using the Binance API.

        Parameters
        ----------
        tickers : typing.List[str]
            A list of ticker symbols to be validated.

        Returns
        -------
        typing.List[str]
            A list containing only the valid ticker symbols found on Binance.

        Raises
        ------
        ValueError
            If no valid tickers are found.

        Warnings
        --------
        UserWarning
            If a ticker is not found on Binance or if an error occurs while checking a ticker.
        """

        if isinstance(tickers, str):
            try:
                tickers = ast.literal_eval(tickers)
            except (ValueError, SyntaxError):
                raise ValueError(
                    "Invalid tickers format. Expected a list of strings or a valid string representation of a list."
                )

        valid_tickers = []
        binance_client = Client(
            "HaaNFyZgjudSuBi8ndlTFgC9wf39RSi9uIRCkLvquAuTxRrgpVxDzuKAstrRjY9l"
        )  # Initialize Binance client

        for t in tickers:
            print(f"Checking ticker {t}")
            try:
                klines = binance_client.get_historical_klines(
                    symbol=f"{t}USDT", interval=Client.KLINE_INTERVAL_1DAY, limit=1
                )
                if klines:
                    valid_tickers.append(t)
                else:
                    warnings.warn(f"Ticker {t} was not found on Binance!")
            except Exception as e:
                warnings.warn(f"An error occurred while checking ticker {t}: {e}")

        if not valid_tickers:
            raise ValueError("No valid tickers provided!")

        return valid_tickers

    @staticmethod
    def get_ticker_name(ticker: str) -> str:
        """
        Get the name of a ticker from Binance.

        Parameters
        ----------
        ticker : str
            The ticker symbol for which to get the name.

        Returns
        -------
        str
            The name of the ticker.

        Raises
        ------
        ValueError
            If no information is found for the ticker or if an error occurs while retrieving the ticker information.
        """
        binance_client = Client(
            "HaaNFyZgjudSuBi8ndlTFgC9wf39RSi9uIRCkLvquAuTxRrgpVxDzuKAstrRjY9l"
        )  # Initialize Binance client

        try:
            ticker_info = binance_client.get_symbol_info(f"{ticker}USDT")
            if ticker_info:
                name = ticker_info["baseAsset"]
                return name
            else:
                raise ValueError(
                    f"No information found for ticker {ticker} on Binance!"
                )
        except Exception as e:
            raise ValueError(
                f"An error occurred while getting ticker name for {ticker}: {e}"
            )