import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from time import sleep
from bs4 import BeautifulSoup
import yfinance as yf


"""
TODO: ADD THE "DATE" COLUMNS TO THE CRYPTO_PRICES CSV
"""

class CryptoDataProcessor:
    def __init__(self, start_year=2019, prices_filename="data/crypto_data_scraping/crypto_prices_rank.csv"):
        self.start_year = start_year
        self.prices_filename = prices_filename
        self.symbol_mapping = {
            "MIOTA": "IOTA",
            "EMPR": "MPWR",
            "MEXC": "MEX",
            "999": "G999",
            "TAGZ5": "TAG",
            "INNBCL": "INNBC",
        }
        self.binance_exceptions = {
            "USDT",
            "LEO",
            "BGB",
            "UST",
            "BSV",
            "XMR",
            "XEM",
            "WAVES",
            "MPWR",
            "CRO",
            "HT",
            "MEX",
            "G999",
            "MIN",
            "STO",
            "TAG",
            "INNBC",
            "HEX",
            "VEST",
            "BUSD",
            "DAI",
            "MOF",
            "REV",
            "KLAY",
            "BTT",
            "TRX",
            "MATIC",
            "DASH",
            "MKR",
            "ZEC",
            "BCH",
            "XTZ",
            "WBTC",
            "FIL",
            "LUNC",
            "USDC",
            "TON",
            "POL",
            "HYPE",
        }
        self.impossible_tickers = {"MPWR", "G999", "ACA", "COMP", "POL", "HYPE"}
        self.df_existing = self._load_existing_data()

    def _load_existing_data(self):
        if os.path.exists(self.prices_filename):
            return pd.read_csv(self.prices_filename, index_col=0, parse_dates=True)
        return pd.DataFrame()

    def run(self):
        last_scraped_date = self.df_existing.index.max()

        if isinstance(
            last_scraped_date, str
        ):  # Convert string to datetime if necessary
            last_scraped_date = datetime.strptime(last_scraped_date, "%Y-%m-%d")

        if pd.isna(
            last_scraped_date
        ):  # If no existing data, start from the default date
            last_scraped_date = datetime(self.start_year - 1, 12, 31)

        dates = pd.date_range(
            start=last_scraped_date + timedelta(days=1), end=datetime.today(), freq="D"
        )

        for date in dates:
            date_str = date.strftime("%Y%m%d")
            url = f"https://coinmarketcap.com/historical/{date_str}/"
            rankings = self._fetch_rankings(date, url)
            if rankings:
                self._fetch_and_save_prices(date, rankings)

    def _fetch_rankings(self, date, url):
        retries, success = 3, False
        while retries > 0 and not success:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 429:
                    print(
                        f"⚠️ Rate limit exceeded for {date}. Retrying in 60 seconds..."
                    )
                    sleep(60)
                    continue
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch data for {date}")

                soup = BeautifulSoup(response.text, "html.parser")
                rows = soup.find_all("tr", class_="cmc-table-row")
                rankings = {}
                for row in rows:
                    columns = row.find_all("td")
                    if len(columns) >= 10:
                        rank = int(columns[0].text.strip())
                        symbol = self.symbol_mapping.get(
                            columns[2].text.strip(), columns[2].text.strip()
                        )
                        rankings[symbol] = rank
                success = True
                return rankings
            except Exception as e:
                print(f"⛑️ Error fetching rankings for {date}: {e}")
                retries -= 1
                sleep(10)
        return None

    def _fetch_and_save_prices(self, date, rankings):
        row_data = {}
        for symbol, rank in rankings.items():
            if rank == -1 or symbol in self.impossible_tickers:
                continue
            try:
                price_data = self._get_price_data(symbol, date)
                if price_data:
                    row_data[symbol] = price_data + [rank]
            except Exception as e:
                print(f"⛑️ Error fetching {symbol} price data on {date}: {e}")
                continue

        if row_data:
            df = pd.DataFrame.from_dict({date: row_data}, orient="index")
            self.df_existing = pd.concat([self.df_existing, df])
            self.df_existing.to_csv(self.prices_filename)
            print(f"✅ Saved data for {date}")

    def _get_price_data(self, symbol, date):
        if symbol in self.binance_exceptions:
            ohlcv = yf.Ticker(f"{symbol}-USD").history(
                start=date, end=date + timedelta(days=1)
            )
            if ohlcv.empty:
                return None
            return [
                float(ohlcv["Open"].values[0]),
                float(ohlcv["High"].values[0]),
                float(ohlcv["Low"].values[0]),
                float(ohlcv["Close"].values[0]),
                float(ohlcv["Volume"].values[0]),
            ]
        else:
            return self._get_binance_ohlcv(symbol, date)

    def _get_binance_ohlcv(self, symbol, date):
        binance_symbol = symbol + "USDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "1d",
            "limit": 1,
            "startTime": int(date.timestamp() * 1000),
            "endTime": int((date + timedelta(days=1)).timestamp() * 1000),
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        data = response.json()
        if not data:
            return None
        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df = df[["open", "high", "low", "close", "volume"]]
        return df.iloc[0].tolist()
