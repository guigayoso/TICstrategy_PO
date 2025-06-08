import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import yfinance as yf
import wikipedia as wp
import io
import ast

class StockDataProcessor:
    def __init__(self, start_year=2019, prices_filename="sp500_data.csv", evolution_filename="sp500_evolution.csv"):
        self.start_year = start_year
        self.prices_filename = prices_filename
        self.evolution_filename = evolution_filename
        self.df_existing = self._load_existing_data()
        self.evolution = self._load_existing_evolution()
        self.changes = self._get_sp500_changes()
        self.evolution = self._update_sp500_evolution()

    def _load_existing_data(self):
        if os.path.exists(self.prices_filename):
            return pd.read_csv(self.prices_filename, index_col=0, parse_dates=True)
        return pd.DataFrame()
    
    def _load_existing_evolution(self):
        if os.path.exists(self.evolution_filename):
            df = pd.read_csv(self.evolution_filename, index_col=0, parse_dates=True, converters={'tickers': ast.literal_eval})
            return df
        return pd.DataFrame()
    
    def get_index_composition(self,date):
        if date in self.evolution.index.tolist():
            return self.evolution.loc[date, 'tickers']
        
        prior_dates = self.evolution[self.evolution.index < date]
        if not prior_dates.empty:
            latest_date = prior_dates.index[-1]
            return self.evolution.loc[latest_date, 'tickers']
        
        return []
    '''
    def _get_sp500_tickers(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch S&P 500 list from Wikipedia, status code: {response.status_code}"
            )
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        tickers = []
        for row in table.find_all("tr")[1:]:  # Skip header row
            ticker = row.find_all("td")[0].text.strip()
            tickers.append(ticker)
        return tickers
    '''
    
    def run(self):
        # Determine the last date for which data exists
        last_scraped_date = self.df_existing.index.max()
        if isinstance(last_scraped_date, str):
            last_scraped_date = datetime.strptime(last_scraped_date, "%Y-%m-%d")
        if pd.isna(last_scraped_date):
            last_scraped_date = datetime(self.start_year - 1, 12, 31)

        # Iterate over each day from the last scraped date to today
        dates = pd.date_range(
            start=last_scraped_date + timedelta(days=1), end=datetime.today(), freq="D"
        )
        for date in dates:
            self._fetch_and_save_prices(date)

    def _fetch_and_save_prices(self, target_date):
        # Try to get trading data for the target date, or if market closed, use the most recent trading day.
        tickers = self.get_index_composition(target_date)
        trading_date, data = self._get_trading_data(tickers, target_date)
        row_data = {}
        if data is None or data.empty:
                print(
                    f"⚠️ No trading data available for {target_date.date()} or prior days for ticker {ticker}. Skipping."
                )
        for ticker in tickers:
            
            # Check if the ticker is in the downloaded data
            if ticker not in data.columns.get_level_values(1).tolist():
                print(f"⚠️ Ticker {ticker} not found in downloaded data. Skipping.")
                continue
            try:
                open_val = data[("Open",ticker)].iloc[0]
                high_val = data[("High",ticker)].iloc[0]
                low_val = data[("Low",ticker)].iloc[0]
                close_val = data[("Close",ticker)].iloc[0]
                volume_val = data[("Volume",ticker)].iloc[0]
                # If the value is NaN, skip this ticker.
                if pd.isna(open_val):
                    continue
                row_data[ticker] = [float(open_val), float(high_val), float(low_val), float(close_val), float(volume_val)]
            except Exception:
                # If a ticker isn't found in the downloaded data, skip it.
                print('benfica')
                continue
            
            

        # Process the downloaded data.
        # The downloaded DataFrame has a MultiIndex for columns:
        # first level: data fields (Open, High, Low, Close, Volume), second level: ticker.
    

        if row_data:
            # Save the data using the original target_date as the index,
            # even if the data came from a previous trading day.
            df = pd.DataFrame.from_dict({target_date: row_data}, orient="index")
            self.df_existing = pd.concat([self.df_existing, df])
            self.df_existing.to_csv(self.prices_filename)
            print(
                f"✅ Saved data for {target_date.date()} (using trading data from {trading_date.date()})"
            )
        else:
            print(f"⚠️ No valid data for {target_date.date()}.")

    def _get_sp500_changes(self): #, title = 'List of S&P 500 companies', filename = 'sp500.csv', match = 'Symbol', use_cache=False, changes = False):
        
        if not self.evolution.empty:
            last_date = self.evolution.index[-1]
        else:
            last_date = datetime(2019,1,1)
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        title = 'List of S&P 500 companies'
        html = wp.page(title).html()
        
        df = pd.read_html(io.StringIO(html), header=0, match='Added')[0]
        df = df[['Date', 'Added', 'Removed']]
        df.columns = ['Date', 'add', 'remove']
        df = df.drop(index = 0)
        df['Date'] = df['Date'].apply(pd.to_datetime)

        df = df.groupby('Date', as_index = False).agg({
            'add': lambda x: [item for item in x if pd.notna(item)],
            'remove': lambda x: [item for item in x if pd.notna(item)]
            })

        df = df[df['Date'] > last_date]
        #filename = 'data/stock_data_scraping/sp500_changes.csv'
        #df.to_csv(filename, header=True, index=False, encoding='utf-8')

        return df

    def _get_trading_data(self, symbol,  target_date, max_tries=10):
        """
        Attempt to download symbol data for the target_date. If no data is available
        (e.g., market closed), search backward (up to max_tries days) for the most recent trading day.
        """
        current_date = target_date
        for _ in range(max_tries):
            end_date = current_date + timedelta(days=1)
            data = yf.download(tickers=symbol,
                start=current_date,
                end=end_date
                )
            if not data.empty:
                # Data is found for current_date
                return current_date, data
            # If no data, step back one day.
            current_date -= timedelta(days=1)
            # Optionally, only check weekdays:
            # if current_date.weekday() >= 5:  # Saturday or Sunday
            #     continue
        # If no trading day found within max_tries, return None.
        return None, None
    
    def _update_sp500_evolution(self, end_date = datetime.today().date()): #filename = 'sp500_csvs/S&P 500 Historical Components and Changes.csv', end_date = dt.today().date()) -> pd.DataFrame:
        
        print('IM RUNNING')
        #self.evolution['Date'] = pd.to_datetime(self.evolution['Date'])
        if self.evolution.empty:
            last_date = datetime(self.start_year,1,1)
        else:
            last_date = self.evolution.index[-1]
        end_date = pd.to_datetime(end_date)
        self.changes = self.changes.loc[(self.changes['Date'] > last_date) & (self.changes['Date'] < end_date)]

        def replace_dots_with_dashes(ticker_list):
            return [ticker.replace('.', '-') if '.' in ticker else ticker for ticker in ticker_list]

        self.evolution.index = pd.to_datetime(self.evolution.index)

        if self.changes.empty:
            print(self.changes.empty)
            return self.evolution
        
        else:

            # Corrigir: função segura para adicionar 'LIN' se necessário
            def add_lin(tickers):
                if 'LIN' not in tickers:
                    return tickers + ['LIN']
                return tickers

            # Aplicar no intervalo de datas
            mask = self.evolution.index >= pd.to_datetime('2018-10-31')
            self.evolution.loc[mask, 'tickers'] = self.evolution.loc[mask, 'tickers'].apply(add_lin)

            for i in range(len(self.changes)):
                change = self.changes.iloc[i]
                new_row = self.evolution.tail(1)
                print(change)
                change['add'] = replace_dots_with_dashes(change['add'])
                change['remove'] = replace_dots_with_dashes(change['remove'])

                tickers = new_row['tickers'].iloc[0]
                tickers += change['add']
                tickers = list(set(tickers) - set (change['remove']))
                        
                d = {'Date': change['Date'], 'tickers': [tickers]}
                new_entry = pd.DataFrame(d, columns=['Date', 'tickers'])
                print(new_entry)
                new_entry.set_index('Date', inplace=True)
                #new_entry.columns = self.evolution.columns
                #new_entry.index = [new_row.index[0] + 1]
                self.evolution = pd.concat([self.evolution, new_entry], axis = 0)
            

            self.evolution['tickers'] = self.evolution['tickers'].apply(replace_dots_with_dashes)

            
        
        filename = 'sp500_evolution.csv'
        self.evolution.to_csv(filename, header=True, index=True)

        return self.evolution



if __name__ == "__main__":
    processor = StockDataProcessor()
    processor.run()