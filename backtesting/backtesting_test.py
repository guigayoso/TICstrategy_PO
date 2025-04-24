import asyncio
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AsyncCache:
    def __init__(self, seconds_to_expire=120, interval=60):
        self.cache = {}
        self.seconds_to_expire = seconds_to_expire
        self.interval = interval
        asyncio.create_task(self.cleanup_task())

    async def set_key_value(self, key, value):
        self.cache[key] = value
        return value

    async def get_key_value(self, key):
        return self.cache.get(key)

    async def get_stock_data(self, tickers, start, end):
        data = {}

        for ticker in tickers:
            cached_data = await self.get_key_value(ticker)
            if cached_data:
                data[ticker] = cached_data

            else:
                data[ticker] = yf.download(ticker, start=start, end=end)
                await self.set_key_value(ticker, data[ticker])

        return data

    async def cleanup_task(self):
        while True:
            await asyncio.sleep(self.interval)
            current_time = time.time()

            keys_to_delete = [
                key
                for key, value in self.cache.items()
                if current_time - value[1] > self.seconds_to_expire
            ]
            for key in keys_to_delete:
                del self.cache[key]


class Backtester:
    def __init__(
        self,
        data,
        strategy=None,
        initial_balance=10000,
        transaction_cost=0.001,
        stop_loss=0.1,
        take_profit=0.2,
        risk_free_rate=0.03,
        benchmark="^GSPC",
    ):
        """
        Multi-asset portfolio backtester.

        :param data: DataFrame with ticker prices.
        :param strategy: Function that determines position allocation.
        :param initial_balance: Starting capital.
        :param transaction_cost: Fee per trade (percentage).
        :param stop_loss: Stop-loss percentage per trade.
        :param take_profit: Take-profit percentage per trade.
        """
        self.data = data
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.total_transactions = 0
        self.total_transaction_costs = 0
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.assets = list(data.keys())
        self.positions = None
        self.metrics = None
        self.risk_free_rate = risk_free_rate
        self.time_series = Backtester.get_time_series(data)
        self.benchmark = benchmark

    @staticmethod
    def get_time_series(data):
        """Gets a dict in the format {
          assetA: [prices],
          assetB: [prices],
          ...
        }

        and extracts the time series
        """

        assets = list(data.keys())
        return data[assets[0]].index

    def apply_strategy(self):
        """Applies the user-defined strategy function."""
        self.positions = self.strategy(self.data)

    @staticmethod
    def convert_data_format(data_dict):
        """
        Convert dictionary from format:
        {
            'AAPL': DataFrame with historical prices from yf,
            'GOOG': DataFrame with historical prices from yf,
            ...
        }
        to a structured format where each stock's OHLCV data is labeled:

            DateTime | AAPL_Open | AAPL_High | AAPL_Low | AAPL_Close | AAPL_Volume | GOOG_Open | ...

        Ensures compatibility with simulate_trading().
        """
        formatted_data = {}

        for ticker, df in data_dict.items():
            df = df[
                ["Open", "High", "Low", "Close", "Volume"]
            ].copy()  # Select relevant columns
            df.columns = [f"{ticker}_{col}" for col in df.columns]  # Rename columns
            formatted_data[ticker] = df

        return pd.concat(formatted_data.values(), axis=1)

    def simulate_trading(self):
        """Runs the backtest with position allocation, transaction costs, and risk limits."""
        balance = self.initial_balance
        holdings = {asset: 0 for asset in self.data.keys()}  # Track number of shares
        portfolio_values = [
            (balance, balance, balance, balance, 0)
        ]  # Close, High, Low, Open, Volume

        data_time = Backtester.convert_data_format(self.data)

        for i in range(1, len(self.time_series)):
            date = pd.to_datetime(self.time_series[i])

            if date not in self.positions.index:
                continue  # Skip if no allocation for this date

            new_allocations = self.positions.loc[date]  # Get allocations
            prices = data_time.iloc[i].astype(float)
            prev_prices = data_time.iloc[i - 1].astype(float)

            # Compute current portfolio value
            total_value = [
                balance
                + sum(
                    holdings[asset] * prices[f"{asset}_Close"] for asset in self.assets
                ),
                balance
                + sum(
                    holdings[asset] * prices[f"{asset}_High"] for asset in self.assets
                ),
                balance
                + sum(
                    holdings[asset] * prices[f"{asset}_Low"] for asset in self.assets
                ),
                balance
                + sum(
                    holdings[asset] * prices[f"{asset}_Open"] for asset in self.assets
                ),
                sum(
                    holdings[asset] * prices[f"{asset}_Volume"] for asset in self.assets
                ),
            ]

            # Stop-loss and take-profit execution
            for asset in self.assets:
                if holdings[asset] > 0:
                    change_pct = (
                        prices[f"{asset}_Close"] - prev_prices[f"{asset}_Close"]
                    ) / prev_prices[f"{asset}_Close"]
                    if change_pct <= -self.stop_loss or change_pct >= self.take_profit:
                        balance += (
                            holdings[asset] * prices[f"{asset}_Close"]
                        )  # Sell at close price
                        holdings[asset] = 0  # Exit position

            # Adjust holdings based on new allocations
            for asset in self.assets:
                if asset not in new_allocations:
                    continue

                target_value = (
                    total_value[0] * new_allocations[asset]
                )  # Desired capital allocation per asset
                new_shares = target_value / prices[f"{asset}_Close"]
                trade_size = (new_shares - holdings[asset]) * prices[
                    f"{asset}_Close"
                ]  # Trade size in dollars

                if np.isnan(trade_size):
                    continue

                balance -= (
                    abs(self.transaction_cost * trade_size) + trade_size
                )  # Update balance
                self.total_transactions += 1
                self.total_transaction_costs += abs(self.transaction_cost * trade_size)

                holdings[asset] = new_shares  # Update holdings

            # Store portfolio value
            total_value[0] = balance + sum(
                holdings[asset] * prices[f"{asset}_Close"] for asset in self.assets
            )
            portfolio_values.append(total_value)

        self.portfolio_value = pd.DataFrame(
            portfolio_values[1:],
            columns=["Close", "High", "Low", "Open", "Volume"],
            index=self.positions.index,
        )

    def run_backtest(self):
        """Executes the backtest and returns results."""
        # self.apply_strategy()
        self.simulate_trading()
        return self.portfolio_value

    def calculate_metrics(self, rerun=False):
        """Calculates performance metrics."""

        if self.portfolio_value is None:
            raise ValueError(
                "Portfolio value is not available. Run the backtest first."
            )

        if self.metrics is not None and not rerun:
            return self.metrics

        def calculate_sharpe_ratio(time_series, risk_free_rate):
            """
            Calculate the Sharpe ratio for a given time series.
            """
            returns = np.log(time_series).diff().dropna()
            excess_returns = np.mean(returns) * 252 - risk_free_rate
            volatility = np.std(returns) * np.sqrt(252)

            return excess_returns / volatility

        def calculate_sortino_ratio(time_series, risk_free_rate):
            """
            Calculate the Sortino ratio for a given time series.
            """

            returns = np.log(time_series).diff().dropna()
            excess_returns = np.mean(returns) * 252 - risk_free_rate
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252)

            return excess_returns / downside_volatility

        def calculate_maximum_drawdown(time_series):
            """
            Calculate the maximum drawdown for a given time series.
            """

            drawdown = (time_series.cummax() - time_series) / time_series.cummax()
            return drawdown.max()

        def calculate_total_return(time_series, transaction_costs):
            """
            Calculate the total return for a given time series.
            """
            return (time_series.iloc[-1] - transaction_costs) / time_series.iloc[0] - 1

        def calculate_annualized_return(time_series, transaction_costs):
            """
            Calculate the annualized return for a given time series.
            """
            total_return = calculate_total_return(time_series, transaction_costs)
            return (1 + total_return) ** (252 / len(time_series)) - 1

        def calculate_annualized_volatility(time_series):
            """
            Calculate the annualized volatility for a given time series.
            """
            returns = np.log(time_series).diff().dropna()
            return np.std(returns) * np.sqrt(252)

        sharpe_ratio = calculate_sharpe_ratio(
            self.portfolio_value["Close"], self.risk_free_rate
        )
        sortino_ratio = calculate_sortino_ratio(
            self.portfolio_value["Close"], self.risk_free_rate
        )
        max_drawdown = calculate_maximum_drawdown(self.portfolio_value["Close"])
        total_return = calculate_total_return(
            self.portfolio_value["Close"], self.total_transaction_costs
        )
        annualized_return = calculate_annualized_return(
            self.portfolio_value["Close"], self.total_transaction_costs
        )
        annualized_volatility = calculate_annualized_volatility(
            self.portfolio_value["Close"]
        )

        self.metrics = {
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
        }

        return self.metrics

    def plot_results(self):
        """Plots the backtest results."""
        # plot the time series porfolio value

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.portfolio_value.index,
            self.portfolio_value["Close"],
            label="Portfolio Value",
        )
        plt.xlabel("Date")
        plt.ylabel("Total Value")

        plt.title("Portfolio Value Over Time")
        plt.legend()
        plt.show()


class StressTester:
    def __init__(self, backtester: Backtester, stress_scenarios=None):
        """
        Stress test the backtest strategy for different market conditions.

        :param backtester: An instance of the Backtester class with the strategy and data.
        :param stress_scenarios: A list of scenarios to simulate during testing.
                                  Each scenario will be a function that modifies the data.
        """
        self.backtester = backtester
        self.stress_scenarios = stress_scenarios if stress_scenarios else []
        self.stress_results = {}

    def apply_stress_scenarios(self, data):
        """
        Applies each stress scenario to the original data and returns the modified data.

        :param data: Original stock data.
        :return: Modified stock data after applying the stress scenarios.
        """
        modified_data = data.copy()

        for scenario in self.stress_scenarios:
            modified_data = scenario(modified_data)

        return modified_data

    def run_stress_test(self):
        """
        Runs the stress test by applying the stress scenarios to the backtest data.
        """
        # Modify the backtest data according to the stress scenarios

        if len(self.stress_scenarios) == 0:
            raise ValueError("No stress scenarios provided.")

        modified_data = self.apply_stress_scenarios(self.backtester.data)

        # Run backtest on modified data
        bt = Backtester(modified_data, self.backtester.strategy)
        results = bt.run_backtest()
        metrics = bt.calculate_metrics()

        # Store results
        self.stress_results = {"metrics": metrics, "portfolio_value": results}

        return self.stress_results

    def get_stress_test_results(self):
        """
        Returns the results of the stress test.
        """
        return self.stress_results


"""
Data is a dictionary with the format:
{
    'AAPL': DataFrame with historical prices from yf,
    'GOOG': DataFrame with historical prices from yf,
    ...
}

"""
"""
VERY IMPORTANT: The strategy function should return a DataFrame with the same index 
as the input data, and columns for each asset with the allocation percentage for that asset.
"""


def allocation_strategy(data_dict, short_window=50):
    """Returns portfolio allocations for each asset as percentages."""
    allocations = pd.DataFrame(index=list(data_dict.values())[0].index)

    for asset, df in data_dict.items():
        ma = df["Close"].rolling(window=short_window).mean()
        weight = np.where(df["Close"] > ma, 0.75, 0.25)  # More weight if above MA
        allocations[asset] = weight

    # Normalize allocations so they sum to 100%
    allocations = allocations.div(allocations.sum(axis=1), axis=0)

    return allocations.fillna(1 / len(data_dict))  # Default equal allocation if NaN


"""
Stress Testing
"""

import random


# Example Stress Scenario: Simulating a market crash
def market_crash(data):
    """Simulates a 30% drop in prices across all assets on a random date."""
    crash_date = random.choice(
        list(data.values())[0].index
    )  # Choose random date from the data
    crash_factor = float(0.6)  # 40% drop
    for ticker, df in data.items():
        df.loc[
            crash_date:
        ] *= crash_factor  # Apply drop to all prices after the crash date
    return data


def high_volatility(data):
    """Injects random volatility spikes to simulate a turbulent market."""
    volatility_range = (
        float(1),
        float(1.05),
    )  # Range for random volatility multiplier (1 to 5%)
    proportion = 0.1  # Proportion of days to inject volatility

    for ticker, df in data.items():
        shock_days = np.random.choice(
            df.index, size=int(len(df) * proportion), replace=False
        )  # Select % of days
        df.loc[shock_days] *= np.random.uniform(
            volatility_range[0],
            volatility_range[1],
            size=(len(shock_days), df.shape[1]),
        )  # inject swings
    return data


# Dictionary of stress test conditions (Scalable)
stress_conditions = {
    "Market Crash": market_crash,
    "High Volatility": high_volatility,
    "Both (Crash + Volatility)": lambda data: high_volatility(
        market_crash(data)
    ),  # Apply both
}

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from copy import deepcopy
import pandas as pd


class DashApp:
    def __init__(
        self,
        tickers,
        async_cache,
        backtester_class,
        stress_tester_class,
        stress_conditions,
    ):
        self.tickers = tickers
        self.async_cache = async_cache
        self.backtester_class = backtester_class
        self.stress_tester_class = stress_tester_class
        self.stress_conditions = stress_conditions
        self.SPdata = None  # Store SPdata as instance variable
        self.original_bt = None  # Store original backtester

        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )

    async def fetch_data(self):
        self.SPdata = await self.async_cache.get_stock_data(
            ["^GSPC"], start="2021-08-01", end="2025-01-01"
        )
        data = await self.async_cache.get_stock_data(
            self.tickers, start="2021-08-01", end="2025-01-01"
        )
        return self.SPdata, data

    async def prepare_data(self, SPdata, data):
        strategy = allocation_strategy
        bt = self.backtester_class(data, strategy)
        self.original_bt = bt  # Store original backtester
        results = bt.run_backtest()
        metrics = bt.calculate_metrics()

        portfolio_data = pd.DataFrame(
            {
                "Date": results.index,
                "Portfolio": results["Close"],
            }
        )

        sp500_data = pd.DataFrame(
            {
                "Date": SPdata["^GSPC"].index,
                "SP500": SPdata["^GSPC"]["Close"].values.flatten(),
            }
        )

        sp500_data_filtered = sp500_data[
            sp500_data["Date"].isin(portfolio_data["Date"])
        ]

        return results, metrics, portfolio_data, sp500_data_filtered, bt

    async def create_backtest_fig(self, portfolio_data, sp500_data_filtered, results):
        backtest_fig = go.Figure()

        backtest_fig.add_trace(
            go.Candlestick(
                x=portfolio_data["Date"],
                open=results.loc[portfolio_data["Date"]]["Open"],
                high=results.loc[portfolio_data["Date"]]["High"],
                low=results.loc[portfolio_data["Date"]]["Low"],
                close=portfolio_data["Portfolio"],
                name="Portfolio",
            )
        )

        backtest_fig.add_trace(
            go.Scatter(
                x=sp500_data_filtered["Date"],
                y=(sp500_data_filtered["SP500"] / sp500_data_filtered["SP500"].iloc[0])
                * portfolio_data["Portfolio"].iloc[0],
                mode="lines",
                name="S&P 500 (Normalized)",
                visible="legendonly",
            )
        )

        backtest_fig.update_layout(
            title="Backtesting Results",
            xaxis_title="Date",
            yaxis=dict(title="Portfolio Value"),
            template="ggplot2",
            height=800,
        )

        return backtest_fig

    async def create_metrics_card(self, metrics, bt):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Backtest Performance Metrics",
                        className="card-title text-center",
                    ),
                    html.Hr(),
                    html.P(
                        f"Sharpe Ratio: {round(metrics['Sharpe Ratio'], 2)}",
                        className="text-center",
                    ),
                    html.P(
                        f"Sortino Ratio: {round(metrics['Sortino Ratio'], 2)}",
                        className="text-center",
                    ),
                    html.P(
                        f"Maximum Drawdown: {round(metrics['Max Drawdown'] * 100, 2)}%",
                        className="text-center",
                    ),
                    html.P(
                        f"Total Return: {round(metrics['Total Return'] * 100, 2)}%",
                        className="text-center",
                    ),
                    html.P(
                        f"Annualized Return: {round(metrics['Annualized Return'] * 100, 2)}%",
                        className="text-center",
                    ),
                    html.P(
                        f"Annualized Volatility: {round(metrics['Annualized Volatility'] * 100, 2)}%",
                        className="text-center",
                    ),
                    html.P(
                        f"Total Transactions: {bt.total_transactions}",
                        className="text-center",
                    ),
                    html.P(
                        f"Total Transaction Costs: {round(bt.total_transaction_costs, 2)}$",
                        className="text-center",
                    ),
                ]
            ),
            className="shadow-lg p-3 mb-5 bg-white rounded",
            style={"width": "50%", "margin": "auto"},
        )

    def create_stress_metrics_content(self, stress_metrics, bt, selected_condition):
        if not stress_metrics:
            return [
                html.H4(
                    "Stress Test Performance Metrics",
                    className="card-title text-center",
                ),
                html.Hr(),
                html.P("Run stress test to see metrics", className="text-center"),
            ]

        return [
            html.H4(
                f"Stress Test Performance Metrics ({selected_condition})",
                className="card-title text-center",
            ),
            html.Hr(),
            html.P(
                f"Sharpe Ratio: {round(stress_metrics['Sharpe Ratio'], 2)}",
                className="text-center",
            ),
            html.P(
                f"Sortino Ratio: {round(stress_metrics['Sortino Ratio'], 2)}",
                className="text-center",
            ),
            html.P(
                f"Maximum Drawdown: {round(stress_metrics['Max Drawdown'] * 100, 2)}%",
                className="text-center",
            ),
            html.P(
                f"Total Return: {round(stress_metrics['Total Return'] * 100, 2)}%",
                className="text-center",
            ),
            html.P(
                f"Annualized Return: {round(stress_metrics['Annualized Return'] * 100, 2)}%",
                className="text-center",
            ),
            html.P(
                f"Annualized Volatility: {round(stress_metrics['Annualized Volatility'] * 100, 2)}%",
                className="text-center",
            ),
            html.P(
                f"Total Transactions: {bt.total_transactions}", className="text-center"
            ),
            html.P(
                f"Total Transaction Costs: {round(bt.total_transaction_costs, 2)}$",
                className="text-center",
            ),
        ]

    async def run(self):
        # Fetch and prepare initial data
        SPdata, data = await self.fetch_data()
        results, metrics, portfolio_data, sp500_data_filtered, bt = (
            await self.prepare_data(SPdata, data)
        )

        # Create initial figures
        backtest_fig = await self.create_backtest_fig(
            portfolio_data, sp500_data_filtered, results
        )
        metrics_card = await self.create_metrics_card(metrics, bt)
        initial_stress_metrics = self.create_stress_metrics_content(
            {}, bt, "Market Crash"
        )

        # Define app layout
        self.app.layout = dbc.Container(
            [
                html.H1(
                    "Backtesting and Stress Testing Results",
                    className="text-center my-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Select Stress Condition:",
                                    className="text-center my-2",
                                ),
                                dcc.Dropdown(
                                    id="stress-condition",
                                    options=[
                                        {"label": name, "value": name}
                                        for name in self.stress_conditions.keys()
                                    ],
                                    value="Market Crash",
                                    clearable=False,
                                ),
                            ],
                            width=6,
                            className="text-center",
                        )
                    ],
                    className="mb-3 d-flex justify-content-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(metrics_card, width=6),
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        initial_stress_metrics, id="stress-test-metrics"
                                    ),
                                    className="shadow-lg p-3 mb-5 bg-white rounded",
                                    style={"width": "100%"},
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="backtest-graph", figure=backtest_fig),
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="stress-test-graph", figure=backtest_fig),
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

        @self.app.callback(
            [
                Output("stress-test-graph", "figure"),
                Output("stress-test-metrics", "children"),
            ],
            Input("stress-condition", "value"),
        )
        def update_stress_test(selected_condition):
            if not selected_condition:
                raise PreventUpdate

            # Create a deep copy of the original backtester's data
            data_copy = deepcopy(self.original_bt.data)

            # Apply the selected stress condition
            stress_function = self.stress_conditions[selected_condition]
            modified_data = stress_function(data_copy)

            # Create new backtester with modified data but same strategy
            stress_bt = self.backtester_class(modified_data, self.original_bt.strategy)
            stress_portfolio = stress_bt.run_backtest()
            stress_metrics = stress_bt.calculate_metrics()

            # Prepare data for graph
            stress_portfolio_data = pd.DataFrame(
                {"Date": stress_portfolio.index, "Portfolio": stress_portfolio["Close"]}
            )

            # Get SP500 data for the stress test period
            sp500_data = pd.DataFrame(
                {
                    "Date": self.SPdata["^GSPC"].index,
                    "SP500": self.SPdata["^GSPC"]["Close"].values.flatten(),
                }
            )
            sp500_data_filtered = sp500_data[
                sp500_data["Date"].isin(stress_portfolio_data["Date"])
            ]

            # Create stress test graph
            stress_fig = go.Figure()
            stress_fig.add_trace(
                go.Candlestick(
                    x=stress_portfolio_data["Date"],
                    open=stress_portfolio["Open"],
                    high=stress_portfolio["High"],
                    low=stress_portfolio["Low"],
                    close=stress_portfolio_data["Portfolio"],
                    name="Stress Test Portfolio",
                )
            )

            stress_fig.add_trace(
                go.Scatter(
                    x=sp500_data_filtered["Date"],
                    y=(
                        sp500_data_filtered["SP500"]
                        / sp500_data_filtered["SP500"].iloc[0]
                    )
                    * stress_portfolio_data["Portfolio"].iloc[0],
                    mode="lines",
                    name="S&P 500 (Normalized)",
                    visible="legendonly",
                )
            )

            stress_fig.update_layout(
                title=f"Stress Test: {selected_condition}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                template="ggplot2",
                height=800,
            )

            # Update metrics content
            stress_metrics_content = self.create_stress_metrics_content(
                stress_metrics, stress_bt, selected_condition
            )

            return stress_fig, stress_metrics_content

        self.app.run_server(debug=True)


if __name__ == "__main__":
    import asyncio

    async def init_and_run():
        # Define tickers for testing
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

        # Initialize cache inside async context
        async_cache = AsyncCache()

        # Create Dash app instance
        dash_app = DashApp(
            tickers=tickers,
            async_cache=async_cache,
            backtester_class=Backtester,
            stress_tester_class=StressTester,
            stress_conditions=stress_conditions,
        )

        # Run the app
        await dash_app.run()

    # Create and run the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_and_run())
