import datetime as dt
import enum
import numpy as np
import pandas as pd
import pydantic
import typing
import typing_extensions
import yahoofinancials as yf
import tc_optimization as tc

'''

Classes of all the arguments we use in our functions

'''


#Quando tivermos a lista de tickers associar cada ticker a um asset type
class ReturnsType(str, enum.Enum):
    """Indicates the type of returns"""

    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"

class AssetType(str, enum.Enum):
    """Indicated the type of asset"""

    stock = "stock"
    bond = "bond"
    cryptocurrency = "cryptocurrency"

class AssetID(pydantic.BaseModel):
    """Indicates the ID of the specific Asset"""
    
    type: AssetType = pydantic.Field(
        ..., description="Indicates the type of Asset"
    )
    ticker: str = pydantic.Field(
        ..., description="Indicates the Asset ticker"
    )
    name: typing.Optional[str] = pydantic.Field(
        default=None, description="Indicates the name of the Asset"
    )
    def get_ticker(self):
        return self.ticker
    def get_name(self):
         return self.name

class Broker(str, enum.Enum):
    """Indicates what broker to use to fetch data"""
    YahooFinance = "Yahoo"
    Binance = "Binance"

class Returns(pydantic.BaseModel, arbitrary_types_allowed=True):
    type: ReturnsType = pydantic.Field(
        ..., description="Indicates the type of returns provided"
    )
    timestamps: typing.List[dt.datetime] = pydantic.Field(
        ..., description="Timestamps of the returns provided"
    )
    rebalancing_frequency: typing.List[int] = pydantic.Field(
        ..., description = "Indicates the rebalancing_frequency"
    )
    returns: np.ndarray = pydantic.Field(
        ..., description="Timeseries of returns, corresponding to timestamps"
    )

    @classmethod
    def from_pandas_dataframe(
            cls, type: ReturnsType, returns: pd.DataFrame
            ) -> typing_extensions.Self:
        # code to convert from pandas to np.darray
        # return cls(type, timestamps, returns_np_array_format)
        pass

    def to_pandas_dataframe(self) -> pd.DataFrame:
        # code to convert from np.darray to pandas
        # return returns_pd_format
        pass

    @staticmethod
    def _example_method_for_validating_something(
            condition_to_validate: bool) -> None:
        if condition_to_validate:
            raise ValueError("Example Error Message")

class Asset(pydantic.BaseModel):
    """Indicates all information characterising a specific Asset"""
    id: AssetID = pydantic.Field(
        ..., description="Inequivocally identified the Asset"
    )

class Asset_Category(str, enum.Enum):
    Top20CryptoByMarketCap = "crypto" #Use API to get this info
    SP500_Stocks= 'SPY' #Get a list of the SP500 tickers
    Crypto_and_Stocks= 'both'
    tickers= ['BTC', 'ETH', 'LTC', 'ADA', 'TRX', 'BCH', 'LINK','BNB', 'DOGE', 'SOL', 'AVAX', 'DOT', 'ETC', 'XRP', 'XLM']
    #tickers= [ 'ETH', 'LTC', 'ADA', 'TRX', 'BCH', 'LINK','BNB', 'DOGE', 'SOL', 'AVAX', 'DOT', 'ETC', 'XRP', 'XLM']
    #tickers= ['BTC', 'ETH']

class Momentum_Type(str, enum.Enum):
    MACD = "MACD"   
    Sigmoid = "Sigmoid"
    Cumulative_Returns= "Cumulative_Returns"
    #RSI= "RSI"

class Mean_Rev_Type(str, enum.Enum):
    Bollinger_Bands= "BB"
    #GARCH= 'GARCH'
    RSI= "RSI"

class Weight_Allocation(str, enum.Enum):
    equal_weight= 'equal_weight'
    signal_strenght= 'signal_strenght'

class Rebalancing_Period(str, enum.Enum):
    daily= '1d'
    EVERY_2_DAYS = '2d'
    EVERY_3_DAYS = '3d'
    EVERY_4_DAYS = '4d'
    EVERY_5_DAYS = '5d'
    WEEKLY = '1w'
    BI_WEEKLY = '2w'
    MONTHLY = '1m'

class HurstFilter(tuple, enum.Enum):
    STANDARD = (0.45, 0.55)
    MODERATE = (0.4, 0.6) 
    STRICT = (0.35, 0.65) 

class RSIFilter(tuple, enum.Enum):
    STANDARD = (30,70)
    STRICT = (20,80)
    LENIENT = (40,60)

class Functional_Constraints:
    def __init__(self,
                 Capital_at_Risk: float, 
                 Hurst_Filter: HurstFilter,
                 RSIFilter: RSIFilter,
                 Take_Profit = 0.2,
                 Stop_Loss = 0.1,
                 Hurst_Exponents_Period= 180,
                 Momentum_Threshold= 10,
                 MACD_Short_Window= 20,
                 MACD_Long_Window= 50,
                 Bollinger_Window= 20,
                 RSI_Window= 5):
        self.take_profit = Take_Profit
        self.stop_loss = Stop_Loss
        self.capital_at_risk = Capital_at_Risk
        self.hurst_filter = Hurst_Filter
        self.hurst_exponents_period = Hurst_Exponents_Period
        self.momentum_threshold = Momentum_Threshold
        self.macd_short_window = MACD_Short_Window
        self.macd_long_window = MACD_Long_Window
        self.bollinger_window = Bollinger_Window
        self.rsi_window = RSI_Window
        self.rsi_filter = RSIFilter

    def get_capital_at_risk(self):
        return self.capital_at_risk

    def get_rsi_window(self):
        return self.rsi_window
    
    def get_take_profit(self):
        return self.take_profit
    
    def get_stop_loss(self):
        return self.stop_loss
    
    @property
    def rsi_overbought(self):
        return self.rsi_filter.value[1]
    
    @property
    def rsi_oversold(self):
        return self.rsi_filter.value[0]

    @property
    def hurst_trend_filter(self):
        return self.hurst_filter.value[1]

    @property
    def hurst_reversion_filter(self):
        return self.hurst_filter.value[0]

    def get_macd_short_window(self):
        return self.macd_short_window

    def get_macd_long_window(self):
        return self.macd_long_window

    def get_bollinger_window(self):
        return self.bollinger_window
    
    def get_momentum_threshold(self):
        return self.momentum_threshold

class Rebalance_Constraints:
    def __init__(self, Max_Factor_Exposure= None, Holding_Limit= None,
                 Long_Only= True, Turnover_Constraint= 0.5, 
                 Transaction_Cost = 0.01, distance_method = tc.DistanceMethod.EUCLIDEAN,
                 Gamma = 0.5, Delta_Range = (-10, 10), delta_step= 0.1):
        self.max_factor= Max_Factor_Exposure
        self.holding_limit= Holding_Limit
        self.long_only= Long_Only
        self.turnover_constraint= Turnover_Constraint
        self.transaction_cost= Transaction_Cost
        self.distance_method= distance_method
        self.gamma= Gamma
        self.delta_range= Delta_Range
        self.delta_step= delta_step
    
    def get_max_factor_exposure(self):
        return self.max_factor

    # Getter for Holding_Limit
    def get_holding_limit(self):
        return self.holding_limit

    # Getter for Long_Only
    def get_long_only(self):
        return self.long_only

    # Getter for Turnover_Constraint
    def get_turnover_constraint(self):
        return self.turnover_constraint
    
    def get_transaction_costs(self):
        return self.transaction_cost
    
    def get_gamma(self):
        return self.gamma
    
    def get_delta_range(self):
        return self.delta_range
    
    def get_delta_step(self):
        return self.delta_step


class Backtest:
    def __init__(self, enabled: bool = True, monte_carlo: bool = False, num_simulations: int = 1000):
        """
        Initializes the Backtest class.

        :param enabled: Whether backtesting is enabled.
        :param monte_carlo: Whether Monte Carlo simulations are enabled.
        :param num_simulations: Number of Monte Carlo simulations to run (if enabled).
        """
        self.enabled = enabled
        self.monte_carlo = monte_carlo
        self.num_simulations = num_simulations

    def is_enabled(self) -> bool:
        """Returns whether backtesting is enabled."""
        return self.enabled

    def is_monte_carlo_enabled(self) -> bool:
        """Returns whether Monte Carlo simulations are enabled."""
        return self.monte_carlo

    def get_num_simulations(self) -> int:
        """Returns the number of Monte Carlo simulations."""
        return self.num_simulations


class Stresstest:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def is_enabled(self) -> bool:
        return self.enabled

class Strategy_Output:
    def __init__(self, Total_Return=None,
                 Annualized_Return= None, 
                 Sharpe_Ratio=None, 
                 VaR=None, 
                 CVaR=None, 
                 Max_Drawdown=None, 
                 Sortino_Ratio=None, 
                 Profit_Ratio=None):
        self.total_return = Total_Return
        self.annualized_return= Annualized_Return
        self.sharpe_ratio = Sharpe_Ratio
        self.var = VaR
        self.cvar = CVaR
        self.max_drawdown = Max_Drawdown
        self.sortino_ratio = Sortino_Ratio
        self.profit_ratio = Profit_Ratio

    # Getter for Total_Return
    def get_total_return(self):
        if self.total_return is None:
            raise ValueError("Total_Return has not been assigned a value.")
        return self.total_return

    def get_annualized_return(self):
        if self.annualized_return is None:
            raise ValueError("Annualized_Return has not been assigned a value.")
        return self.annualized_return

    # Getter for Sharpe_Ratio
    def get_sharpe_ratio(self):
        if self.sharpe_ratio is None:
            raise ValueError("Sharpe_Ratio has not been assigned a value.")
        return self.sharpe_ratio

    # Getter for VaR
    def get_var(self):
        if self.var is None:
            raise ValueError("VaR has not been assigned a value.")
        return self.var

    # Getter for CVaR
    def get_cvar(self):
        if self.cvar is None:
            raise ValueError("CVaR has not been assigned a value.")
        return self.cvar

    # Getter for Max_Drawdown
    def get_max_drawdown(self):
        if self.max_drawdown is None:
            raise ValueError("Max_Drawdown has not been assigned a value.")
        return self.max_drawdown

    # Getter for Sortino_Ratio
    def get_sortino_ratio(self):
        if self.sortino_ratio is None:
            raise ValueError("Sortino_Ratio has not been assigned a value.")
        return self.sortino_ratio

    # Getter for Profit_Ratio
    def get_profit_ratio(self):
        if self.profit_ratio is None:
            raise ValueError("Profit_Ratio has not been assigned a value.")
        return self.profit_ratio
