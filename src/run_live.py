from datetime import datetime as dt
import enum
import numpy as np
import pandas as pd
import pydantic
import typing
import typing_extensions
import yahoofinancials as yf
import data_models as dm
import portfolio_strategy as strategy
import tc_optimization as tc
import live



Live_Strategy = live.Live(Best_Delta=0.235, 
                 Asset_Category=dm.Asset_Category.Top20CryptoByMarketCap,
                 Momentum_Type= dm.Momentum_Type.MACD,
                 Mean_Rev_Type= dm.Mean_Rev_Type.RSI,
                 Rebalancing_Period= dm.Rebalancing_Period.WEEKLY,
                 Functional_Constraints= dm.Functional_Constraints(
                     Take_Profit= 0.2,
                     Stop_Loss= 0.1, Capital_at_Risk= 0.6, 
                     Hurst_Filter = dm.HurstFilter.STANDARD,
                     RSIFilter= dm.RSIFilter.LENIENT,
                     Hurst_Exponents_Period = 102,
                     MACD_Short_Window= 1, MACD_Long_Window= 2,
                     Bollinger_Window= 20,
                     RSI_Window= 5),
                 Rebalance_Constraints= dm.Rebalance_Constraints(
                     Long_Only= True, Turnover_Constraint= 0.5, 
                     distance_method = tc.DistanceMethod.NORMALIZED_EUCLIDEAN,
                     Transaction_Cost = 0.01),  # Fazer ligação para usar mesmos parametros do backtesting                
                        Momentum_Days= 30)

#Live_Strategy.get_live_data() # Equipe de Strategy implementation

Live_Strategy.run_live()



