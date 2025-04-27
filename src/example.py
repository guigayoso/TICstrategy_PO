from datetime import datetime as dt
import enum
import numpy as np
import pandas as pd
import pydantic
import typing
import typing_extensions
import yahoofinancials as yf
import data_models as dm
import src.portfolio_strategy as strategy
import src.tc_optimization as tc


Strategy= strategy.Strategy_Portfolio(
                        start_date = dt(2024, 1, 2),
                        end_date = dt(2024, 12, 31), # Actually, the end is today
                        Asset_Category=dm.Asset_Category.Top20CryptoByMarketCap,
                        Momentum_Type= dm.Momentum_Type.MACD,
                        Mean_Rev_Type= dm.Mean_Rev_Type.RSI,
                        Rebalancing_Period= dm.Rebalancing_Period.daily,
                        Functional_Constraints= dm.Functional_Constraints(
                            Take_Profit= 0.2,
                            Stop_Loss= 0.1, Capital_at_Risk= 0.6, 
                            Hurst_Filter = dm.HurstFilter.STANDARD,
                            RSIFilter= dm.RSIFilter.STANDARD,
                            Hurst_Exponents_Period = 180,
                            Momentum_Threshold= 0.01,
                            MACD_Short_Window= 12, MACD_Long_Window= 26,
                            Bollinger_Window= 20),
                        Rebalance_Constraints= dm.Rebalance_Constraints(
                            Long_Only= True, Turnover_Constraint= 0.5, 
                            distance_method = tc.DistanceMethod.NORMALIZED_EUCLIDEAN,
                            Transaction_Cost = 0.01, Gamma = 0,
                            Delta_Range = (0, 1), delta_step= 1),
                            Momentum_Days= 30                    
                                    )

Strategy.run_strategy()

Strategy.backtest()

Strategy.stress_test()