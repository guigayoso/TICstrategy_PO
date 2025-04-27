import sys
import os

# Adiciona o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data_analysis as analysis
import signalling
import data_models as dm
import portfolio_weights as pw
from datetime import datetime as dt
import tc_optimization as tc

class PortfolioOptimization():

    def __init__(self, closes, previous_weights = None):
        self.closes = closes
        self.previous_weights = previous_weights
        self.Best_Delta = 0.124
        self.Momentum_Type = dm.Momentum_Type.Cumulative_Returns
        self.Mean_Rev_Type = dm.Mean_Rev_Type.RSI
        self.Rebalancing_Period = dm.Rebalancing_Period.daily
        self.Functional_Constraints = dm.Functional_Constraints(
            Take_Profit= 0.2,
            Stop_Loss= 0.1, Capital_at_Risk= 0.6, 
            Hurst_Filter = dm.HurstFilter.STANDARD,
            RSIFilter= dm.RSIFilter.STANDARD,
            Hurst_Exponents_Period = 180,
            MACD_Short_Window= 12, MACD_Long_Window= 26,
            Bollinger_Window= 20,
            RSI_Window= 5)
        self.Rebalance_Constraints = dm.Rebalance_Constraints(
            Long_Only= True, Turnover_Constraint= 0.5, 
            distance_method = tc.DistanceMethod.NORMALIZED_EUCLIDEAN,
            Transaction_Cost = 0.01)
        self.momentum_days = 30

    def get_weights(self):

        # Gerar sinais
        filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(self.closes, rebalancing_period=self.Rebalancing_Period, 
                                                                                                            hurst_exponents_period=self.Functional_Constraints.hurst_exponents_period, 
                                                                                                            mean_rev_type= self.Mean_Rev_Type, momentum_type= self.Momentum_Type, 
                                                                                                            functional_constraints= self.Functional_Constraints, 
                                                                                                            momentus_days_period=self.momentum_days,
                                                                                                            live_analysis = True), hurst_thresholds=self.Functional_Constraints.hurst_filter, 
                                                                                                            mean_rev_type= self.Mean_Rev_Type, momentum_type= self.Momentum_Type, live_analysis=True)
        buy_and_sells = signalling.buy_and_sell_signalling(filtered_data, mean_rev_type = self.Mean_Rev_Type, momentum_type = self.Momentum_Type, functional_constraints = self.Functional_Constraints)

        assets_to_buy, _ = analysis.extract_assets(buy_and_sells, self.Functional_Constraints.hurst_exponents_period)
        print("Assets to buy:", assets_to_buy)

        if len(assets_to_buy) > 0:
            buy_array, _, _, _ = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value = 0)
            print("Buy array:", buy_array)

            target_weights = [[ticker, dt.today(), weight] for ticker, weight in zip(assets_to_buy, buy_array)]
            print("Target weights:", target_weights)

            if self.previous_weights is not None:
                
                alpha = tc.adjust_alpha(self.Rebalance_Constraints.distance_method, target_weights, self.previous_weights, self.Best_Delta, tomas = True)

                rebalanced_weights = []
                for prev, target in zip(self.previous_weights, target_weights):
                    ticker = prev[0]  # Assuming the ticker (str) is the same
                    date = prev[1]    # Assuming the datetime is the same
                    weight = alpha * prev[2] + (1 - alpha) * target[2]
                    rebalanced_weights.append([ticker, date, weight])
                
                print("Rebalanced weights:", rebalanced_weights)
                return rebalanced_weights

            else:
                print("No previous weights, using target weights")
                return target_weights

        else:
            print("No assets to buy")
            if self.previous_weights is not None:
                print("Using previous weights")
                return [[ticker, dt.today(), weight] for ticker, _, weight in self.previous_weights]  
            else:
                print("No previous weights, returning equal weights")
                equal_weights = [[ticker, dt.today(), 1 / len(self.closes.columns)] for ticker in self.closes.columns]
                return equal_weights
    


        