import src.data_analysis as analysis
import src.signalling as signalling
import src.data_models as dm
import src.portfolio_weights as pw
from datetime import datetime as dt
import src.tc_optimization as tc
import pandas as pd



class PortfolioOptimization():

    def __init__(self, best_delta: float, mom_type: dm.Momentum_Type, mean_rev_type: dm.Mean_Rev_Type,
                  rebalancing_period: dm.Rebalancing_Period,
                  functional_constraints: dm.Functional_Constraints, rebalance_constraints: dm.Rebalance_Constraints,
                  mom_days: int = 30,
                  closes: pd.DataFrame = None,
                  previous_weights = None):
        self.closes = closes
        self.last_date = closes.index[-1]
        self.previous_weights = previous_weights
        self.best_delta = best_delta
        self.momentum_type = mom_type
        self.mean_rev_type = mean_rev_type
        self.rebalancing_period = rebalancing_period
        self.functional_constraints = functional_constraints
        self.rebalance_constraints = rebalance_constraints
        self.momentum_days = mom_days

    def get_weights(self):

        # Gerar sinais
        filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(  self.closes,
                                                                                                                self.rebalancing_period, 
                                                                                                                self.functional_constraints.hurst_exponents_period, 
                                                                                                                self.mean_rev_type,
                                                                                                                self.momentum_type, 
                                                                                                                self.functional_constraints, 
                                                                                                                self.momentum_days,
                                                                                                                live_analysis = True),
                                                                                                            hurst_thresholds=self.functional_constraints.hurst_filter,
                                                                                                            mean_rev_type= self.mean_rev_type,
                                                                                                            momentum_type= self.momentum_type,
                                                                                                            live_analysis=True)
        
        buy_and_sells = signalling.buy_and_sell_signalling(filtered_data, mean_rev_type = self.mean_rev_type, momentum_type = self.momentum_type, functional_constraints = self.functional_constraints)

        assets_to_buy, _ = analysis.extract_assets(buy_and_sells, self.functional_constraints.hurst_exponents_period)
        print("Assets to buy:", assets_to_buy)

        if len(assets_to_buy) > 0:
            buy_array, _, _, _ = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value = 0)
            print("Buy array:", buy_array)

            target_weights = [[ticker, self.last_date, weight] for ticker, weight in zip(assets_to_buy, buy_array)]
            print("Target weights:", target_weights)

            if self.previous_weights is not None:
                
                alpha = tc.adjust_alpha(self.rebalance_constraints.distance_method, target_weights, self.previous_weights, self.Best_Delta, tomas = True)

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
                return [[ticker, self.last_date, weight] for ticker, _, weight in self.previous_weights]  
            else:
                print("No previous weights, returning equal weights")
                equal_weights = [[ticker, self.last_date, 1 / len(self.closes.columns)] for ticker in self.closes.columns]
                return equal_weights
    


        