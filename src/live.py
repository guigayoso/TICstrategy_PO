import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta

import data_models as dm
import data.data_collection as dc
import data_analysis as analysis
import signalling
import portfolio_weights as pw
import tc_optimization as tc

import time

class Live:
    def __init__(self, 
                 Best_Delta: float,
                 Asset_Category: dm.Asset_Category,
                 Momentum_Type: dm.Momentum_Type,
                 Mean_Rev_Type: dm.Mean_Rev_Type,
                 Rebalancing_Period: dm.Rebalancing_Period,
                 Functional_Constraints: dm.Functional_Constraints,
                 Rebalance_Constraints: dm.Rebalance_Constraints,
                 Momentum_Days: int
                    ):

                    self.Best_Delta = Best_Delta
                    self.Asset_Category = Asset_Category
                    self.Momentum_Type = Momentum_Type
                    self.Mean_Rev_Type = Mean_Rev_Type
                    self.Rebalancing_Period = analysis.get_rebalancing_period_days(Rebalancing_Period)
                    self.Functional_Constraints = Functional_Constraints
                    self.Rebalance_Constraints = Rebalance_Constraints
                    self.target_weights = pd.DataFrame()
                    self.rebalancing_weights = pd.DataFrame()
                    self.last_rebalancing_date = None
                    self.momentum_days = Momentum_Days

                    self.stop_loss = self.Functional_Constraints.get_stop_loss()
                    self.take_profit = self.Functional_Constraints.get_take_profit()
                    self.hurst_exponents_period = self.Functional_Constraints.hurst_exponents_period
                    self.hurst_thresholds = self.Functional_Constraints.hurst_filter

                    #self.first_weights()


    def get_live_data(self):
        pass

    def first_weights(self):
        
        dcollection = dc.DataProcessor(self.Asset_Category, None, dt.today() - timedelta(days=self.Functional_Constraints.hurst_exponents_period))
        closes = dcollection.get_asset_close_prices()

        print(f"First Closes: {closes}")

        # Generate first signal
        print("Mean Rev Type: ", self.Mean_Rev_Type)
        filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(closes, rebalancing_period=self.Rebalancing_Period, 
                                                                                                              hurst_exponents_period=self.hurst_exponents_period, 
                                                                                                              mean_rev_type= self.Mean_Rev_Type, momentum_type= self.Momentum_Type, 
                                                                                                              functional_constraints= self.Functional_Constraints, 
                                                                                                              momentus_days_period=self.momentum_days,
                                                                                                              live_analysis = True), hurst_thresholds=self.hurst_thresholds, 
                                                                                                              mean_rev_type= self.Mean_Rev_Type, momentum_type= self.Momentum_Type, live_analysis=True)
        buy_and_sells = signalling.buy_and_sell_signalling(filtered_data, mean_rev_type = self.Mean_Rev_Type, momentum_type = self.Momentum_Type, functional_constraints = self.Functional_Constraints)

        print(f"filtered_data: {filtered_data}")
        print(f"buy_and_sells: {buy_and_sells}")

        #print(f"First Buy and Sells: {analysis.get_data_analysis(closes, rebalancing_period=self.Rebalancing_Period, hurst_exponents_period=self.hurst_exponents_period)}")

        assets_to_buy, _ = analysis.extract_assets(buy_and_sells, self.hurst_exponents_period - self.hurst_exponents_period)

        print(f"First Assets to Buy: {assets_to_buy}")

        # First weights
        buy_array, _, _, _ = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value = 0)

        #self.weights = buy_array

        # Update weights df
        self.target_weights = pd.DataFrame(columns=assets_to_buy)
        self.rebalancing_weights = pd.DataFrame(columns=assets_to_buy)

        print(f"First Weights: {buy_array}")

        pw.update_weights_df(self.target_weights, dt.today().replace(microsecond=0), assets_to_buy, buy_array)
        pw.update_weights_df(self.rebalancing_weights, dt.today().replace(microsecond=0), assets_to_buy, buy_array)

        self.last_rebalancing_date = dt.today() # dia 180

        return self.rebalancing_weights
    
    def get_current_weights(self):
        # Vai poder ser chamado sempre, não só no dia de rebalancing
        # Pegar no ultimo dia de rebalancing e evoluir ate today

        dcollection = dc.DataProcessor(self.Asset_Category, None, self.last_rebalancing_date)
        closes = dcollection.get_asset_close_prices()
        print(f" Closes: {closes}")

        cumulative_returns = (1 + closes.pct_change()).cumprod() - 1
        evolution = cumulative_returns.iloc[-1]

        new_contributions = self.rebalancing_weights.loc[self.last_rebalancing_date] * (1 + evolution)
        unbalanced_weights = new_contributions / sum(new_contributions)

        #self.weights = unbalanced_weights
        print(f" Weights Updated: {unbalanced_weights}")

        #if dt.today() == self.last_rebalancing_date + timedelta(days=self.Rebalancing_Period):
        #    self.last_rebalancing_date = dt.today()
        return unbalanced_weights
    
    def check_pt_sl_trigger(self, closes, current_weights):
        """
        Check if the stop loss or take profit conditions are met for each asset.
        Parameters:
            closes (pd.DataFrame): DataFrame with adjusted close prices for different assets since the beginning of the rebalancing period.
            current_weights (pd.Series): Current weights of the assets in the portfolio.
        Returns:
            pd.Series: Updated weights after applying stop loss and profit take.
        """
        cumulative_returns = (closes / closes.iloc[0]) - 1 # Cumulative returns since the beginning of the rebalancing period
        updated_weights = current_weights.copy()

        for asset in current_weights.index:
            if cumulative_returns[asset].iloc[-1] < -self.Functional_Constraints.Stop_Loss:
                updated_weights[asset] = 0
            elif cumulative_returns[asset].iloc[-1] > self.Functional_Constraints.Take_Profit:
                updated_weights[asset] = 0
            else:
                updated_weights[asset] = current_weights[asset]
        updated_weights = updated_weights / updated_weights.sum()
        return updated_weights

    def rebalancing(self, assets_to_buy):
        # Vai ser chamado só quando houver sinais a cada rebalancing period days
            
        buy_array, _, _, _ = pw.calculate_uniform_weights(assets_to_buy, [], shorting_value = 0)

        pw.update_weights_df(self.target_weights, dt.today(), assets_to_buy, buy_array)

        unbalanced_weights = self.get_current_weights()

        alpha = tc.adjust_alpha(self.Rebalance_Constraints.distance_method, self.target_weights.iloc[-1], unbalanced_weights, self.Best_Delta)

        rebalanced_weights = alpha * unbalanced_weights + (1 - alpha) * self.target_weights.iloc[-1]

        pw.update_weights_df(self.rebalancing_weights, dt.today(), assets_to_buy, rebalanced_weights)
            
        self.last_rebalancing_date = dt.today()

    def execute_trade(self):
        pass

    def run_live(self):
        
        self.first_weights()

        print("Running Live Strategy")
        print(f"First Weights: {self.rebalancing_weights}")

        while True:
             
            if dt.today().date() == (self.last_rebalancing_date + timedelta(days=self.Rebalancing_Period)).date():

                dcollection = dc.DataProcessor(self.Asset_Category, None, dt.today() - timedelta(days=self.Functional_Constraints.hurst_exponents_period))
                closes = dcollection.get_asset_close_prices()

                filtered_data, trendy_assets, mean_reverting_assets = analysis.filter_data(analysis.get_data_analysis(closes, rebalancing_period=self.Rebalancing_Period, 
                                                                                                                      hurst_exponents_period=self.hurst_exponents_period, mean_rev_type= self.Mean_Rev_Type, 
                                                                                                                      momentum_type= self.Momentum_Type, functional_constraints= self.Functional_Constraints,
                                                                                                                      momentus_days_period=self.momentum_days,
                                                                                                                      live_analysis = True), 
                                                                                                                      hurst_thresholds=self.hurst_thresholds)
                buy_and_sells = signalling.buy_and_sell_signalling(filtered_data, mean_rev_type = self.Mean_Rev_Type, momentum_type = self.Momentum_Type, functional_constraints = self.Functional_Constraints)

                assets_to_buy, _ = analysis.extract_assets(buy_and_sells, self.hurst_exponents_period) # mudar para ficar: 7, 14, 21 ou 30, 60, 90

                if len(assets_to_buy) > 0: 
                    print(f"Assets to Buy: {assets_to_buy} on date {dt.today()}")
                    self.rebalancing(assets_to_buy) # Vai me dar os pesos ideais para a iteracao atual
                else:
                    print(f"No Assets to Buy on date {dt.today()}")
                    unbalanced_weights = self.get_current_weights() # Caso nao tenha target weughts para a iteracao atual, preencher tabela de pesos ideais com a evolucao dos ideais da iteracao passada

                    pw.update_weights_df(self.target_weights, dt.today(), assets_to_buy, np.nan)
                    pw.update_weights_df(self.rebalancing_weights, dt.today(), assets_to_buy, unbalanced_weights)

                    self.last_rebalancing_date = dt.today()
            
            # Track Weights
            if dt.today().date() != self.last_rebalancing_date.date():
                self.get_current_weights()
            #self.execute_trade()
            time.sleep(86400) # 1 day
            #break


                 