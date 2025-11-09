#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:10:57 2023

@author: skm
"""
import yahooquery as yq
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import HistGradientBoostingRegressor as GB
import warnings
from datetime import timedelta, date
import plotly.graph_objects as go
warnings.filterwarnings(action='ignore', category=UserWarning)  #ignore sklearn warning at train_test_split
import plotly.io as pio
pio.renderers.default = 'browser'
#%% get_historical_data
class PredictiveModel:
    def __init__(self):
        pass
    
    def get_historical_data(self,ticker):
        """use yahooquery to pull historical data based on ticker, period requested, and interval requested"""
        
        # pull data for required period and create custom columns
        main = yq.Ticker(ticker).history(period='max',interval='1d').reset_index() # get all data at max time for sma calcs
        ten = yq.Ticker('^tnx').history(period='max',interval='1d').reset_index().rename(columns={'close':'tenyr'})
        main = main.merge(ten[['date','tenyr']],on='date',how='left')
        
        main = main.assign(
            adjclosesma = lambda x: round(x['adjclose'].rolling(30).mean(),2),
            highfib = lambda x: round(x['adjclose'].rolling(200).max() - ((x['adjclose'].rolling(200).max() -x['adjclose'].rolling(200).min())*0.236),2),
            lowfib = lambda x: round(x['adjclose'].rolling(200).min(),2) + \
            ((round(x['adjclose'].rolling(200).max(),2)-round(x['adjclose'].rolling(200).min(),2))*0.236),
            adjclosestd = lambda x: abs(round(x['adjclose'].rolling(200).std(),4)),
            tenyrsma = lambda x: round(x['tenyr'].rolling(30).mean(),2),
            tenyrstd = lambda x: abs(round(x['tenyr'].rolling(30).std(),4)),
            tenyrsmastd = lambda x: x['tenyrstd']*x['tenyrsma'],
            twohundredsma = lambda x: x['adjclose'].rolling(200).mean(),
            fiftysma = lambda x: x['adjclose'].rolling(50).mean(),
            higher_twohundred = lambda x: np.where(x['adjclose']>x['twohundredsma'],1,0),
            higher_fifty = lambda x:  np.where(x['adjclose']>x['fiftysma'],1,0),
            volumestd = lambda x: abs(x['volume'].rolling(30).std()),
            volumesma = lambda x: x['volume'].rolling(30).mean(),
            volumesmastd = lambda x: np.log(x['volumesma'] * x['volumestd']),
            adjclosesmastd = lambda x: x['adjclosesma'] * x['adjclosestd'],
            #twohundred_sma_diff_pct = lambda x: round(x['adjclose']/x['twohundredsma']),
           # twohundred_sma_diff_doll = lambda x: round(x['twohundredsma']-x['adjclose']),
            gainthirty = lambda x: ((x['adjclose']/x['adjclose'].shift(30))-1)*100,
            gainseven = lambda x: ((x['adjclose']/x['adjclose'].shift(7))-1)*100,
           # gainday = lambda x: ((x['adjclose']/x['adjclose'].shift(1))-1)*100,
            logclose = lambda x: np.log(x['adjclose']),
            logten = lambda x: np.log(x['tenyr'])
             )
        
        # select columns
        final = main[['date',
                      'symbol',
                      'adjclose',
                      'logclose',
                      #'tenyr',
                      #'adjclosesma',
                      #'highfib',
                      'higher_fifty',
                      'higher_twohundred',
                      #'lowfib',
                      'adjclosestd',
                      #'adjclosesmastd',
                     # 'tenyrsma',
                     # 'tenyrstd',
                     # 'tenyrsmastd',
                      #'twohundredsma',
                      #'fiftysma',
                     # 'volumesma',
                     # 'volumestd',
                     # 'volumesmastd',
                     # 'gainday',
                      'gainthirty',
                      'gainseven',
                      #'targetadjclose',
                      #'twohundred_sma_diff_doll',
                      #'twohundred_sma_diff_pct'
                      ]]
        
        final = round(final,2)
        return final
    
    
    def add_days(self,dataframe,days:int):
        temp = dataframe.copy()
        for i in range(1,days+1):
            temp[f'{i}_out'] = temp['adjclose'].shift(-i)
            yield temp
            
            
    def score_model(self,x_train,x_test,y_train,y_test,y_hat,dataframe,ticker,model,dte):
        df = dataframe.copy()
        global mape
        global testpreds
        testpreds = x_test.copy() # new df of x_test to join with yhat and date
        testpreds['yhat'] = y_hat # yhat will be ordered by x_test
        testpreds['actual'] = y_test # y_testw ill be ordered by x_test
        testpreds = testpreds.join(df['date'],how='left').sort_values('date').dropna() #join date on index 
        testpreds['residual'] = testpreds['yhat'] - testpreds['actual']
        testpreds['sqres'] = testpreds['residual']**2
        testpreds['ssr'] = (testpreds['yhat'] - testpreds['actual'].mean())**2
        mape = round(np.mean(np.abs((testpreds['actual'] - testpreds['yhat']) / testpreds['actual'])) * 100,2)
        
        
      #  print(f"PREDICTION ON TODAY'S {ticker} DATA in {dte} DTE")
        current_df = df[(df[f'{dte}_out'].isna()) * (df['date']==max(df['date']))]
        current_df.insert(1,'dte',dte)
        to_drop = [i for i in current_df.columns if "out" in i]
        to_drop.remove(f'{dte}_out')
        current_df = current_df.drop(columns=to_drop)
        curr_x = current_df.drop(columns=['date','dte',f'{dte}_out','symbol','adjclose'])
        #print(curr_x.columns)
        currpreds = list(map(lambda x: round(x,2),model.predict(curr_x)))
        current_df.insert(3,'yhat',currpreds)
        current_df = current_df.assign(
            yhat_low = lambda x: round(x['yhat']*(1-((mape*0.5)/100)),2),
            yhat_high = lambda x: round(x['yhat']*(1+((mape*0.5)/100)),2))
        return current_df[['dte','yhat','yhat_low','yhat_high']]
        
    
    def run_regression(self,ticker,dte,vizualize:bool=True):
      
        dte_yhats = pd.DataFrame()
        originaldata = self.get_historical_data(ticker)
        main = list(self.add_days(originaldata,dte))[-1]
        
        for i in range(1,dte+1):
            to_drop = [i for i in main.columns if "out" in i]
            to_drop.remove(f'{i}_out')
            df = main.drop(columns=to_drop)
            df = df.dropna().reset_index(drop=True)
            x_train, x_test, y_train, y_test = tts(df.drop(columns=['date',
                                                                    #'targetadjclose',
                                                                    'symbol',
                                                                    'adjclose',
                                                                    f'{i}_out']),
                                                                    df[f'{i}_out'],
                                                                    test_size=0.01,
                                                                    shuffle=False,
                                                                    random_state=42)
            model =  GB() #tree.DecisionTreeRegressor(max_depth=10,min_samples_split=5,criterion='squared_error')
            model = model.fit(x_train, y_train)
        
            y_hat = [round(x,2) for x in model.predict(x_test)]
            new_yhat = self.score_model(x_train,x_test,y_train,y_test,y_hat,main,ticker,model,i)
            dte_yhats = pd.concat([dte_yhats,new_yhat])
            dte_yhats['date'] = dte_yhats['dte'].apply(lambda x: timedelta(days=x)+date.today())

        if vizualize:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dte_yhats['dte'],
                                    y=dte_yhats['yhat'],
                                    line_color='dodgerblue',
                                    name='Prediction'))
            fig.add_trace(go.Scatter(x=dte_yhats['dte'],
                                    y=dte_yhats['yhat_high'],
                                    line_color='green',
                                    name='High-End'))
            fig.add_trace(go.Scatter(x=dte_yhats['dte'],
                                    y=dte_yhats['yhat_low'],
                                    line_color='firebrick',
                                    name='Low-End'))
            fig.update_xaxes(type='category',title_text='Days From Today')
            fig.update_layout(title_text=f'{ticker.upper()} Daily Predictions')
        
            return dte_yhats,fig.show()
        else:
            return dte_yhats
            
    
