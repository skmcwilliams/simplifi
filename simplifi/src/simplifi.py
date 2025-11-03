# %%
import numpy as np
from math import sqrt
from scipy.stats import norm
import yahooquery as yq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
from bs4 import BeautifulSoup as bs
import requests
from urllib.request import urlopen
import json
from sklearn import tree, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as tts,cross_val_score  as cvs, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor as GB
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.classifier import confusion_matrix,precision_recall_curve,classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)  #ignore sklearn warning at train_test_split
# %%
class Simplifi:
    def __init__(self,ticker:str):
        self.ticker = ticker
        self.yf = yq.Ticker(self.ticker)

    def get_historical_data(self,make_ohlc:bool=False):
        hist = self.yf.history(period='30d',interval='1d').reset_index()
        hist['date'] = list(map(str,hist['date']))
        hist['Day'] = hist['date'].apply(lambda x: x.split()[0])
        hist['avg_price'] = (hist['high']+hist['close']+hist['low'])/3
        hist['returns'] = np.log(hist['close'])/np.log(hist['close'].shift(1))
        if make_ohlc:
            self.__make_ohlc(hist)
        return hist

    def get_10_year(self)->float:
        """get 10-year treasury from Yahoo Finance"""
        ten_yr = self.yf.history().reset_index()['close'].iloc[-1]
        treas = round(ten_yr/100,4)
        return treas

    def blackscholes(self)->float:
        """
        Black-Scholes option pricing model based on current options chain from Yahoo Finance
        """
        
        df = self.get_historical_data() # get daily ticker data from last 30 days
        px.histogram(df['close'],title=f'{self.ticker.upper()} Close Price Last 30 Days').show()
        px.histogram(df['returns'],title=f'{self.ticker.upper()} Log Returns Last 30 Days').show() #histogram of log returns
        options = simplifi.yf.option_chain.reset_index()
        options['dte'] = (pd.to_datetime(options['expiration']) - pd.to_datetime("now")).dt.days
        options['t'] = options['dte']/365
        # get option data for analysis
        self.sd = round(df['returns'].std() * sqrt(252),2) # annualize stdev of log daily returns based on # of annual trading days
        self.r = self.get_10_year()
        options['d1'] = (np.log(options['lastPrice']/options['strike']) + (self.r + self.sd**2/2)* options['t'])/(self.sd*np.sqrt(options['t']))
        options['d2'] = options['d1'] - self.sd * np.sqrt(options['t'])
        options['black_scholes_value'] = np.where(options['optionType']=="calls",options['lastPrice']* norm.cdf(options['d1'], 0, 1) - \
                                                    options['strike'] * np.exp(-self.r * options['t']) * norm.cdf(options['d2'], 0, 1),np.where(
                                                        options['optionType']=='puts',options['strike'] * np.exp(-self.r * options['t']) * \
                                                        norm.cdf(-options['d2'], 0, 1) - options['lastPrice'] * norm.cdf(-options['d1'], 0, 1),np.nan))

        return options
        
    def __make_ohlc(self,df)->go.Figure:
        ohlc_fig = make_subplots(specs=[[{"secondary_y": True}]]) # creates ability to plot vol and $ change within main plot
    
        #include OHLC (already comes with rangeselector)
        ohlc_fig.add_trace(go.Candlestick(x=df['date'],
                        open=df[f'{self.ticker}_open'], 
                        high=df[f'{self.ticker}_high'],
                        low=df[f'{self.ticker}_low'], 
                        close=df[f'{self.ticker}_close'],name='Daily Candlestick'),secondary_y=True)
        
        ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{self.ticker}_200_sma'],name='200-day SMA',line=dict(color='cyan')),secondary_y=True)
        ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{self.ticker}_50_sma'],name='50-day SMA',line=dict(color='navy')),secondary_y=True)
        
        # include a go.Bar trace for volume
        ohlc_fig.add_trace(go.Bar(x=df['date'], y=df[f'{self.ticker}_volume'],name='Volume'),
                        secondary_y=False)
    
        ohlc_fig.layout.yaxis2.showgrid=False
        ohlc_fig.update_xaxes(type='category')
        ohlc_fig.update_layout(title_text=f'{self.ticker} Price Chart')
        ohlc_fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        return ohlc_fig.show()
            
        
    def finviz_fundamentals(self)->pd.DataFrame:
        try:
            url = f'https://www.finviz.com/quote.ashx?t={self.ticker.lower()}'
            request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
            soup = bs(request.text, "lxml")
            stats = soup.find('table',class_='snapshot-table2')
            fundamentals =pd.read_html(str(stats))[0]
            
            # Clean up fundamentals dataframe
            fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
            colOne = []
            colLength = len(fundamentals)
            for k in np.arange(0, colLength, 2):
                colOne.append(fundamentals[f'{k}'])
            attrs = pd.concat(colOne, ignore_index=True)
        
            colTwo = []
            colLength = len(fundamentals)
            for k in np.arange(1, colLength, 2):
                colTwo.append(fundamentals[f'{k}'])
            vals = pd.concat(colTwo, ignore_index=True)
            
            fundamentals = pd.DataFrame()
            fundamentals['Attributes'] = attrs
            fundamentals[f'{self.ticker.upper()}'] = vals
            fundamentals = fundamentals.set_index('Attributes')
            fundamentals = fundamentals.T
            
            # catch known duplicate column name EPS next Y
        # fundamentals.rename(columns={fundamentals.columns[28]:'EPS growth next Y'},inplace=True)
            return fundamentals
    
        except Exception as e:
            return e
        
        
    def finviz_ratings(self)->pd.DataFrame:
        url = f'https://www.finviz.com/quote.ashx?t={self.ticker.lower()}'
        request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs(request.text, "lxml")
        stats = soup.find('table',class_='fullview-ratings-outer')
        ratings =pd.read_html(str(stats))[0]
        ratings['date'] = ratings[0].apply(lambda x: x.split()[0][:9])
        ratings['rating'] = ratings[0].apply(lambda x: x.split()[0][9:])
        ratings['firm'] = ratings[0].apply(lambda x: x.split()[1])
        ratings.drop(columns=0,inplace=True)
        return ratings
        
        
    def ddm_valuation(self)->float:
        last_yr_rate = self.asset_profile("trailingAnnualDividendRate")
        last_div_rate = self.asset_profile("dividendRate")
        try:
            growth = (last_div_rate-last_yr_rate)/last_yr_rate
        except ValueError as e:
            print("Dividend growth is incalculable due to lack of historical dividend payment")
        nxt_yr_div = self.yf.asset_profile("dividendRate") * (1+growth)
        ddm_val = nxt_yr_div/(self.get_cost_of_equity()-growth)
        return ddm_val
 
                
    def __get_cost_of_equity(self)->float:
        beta = self.asset_profile('beta')
        if type(beta) is str: # string-based tickers are typically new and associated risk is high
            beta=1.75 
        rm= 0.085 # static expected market return of 8.5%
        rfr = Simplifi.get_10_year()
        re= rfr+beta*(rm-rfr)
        print("Valuation based on the following:")
        print(f"Risk Free Rate: {round(rfr*100,2)}%")
        print(f"Expected Market Return: {round(rm*100,2)}%")
        return re
    
#%%
    warnings.filterwarnings(action='ignore', category=UserWarning)  #ignore sklearn warning at train_test_split
    #%% get_historical_data
    def get_historical_data(ticker,dte):
        """use yahooquery to pull historical data based on ticker, period requested, and interval requested"""
        
        global d
        d = dte
        # pull data for required period and create custom columns
        main = yq.Ticker(ticker).history(period='max',interval='1d').reset_index() # get all data at max time for sma calcs
        ten = yq.Ticker('^tnx').history(period='max',interval='1d').reset_index().rename(columns={'close':'tenyr'})
        main = main.merge(ten[['date','tenyr']],on='date',how='left')
        
        main = main.assign(
            targetadjclose = lambda x: x['adjclose'].shift(-dte),
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
        
        start = np.random.choice(range(0,len(main)))
        assert main['targetadjclose'].iloc[start]==main['adjclose'].iloc[start+dte],\
        f"error at {start}:{main['targetadjclose'].iloc[start]=}, {start+100}:{main['adjclose'].iloc[start+30]}"


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
                    'targetadjclose',
                    #'twohundred_sma_diff_doll',
                    #'twohundred_sma_diff_pct'
                    ]]
        
        final = round(final,2)
        return final
    
    def score_model(x_train,x_test,y_train,y_test,y_hat,dataframe,ticker,model,dte):
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
    
        # print(f"Model Results on Test Data {ticker} {dte} DTE \n")
        sse =  round(testpreds['sqres'].sum(),2)            
        ssr = round(testpreds['ssr'].sum(),2)
        tss = round(ssr+sse,2)
        rsquared = round(ssr/tss,3)
        mse = round(mean_squared_error(y_test,y_hat),2)
        rmse = round(np.sqrt(mse),2)
        normrmse = round(rmse/(df['targetadjclose'].max() - df['targetadjclose'].min()),3) # is extremeley low, rmse is strong relative to 'close' range
        measures = ['SSE', 'SSR', 'TSS', 'r2', 'MSE','RMSE','Normalized RMSE','MAPE']
        results = [sse,ssr,tss,rsquared,mse,rmse,normrmse,f'{mape}%']
        resdf = pd.DataFrame(list(zip(measures,results)),columns=['Metric','Score'])
        
        
    #  print(f"PREDICTION ON TODAY'S {ticker} DATA in {dte} DTE")
        current_df = df[(df['targetadjclose'].isna()) * (df['date']==max(df['date']))]
        current_df.insert(1,'dte',dte)
        curr_x = current_df.drop(columns=['date','dte','targetadjclose','symbol','adjclose'])
        currpreds = list(map(lambda x: round(x,2),model.predict(curr_x)))
        current_df.insert(3,'yhat',currpreds)
        current_df = current_df.assign(
            yhat_low = lambda x: round(x['yhat']*(1-((mape*0.5)/100)),2),
            yhat_high = lambda x: round(x['yhat']*(1+((mape*0.5)/100)),2))
        return current_df[['dte','yhat','yhat_low','yhat_high']]
        

    def run_regression(ticker,dte):
    
        dte_yhats = pd.DataFrame()
        for i in range(1,dte+1):
        
            main = get_historical_data(ticker, i)
            df = main.dropna().reset_index(drop=True)
            x_train, x_test, y_train, y_test = tts(df.drop(columns=['date','targetadjclose','symbol','adjclose']),df['targetadjclose'],
                                                        test_size=0.01,shuffle=False,random_state=42)
            model =  GB() #tree.DecisionTreeRegressor(max_depth=10,min_samples_split=5,criterion='squared_error')
            model = model.fit(x_train, y_train)
        
        
            y_hat = [round(x,2) for x in model.predict(x_test)]
            new_yhat = score_model(x_train,x_test,y_train,y_test,y_hat,main,ticker,model,i)
            dte_yhats = pd.concat([dte_yhats,new_yhat])
            
        plt.plot(dte_yhats['dte'],dte_yhats['yhat'])
        plt.plot(dte_yhats['dte'],dte_yhats['yhat_high'])
        plt.plot(dte_yhats['dte'],dte_yhats['yhat_low'])
        plt.legend(['Yhat','Yhat-High','Yhat-Low'])
        plt.title(f"{ticker.upper()} Model Predictions")
        plt.show()
        
        print(dte_yhats)
            
    
        
#%%RUN Regression
run_regression('itot',20)
            

            
#%% Test Cases
if __name__ == '__main__':
    simplifi = Simplifi('AAPL')
    simplifi.blackscholes()
    simplifi.get_historical_data()
    simplifi.finviz_ratings()
    simplifi.finviz_fundamentals()




# %%
