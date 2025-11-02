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
        hist['returns'] = np.log(hist['close']) - np.log(hist['close'].shift(1))
        if make_ohlc:
            self.__make_ohlc(hist)
        return hist

    def get_10_year(self):
        """get 10-year treasury from Yahoo Finance"""
        ten_yr = self.yf.history().reset_index()['close'].iloc[-1]
        treas = round(ten_yr/100,4)
        return treas
    

    def dte_price(self)->pd.DataFrame:
        prices = np.arange(0.95*self.K,self.K*1.05,0.10)
        if self.T<=10:
            days = range(1,self.T+1)
        else:
            days = range(1,self.T+1,7)
        days = [day-1 for day in days]
        paired = [(price,day) for price in prices for day in days]
        prices_results = list(map(lambda p: (p[0],self.K,self.r,p[1],self.sd,self.option_type),paired))
        plot_df = pd.DataFrame(paired,columns=['price','dte'])
        plot_df['option_value'] = prices_results
        return plot_df

    def blackscholes(self,
                    S:float | int, 
                    K:float | int, 
                    T:int, 
                    option_type:str,
                    plot_data:bool=True)->float:
        """
        S = Stock Price

        K = Strike Price at Expiration 

        T = Time to Expiration

        option_type = call or put
        """
        self.K = K
        self.S = S
        self.option_type = option_type.lower()
        allowed_option_types = {"call", "put"}
        if self.option_type not in allowed_option_types:
            raise ValueError(f"Invalid option type: '{option_type}'. Option Ttpe must be one of {allowed_option_types}.")
        else:
            df = self.get_historical_data() # get daily ticker data from last 30 days

            px.histogram(df['close'],title=f'{self.ticker.upper()} Close Price Last 30 Days').show()
            px.histogram(df['returns'],title=f'{self.ticker.upper()} Log Returns Last 30 Days').show() #histogram of log returns
            
            self.T = T/365
            # get option data for analysis
            self.sd = round(df['returns'].std() * sqrt(252),2) # annualize stdev of log daily returns based on # of annual trading days
            self.r = self.get_10_year()
            self.price = round(df['close'].iloc[-1],2)
            
 
            print([i for i in np.arange(int(self.price*.75),int(self.price*1.25),0.50)])
            d1 = (np.log(S/K) + (self.r + self.sd**2/2)* T)/(self.sd*np.sqrt(T))
            d2 = d1 - self.sd * np.sqrt(T)

            if option_type == "call":
                value = S * norm.cdf(d1, 0, 1) - K * np.exp(-self.r * T) * norm.cdf(d2, 0, 1)
            elif option_type == "put":
                value = K * np.exp(-self.r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
            
            if plot_data:
                params = {'Price: ':self.price,'Annual Volatility: ':self.sd,'10-yr: ':f'{round(self.r*100,2)}%',"Black-Scholes Value: ":value}
                for i in params.items():
                    print(f"{i[0]} {i[1]}")
                    
                prices_df = self.dte_price()
                # print(prices_df)

                px.line(prices_df,x='price',y='option_value',color='dte',title=f'{self.ticker.upper()} {K} dte {option_type}')


                go.Figure(data=go.Heatmap(
                                z=prices_df['option_value'],
                                x=prices_df['dte'],
                                y=prices_df['price'],
                                hoverongaps = True),
                                )
            return round(value,2)
        
    def __make_ohlc(self,df):
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
   
    class FinViz:
        def __init__(self):
            self.ticker = Simplifi.ticker
            
        
        def fundamentals(self):
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
        
        def ratings(self):
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
        

    class DDM:
        def __init__(self):
            self.ticker = Simplifi.ticker
            self.yf = Simplifi.yf
            pass
        
        def get_ddm_val(self):
            last_yr_rate = self.yf.asset_profile("trailingAnnualDividendRate")
            last_div_rate = self.yf.asset_profile("dividendRate")
            try:
                growth = (last_div_rate-last_yr_rate)/last_yr_rate
            except ValueError as e:
                print("Dividend growth is incalculable due to lack of historical dividend payment")
            nxt_yr_div = self.yf.asset_profile("dividendRate") * (1+growth)
            ddm_val = nxt_yr_div/(self.get_cost_of_equity()-growth)
            return ddm_val

                
                
        def get_cost_of_equity(self):
            beta = self.yf.asset_profile('beta')
            if type(beta) is str: # string-based tickers are typically new and associated risk is high
                beta=1.75 # 
                
            rm= 0.085
            rfr = Simplifi.get_10_year()
            re= rfr+beta*(rm-rfr)
            print("Valuation based on the following:")
            print(f"Risk Free Rate: {round(rfr*100,2)}%")
            print(f"Expected Market Return: {round(rm*100,2)}%")
            return re
            

            

if __name__ == '__main__':
    simplifi = Simplifi('AAPL')
    simplifi.blackscholes(200.00,190.00,7,'call',True)
    simplifi.get_historical_data()
    simplifi.FinViz.ratings()
    simplifi.FinViz.fundamentals()




# %%
