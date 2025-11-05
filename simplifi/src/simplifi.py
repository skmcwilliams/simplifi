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
import requests
from bs4 import BeautifulSoup as bs
from io import StringIO

class Simplifi:
    def __init__(self,ticker:str):
        self.ticker = ticker
        self.yf = yq.Ticker(self.ticker)

    def get_historical_data(self,make_ohlc:bool=False):
        hist = self.yf.history(period='10y',interval='1d').reset_index()
        hist['date'] = list(map(str,hist['date']))
        hist['Day'] = hist['date'].apply(lambda x: x.split()[0])
        hist['avg_price'] = (hist['high']+hist['close']+hist['low'])/3
        hist['200_ma'] = hist['close'].rolling(window=200).mean()
        hist['50_ma'] = hist['close'].rolling(window=50).mean()
        hist['log_returns'] = np.log(hist['close'])/np.log(hist['close'].shift(1))
        if make_ohlc:
            self.__make_ohlc(hist)
        return hist

    def get_10_year(self)->float:
        """get 10-year treasury from Yahoo Finance"""
        ten = yq.Ticker('^TNX')
        ten_yr = ten.history().reset_index()
        ten_yr = ten_yr['close'].iloc[-1]
        treas = round(ten_yr/100,4)
        return treas

    def blackscholes(self)->pd.DataFrame:
        """
        Black-Scholes option pricing model based on current options chain from Yahoo Finance
        """
        
        df = self.get_historical_data() # get daily ticker data from last 30 days
        df = df.sort_values(by='date',ascending=True).reset_index()
        df = df.tail(30) # lasty 30 trading days for stdev calc
        options = simplifi.yf.option_chain.reset_index()
        options['dte'] = (pd.to_datetime(options['expiration']) - pd.to_datetime("now")).dt.days
        options['t'] = options['dte']/365
        # get option data for analysis
        self.sd = round(df['log_returns'].std() * sqrt(252),2) # annualize stdev of log daily returns based on # of annual trading days
        self.r = self.get_10_year()
        options['d1'] = (np.log(options['lastPrice']/options['strike']) + (self.r + self.sd**2/2)* options['t'])/(self.sd*np.sqrt(options['t']))
        options['d2'] = options['d1'] - self.sd * np.sqrt(options['t'])
        options['black_scholes_value'] = np.where(options['optionType']=="calls",options['lastPrice']* norm.cdf(options['d1'], 0, 1) - \
                                                    options['strike'] * np.exp(-self.r * options['t']) * norm.cdf(options['d2'], 0, 1),np.where(
                                                        options['optionType']=='puts',options['strike'] * np.exp(-self.r * options['t']) * \
                                                        norm.cdf(-options['d2'], 0, 1) - options['lastPrice'] * norm.cdf(-options['d1'], 0, 1),np.nan))
        options = options[['symbol',
                           'expiration',
                           'dte',
                           'inTheMoney',
                           'optionType',
                           'contractSymbol',
                           'strike',
                           'lastPrice',
                           'change',
                           'percentChange',
                           'volume',
                           'openInterest',
                           'bid',
                            'ask',
                            'impliedVolatility',
                            'contractSize',
                            'lastTradeDate',
                           'black_scholes_value',
                           ]]
        return options
        
    def __make_ohlc(self,df)->go.Figure:
        ohlc_fig = make_subplots(specs=[[{"secondary_y": True}]]) # creates ability to plot vol and $ change within main plot
        df = df[df['volume']>0] # filter out 0 volume days
        #include OHLC (already comes with rangeselector)
        ohlc_fig.add_trace(go.Candlestick(x=df['date'],
                        open=df[f'open'], 
                        high=df[f'high'],
                        low=df[f'low'], 
                        close=df[f'close'],name='Daily Candlestick'),secondary_y=True)
    
        # include a go.Bar trace for volume
        ohlc_fig.add_trace(go.Bar(x=df['date'], 
                                  y=df['volume'],
                                  name='Volume',
                                  marker_color='MediumPurple'),
                        secondary_y=False)
        
        ohlc_fig.add_trace(go.Scatter(x=df['date'], 
                                  y=df['200_ma'],
                                  name='200-day MA',
                                  marker_color='Cyan'),
                        secondary_y=True)
        
        ohlc_fig.add_trace(go.Scatter(x=df['date'], 
                                  y=df['50_ma'],
                                  name='50-day MA',
                                  marker_color='navy'),
                        secondary_y=True)
    
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
                        dict(count=3,
                            label="3y",
                            step="year",
                            stepmode="backward"),
                        dict(count=5,
                            label="5y",
                            step="year",
                            stepmode="backward"),
                        # dict(count=10,
                        #     label="10y",
                        #     step="year",
                        #     stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        # Update the x-axis to include rangebreaks for weekends
        ohlc_fig.update_xaxes(
        rangebreaks=[
        dict(bounds=["sat", "mon"]), # hide weekends
            ]
        )       
        return ohlc_fig.show()

        
    def ddm_valuation(self)->float:
        growth = self.yf.earnings_trend[self.ticker]['trend'][0]['growth'] #EPS Growth %
        try:
            nxt_yr_div = self.yf.summary_detail[self.ticker]["dividendRate"] * (1+growth)
        except KeyError:
            print("No dividend data available for DDM valuation.")
            return None
        else:
            price = self.yf.summary_detail[self.ticker]['previousClose']
            coe = (nxt_yr_div / price) + self.yf.summary_detail[self.ticker]["dividendRate"]
            ddm_val = nxt_yr_div/(coe-growth)
            print(f"Simplifi DDM Valuation: ${round(ddm_val*100,2)}")
            return ddm_val
 
                
    def get_capm_return(self,target_return:float=None)->float:
        beta = self.yf.summary_detail[self.ticker]['beta']
        if type(beta) is str: # string-based tickers are typically new and associated risk is high
            beta=1.75
        if target_return:
            rm = target_return
        else: 
            rm = 0.085 # static expected market return of 8.5%
        rfr = self.get_10_year()
        capm = rfr+beta*(rm-rfr) # CAPM formula
        print("Valuation based on the following:")
        print(f"Risk Free Rate (10yr Treasury): {round(rfr*100,2)}%")
        print(f"Expected Market Return: {round(rm*100,2)}%")
        print(f"Simplifi CAPM Return: {round(capm*100,2)}%")
        return capm
             
# Testing
if __name__ == '__main__':
    simplifi = Simplifi('ULTA')
    simplifi.blackscholes()
    simplifi.get_historical_data(make_ohlc=True)
    simplifi.ddm_valuation()
    simplifi.get_capm_return(target_return=0.07)
