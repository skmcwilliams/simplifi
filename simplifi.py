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
from regression import PredictiveModel
pio.renderers.default = 'browser'

class Simplifi:
    def __init__(self,ticker:str):
        self.ticker = ticker
        self.yf = yq.Ticker(self.ticker)
        self.model = PredictiveModel()

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
    
    def run_regression(self,dte:int, make_viz:bool=True)->pd.DataFrame:
        dte_yhats, fig = self.model.run_regression(self.ticker,dte,vizualize=make_viz)
        return dte_yhats
    
    def make_comp_chart(self,df:pd.DataFrame)->go.Figure:
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{self.ticker}_pct_change'],name=f'{self.ticker}'))
        comp_fig.add_trace(go.Scatter(x=df['date'],y=df['SPY_pct_change'],name='SPY'))
        comp_fig.add_trace(go.Scatter(x=df['date'],y=df['DIA_pct_change'],name='DIA'))
        comp_fig.add_trace(go.Scatter(x=df['date'],y=df['QQQ_pct_change'],name='QQQ'))
        comp_fig.update_xaxes(type='category')
        comp_fig.update_layout(
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
            ),
            yaxis = dict(
                tickformat = '.0%',
                autorange=True, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
                fixedrange=False, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
                ),
            title_text=f'{self.ticker} vs. Indices Historical Prices',
        )

        return comp_fig.show()

    def pct_change(df,ticker)->pd.DataFrame:
        """read in dataframe to be modified and tickers to perfrom percent chanage on within dataframe"""
        df[f'{ticker}_pct_change']=''
        df['f{ticker}_pct_change'] = (df[f'{ticker}_close']-df[f'{ticker}_close'].iloc[0])/df[f'{ticker}_close'].iloc[0]
        return df
        
    def dcf_valuation(self):
        q_bs_df = pd.DataFrame()
        a_cf_df=pd.DataFrame()

        #gather historical data for ticker and indices data
        ticker_hist = self.get_historical_data()
        indices = ['SPY','DIA','QQQ',self.ticker]
        indicies_ticker = yq.Ticker(indices)
        joint_df = indicies_ticker.history(period='10y',interval='1d').reset_index()
        
        joint_df[f'{self.ticker}_pct_change'] = (joint_df[f'{self.ticker}_close']-joint_df[f'{self.ticker}_close'].iloc[0])/joint_df[f'{ticker}_close'].iloc[0]
        joint_df['SPY_pct_change'] = (joint_df['SPY_close']-joint_df['SPY_close'].iloc[0])/joint_df['SPY_close'].iloc[0]
        joint_df['QQQ_pct_change'] = (joint_df['QQQ_close']-joint_df['QQQ_close'].iloc[0])/joint_df['QQQ_close'].iloc[0]
        joint_df['DIA_pct_change'] = (joint_df['DIA_close']-joint_df['DIA_close'].iloc[0])/joint_df['DIA_close'].iloc[0]
        
        #,plot historical figs
        self.make_comp_chart(self.ticker,joint_df)
        
        
        #GET QUARTERLY BALANCE SHEET, CASH FLOW
        balance_sheet= self.yf.balance_sheet('q',False)
        cash_flow_df = self.yf.cash_flow('a',True).reset_index()
        cash_flow_df = cash_flow_df.drop_duplicates(subset='asOfDate')
        cash_flow_df['asOfDate'] = list(map(str,cash_flow_df['asOfDate']))
        cash_flow_df['year'] = cash_flow_df['asOfDate'].apply(lambda x: x.split('-')[0])
        cash_flow_df.insert(0,'Period',cash_flow_df['year']+'-'+cash_flow_df['periodType'])
        
        
        # PLOT HISTORICAL CASH FLOWS
        cf_fig = px.bar(data_frame=cash_flow_df,x='Period',y='FreeCashFlow',orientation='v',title=f'{self.ticker} Historical Free Cash Flows')
        cf_fig.show()
        
        # CREATE VARIABLES TO PRINT AT BEGINNING
        try:
            total_debt = balance_sheet.iloc[-1]['TotalDebt']
        except KeyError:
            total_debt=0
        
        try:
            debt_payment = np.nan_to_num(cash_flow_df.iloc[-1]['RepaymentOfDebt']*-1)
        except KeyError:
            debt_payment = 0
        
            
        try:
            cash_and_ST_investments = balance_sheet.iloc[-1]['CashAndCashEquivalents']
        except KeyError:
            cash_and_ST_investments = balance_sheet.iloc[-1]['CashCashEquivalentsAndShortTermInvestments']
            while pd.isnull(cash_and_ST_investments):
                for i in range(1,len(balance_sheet)):
                    cash_and_ST_investments = balance_sheet.iloc[-i]['CashCashEquivalentsAndShortTermInvestments']
        
        while pd.isnull(cash_and_ST_investments):
            for i in range(1,len(balance_sheet)):
                cash_and_ST_investments = balance_sheet.iloc[-i]['CashAndCashEquivalents']
                
            
        cash_flow = cash_flow_df.iloc[-1]['FreeCashFlow']

        try:
            quick_ratio = balance_sheet.iloc[-1]['CurrentAssets']/balance_sheet.iloc[-1]['CurrentLiabilities']
        except KeyError: # means not available
            quick_ratio = 0
        
        print(f'{self.ticker.upper()} Financial Overview')
        print(f"Free Cash Flow: {cash_flow}")
        print(f"Total Debt:{total_debt} ")
        print(f"Cash and ST Investments: {cash_and_ST_investments}")
        print(f"Quick Ratio: {round(quick_ratio,3)}")
        
        # SET DCF VARIABLES
        total_equity = self.yf.summary_detail[self.ticker]['marketCap']
        try:
            beta = self.yf.summary_detail[self.ticker]['beta']
        except KeyError:
            beta=2.0 # if no beta available, set to 2.0 for higher risk


        # LEFT OFF HERE IN THE DCF CALC, NEED WORKAROUND FOR FINVIZ TO COMPLETE THE REMAINING
        current_price = self.pri
        shares_outstanding = total_equity/current_price
        tax_rate = dcf.get_tax_rate(sticker,key)
        treasury = self.get_10_year()
        wacc = dcf.get_wacc(total_debt,total_equity,debt_payment,tax_rate,beta,treasury,ticker)
        
        # CALL STRATEGISK DCF VALUATION
        intrinsic_value = dcf.calculate_intrinsic_value(self.ticker,
                                                    cash_flow_df, total_debt, 
                                                    cash_and_ST_investments, 
                                                    finviz_df, wacc,shares_outstanding)
        
        
        # CALL FINANCIAL MODEL PREP DCF VALUATION
        fmp_dcf = dcf.get_fmp_dcf(self.ticker,key)
        print(f"Estimated {self.ticker} Valuation Results:\nCalculated Discounted Cash Flow Value: {round(intrinsic_value,2)}")
        print(f"Current Price: {round(current_price,2)}")
        print(f"Margin: {round((1-current_price/intrinsic_value)*100,2)}%")
        
        try:
            print(f"\nFinancial Modeling Prep {ticker} DCF Target Price: {round(fmp_dcf[0]['dcf'],2)}")
        except IndexError:
            print(f"\nNo {ticker} valuation by Financial Modeling Prep")
        
        try:
            print(f"\nFinViz {ticker} Target Price: {float(finviz_df['Target Price'])}\n")
        except ValueError:
            print('No FinViz Valuation Available')
        
        corr_5 = joint_df.corr().at['SPY_close',f'{self.ticker}_close']
        corr_1 = joint_df[-365:].corr().at['SPY_close',f'{self.ticker}_close']
        corr_90 = joint_df[-90:].corr().at['SPY_close',f'{self.ticker}_close']
        corr_30 = joint_df[-30:].corr().at['SPY_close',f'{self.ticker}_close']
        corrs = [corr_5,corr_1,corr_90,corr_30]
        periods = ['5Y','1Y','90Day','30Day']
        zipped = zip(periods,corrs)
        print(f'Correlation to S&P500 Over Time: {[i for i in zipped]}')
        
        yahoo_ratings = self.yf.recommendation_trend.reset_index()
        yahoo_ratings.rename(columns={'period':'Period'},inplace=True)
        yahoo_ratings.at[0,'Period'] = 'Current'
        yahoo_ratings.at[1,'Period'] = '1 Month Back' 
        yahoo_ratings.at[2,'Period'] = '2 Months Back'
        yahoo_ratings.at[3,'Period'] = '3 Months Back' 
        ratings_fig = px.bar(yahoo_ratings,x='Period',y=['strongBuy','buy','hold','sell','strongSell'],
                            title=f'{self.ticker} Yahoo Recommendation Trends')
        ratings_fig.show()

# Testing
if __name__ == '__main__':
    simplifi = Simplifi('ULTA')
    simplifi.blackscholes()
    simplifi.get_historical_data(make_ohlc=True)
    simplifi.ddm_valuation()
    simplifi.get_capm_return(target_return=0.07)
    simplifi.run_regression(dte=10,make_viz=True)
