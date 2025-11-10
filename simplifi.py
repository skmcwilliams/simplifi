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
from millify import millify
pio.renderers.default = 'browser'

class Simplifi:
    def __init__(self,ticker:str):
        self.ticker = ticker
        self.yf = yq.Ticker(self.ticker)
        self.model = PredictiveModel()
        self.rfr = self.get_10_year()
        self.historical_df = self.get_historical_data()
        self.price = self.historical_df['close'][self.historical_df.date==self.historical_df.date.max()].values[0]

    def get_historical_data(self,make_ohlc:bool=False):
        hist = self.yf.history(period='10y',interval='1d',adj_ohlc=False).reset_index()
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
        
        df = self.historical_df.copy() # get daily ticker data
        df = df.sort_values(by='date',ascending=True).reset_index()
        df = df.tail(30) # lasty 30 trading days for stdev calc
        options = simplifi.yf.option_chain.reset_index()
        options['dte'] = (pd.to_datetime(options['expiration']) - pd.to_datetime("now")).dt.days
        options['t'] = options['dte']/365
        # get option data for analysis
        self.sd = round(df['log_returns'].std() * sqrt(252),2) # annualize stdev of log daily returns based on # of annual trading days
        r = self.rfr # risk-free rate based on 10-year treasury
        options['d1'] = (np.log(options['lastPrice']/options['strike']) + (r + self.sd**2/2)* options['t'])/(self.sd*np.sqrt(options['t']))
        options['d2'] = options['d1'] - self.sd * np.sqrt(options['t'])
        options['black_scholes_value'] = np.where(options['optionType']=="calls",options['lastPrice']* norm.cdf(options['d1'], 0, 1) - \
                                                    options['strike'] * np.exp(-r * options['t']) * norm.cdf(options['d2'], 0, 1),np.where(
                                                        options['optionType']=='puts',options['strike'] * np.exp(-r * options['t']) * \
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

        ohlc_fig.add_trace(go.Scatter(x=df['date'], 
                                  y=df['close'],
                                  name='Price',
                                  marker_color='pink'),
                        secondary_y=True)
        
        if 'adjclose' in df.columns:
            ohlc_fig.add_trace(go.Scatter(x=df['date'], 
                                    y=df['adjclose'],
                                    name='Adjusted Close (Dividends/Splits Included)',
                                    marker_color='lime'),
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
        print("--------------------------------------------------")
        try:
            nxt_yr_div = self.yf.summary_detail[self.ticker]["dividendRate"] * (1+growth)
        except KeyError:
            print("No dividend data available for DDM valuation.")
            print("--------------------------------------------------")
            return None
        else:
            price = self.yf.summary_detail[self.ticker]['previousClose']
            coe = (nxt_yr_div / price) + self.yf.summary_detail[self.ticker]["dividendRate"]
            ddm_val = nxt_yr_div/(coe-growth)
            print(f"Simplifi DDM Valuation: ${round(ddm_val*100,2)}")
            print("--------------------------------------------------")
            return ddm_val
 
                
    def get_capm_return(self,target_return:float=None)->float:
        beta = self.yf.summary_detail[self.ticker]['beta']
        if type(beta) is str: # string-based tickers are typically new and associated risk is high
            beta=1.75
        if target_return:
            rm = target_return
        else: 
            rm = 0.085 # static expected market return of 8.5%
        rfr = self.rfr
        capm = rfr+beta*(rm-rfr) # CAPM formula
        print("--------------------------------------------------")
        print("Valuation based on the following:")
        print(f"Risk Free Rate (10yr Treasury): {round(rfr*100,2)}%")
        print(f"Expected Market Return: {round(rm*100,2)}%")
        print(f"Simplifi CAPM Return: {round(capm*100,2)}%")
        print("--------------------------------------------------")
        return capm
    
    def run_regression(self,dte:int, make_viz:bool=True)->pd.DataFrame:
        dte_yhats, fig = self.model.run_regression(self.ticker,dte,self.historical_df,viz=make_viz)
        return dte_yhats
    
    def make_comp_chart(self)->go.Figure:
        comp_fig = go.Figure()
        #gather historical data for ticker and indices data
        indices = ['SPY','DIA','QQQ',self.ticker]
        indices_ticker = yq.Ticker(indices)
        joint_df = indices_ticker.history('10y','1d').reset_index()
        joint_df = joint_df.sort_values(by=['symbol','date'],ascending=True).reset_index(drop=True)
        joint_df = joint_df[['symbol','date','close']]
        # compute percent change per symbol relative to that symbol's first close value
        joint_df['pct_change'] = joint_df.groupby('symbol')['close'].transform(lambda x: (x - x.iloc[0]) / x.iloc[0])
        df = pd.DataFrame(joint_df.groupby(['symbol','date'],as_index=False)['pct_change'].mean())# reset to pull symbol out of index
        comp_fig_scatter = px.line(df,'date',y='pct_change',color='symbol')
        for trace in comp_fig_scatter['data']:
            comp_fig.add_trace(trace)
        # comp_fig.add_trace(go.Scatter(x=df['date'],y=df['SPY_pct_change'],name='SPY'))
        # comp_fig.add_trace(go.Scatter(x=df['date'],y=df['DIA_pct_change'],name='DIA'))
        # comp_fig.add_trace(go.Scatter(x=df['date'],y=df['QQQ_pct_change'],name='QQQ'))
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
                        dict(count=3,
                            label="3y",
                            step="year",
                            stepmode="backward"),
                        dict(count=5,
                            label="5y",
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

    def pct_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percent change for `ticker` relative to the first value for each symbol.

        If `df` contains a column named "symbol" and per-symbol close columns (e.g. '<ticker>_close'),
        this computes the percent difference relative to the first close for each symbol. Otherwise,
        it computes percent difference relative to the first row of the specified ticker close column.
        Results are written to a new column named '<ticker>_pct_change'.
        """
        col = "_close"
        out_col = "_pct_change"

        if col not in df.columns:
            df[out_col] = pd.NA
            return df

        if 'symbol' in df.columns:
            # compute first value per symbol and use it as the base for percent change
            first_vals = df.groupby('symbol')[col].transform('first')
            df[f'{symbol}_out_col'] = (df[col] - first_vals) / first_vals
        else:
            first_val = df[col].iloc[0] if not df[col].empty else pd.NA
            df[f'{symbol}_out_col'] = (df[col] - first_val) / first_val

        return df
        
    def dcf_valuation(self):
        q_bs_df = pd.DataFrame()
        a_cf_df=pd.DataFrame()

        #GET QUARTERLY BALANCE SHEET, CASH FLOW
        balance_sheet= self.yf.balance_sheet('q')
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
        print(f"Free Cash Flow: ${millify(cash_flow)}")
        print(f"Total Debt: ${millify(total_debt)} ")
        print(f"Cash and ST Investments: ${millify(cash_and_ST_investments)}")
        print(f"Quick Ratio: {round(quick_ratio,3)}")
        
        # SET DCF VARIABLES
        total_equity = self.yf.summary_detail[self.ticker]['marketCap']
        try:
            beta = self.yf.summary_detail[self.ticker]['beta']
        except KeyError:
            beta=1.75 # if no beta available, set to 1.75 for higher risk


        # LEFT OFF HERE IN THE DCF CALC, NEED WORKAROUND FOR FINVIZ TO COMPLETE THE REMAINING
        current_price = self.price
        shares_outstanding = total_equity/current_price
        income_stmnt = self.yf.income_statement()
        income_stmnt = income_stmnt[income_stmnt['asOfDate']==income_stmnt['asOfDate'].max()]
        tax_rate = income_stmnt['TaxRateForCalcs'].iloc[0].astype(float)
        treasury = self.rfr
        wacc = self.__get_wacc(total_debt,total_equity,debt_payment,tax_rate,beta,treasury,self.ticker)
        
        # CALL DCF VALUATION
        intrinsic_value = self.__intrinsic_value(cash_flow_df, total_debt, 
                                                    cash_and_ST_investments, 
                                                    wacc,shares_outstanding)
        
        print(f"Simplifi DCF Valuation: ${round(intrinsic_value,2)}")
        print(f"Based on WACC of {round(wacc*100,2)}%")
        return intrinsic_value
    
    def make_yahoo_ratings_fig(self)->go.Figure:
        yahoo_ratings = self.yf.recommendation_trend.reset_index()
        yahoo_ratings.rename(columns={'period':'Period'},inplace=True)
        yahoo_ratings.at[3,'Period'] = 'Current'
        yahoo_ratings.at[2,'Period'] = '1 Month Back' 
        yahoo_ratings.at[1,'Period'] = '2 Months Back'
        yahoo_ratings.at[0,'Period'] = '3 Months Back' 
        ratings_fig = px.bar(yahoo_ratings,x='Period',y=['strongBuy','buy','hold','sell','strongSell'],
                            title=f'{self.ticker} Yahoo Recommendation Trends')
        ratings_fig.show()

    def make_stdev_fig(self)->go.Figure:
        df = self.historical_df.copy()
        df['30_std'] = (df['close'].rolling(30).std()/df['close'])*100 #std over 30 days divided by current price
        df['7_std'] = (df['close'].rolling(7).std()/df['close'])*100 #std over 7 days divided by current price
        #df['60_std'] = (df['close'].rolling(60).std()/df['close'].rolling(60).mean())*100
        stdev_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        stdev_fig.add_trace(go.Bar(x=df['date'],
                                y=df['30_std'],
                                name='30-Day StDev',
                                marker=dict(color='magenta',opacity=0.3)),
                                secondary_y=True)
        
        stdev_fig.add_trace(go.Bar(x=df['date'],
                                y=df['7_std'],
                                name='7-Day StDev',
                                marker=dict(color='navy',opacity=0.3)),
                                secondary_y=True)
        
        stdev_fig.add_trace(go.Scatter(x=df['date'],
                                    y=df['close'],
                                    name='Price',
                                    line_color='cyan'),
                                    secondary_y=False)
        

        
        stdev_fig.layout.yaxis2.showgrid=False
        stdev_fig.update_xaxes(type='category',nticks=10,tickangle=15)
        stdev_fig.update_layout(title_text=f'{self.ticker} 5-Year Volatility-Price Chart',
                            xaxis=dict(rangeslider=dict(visible=False)),barmode='group')
        stdev_fig.update_yaxes(title_text="StDev as % of Price", secondary_y=True)
        stdev_fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        
        return stdev_fig.show()
    
    def __get_wacc(self,total_debt,equity,debt_pmt,tax_rate,beta,rfr,required_return)->float:
        
        if type(beta) is str:
            beta=1.75
        
        beta = float(beta)
        rfr = float(rfr)
        rm = float(required_return)
        re = rfr+beta*(rm-rfr)
        
        if total_debt<1 or debt_pmt<1:
            wacc = re
        else:
            rd= debt_pmt/total_debt
            value = total_debt+equity
            wacc = (equity/value*re) + ((total_debt/value * rd) * (1 - tax_rate))
    
        return wacc
    

    def __intrinsic_value(self,cash_flow_df, total_debt, cash_and_ST_investments, 
                                    discount_rate,shares_outstanding,name)->tuple[go.Figure,float,float,float,float]:
        eps_data = self.yf.key_stats[self.ticker]

        def readable_nums(num_list):
            for num in num_list:
                yield millify(num,precision=2)

        def calc_cashflow():

            cf = cash_flow
            for year in range(1,6):
                cf *= (1 + st_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            for year in range(6,11):
                cf *= (1 + lt_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            for year in range(11,21):
                cf *= (1 + terminal_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            
        try:
           st_growth =  float(eps_data['forwardEps']) / 100
        
        except ValueError: # means EPS next 5Y is string, so cannot be divided, onto substitute method
            st_growth = (float(eps_data["trailingEps"])/ 100) * .75 # set to 75% of last 5y growth instead

        lt_growth = min(st_growth*0.75,0.15)
        
        if lt_growth >=0.15: # Tier terminal growth based on long-term growth
            terminal_growth = min(0.075,.75*lt_growth)

        else:
            terminal_growth = min(0.05, 0.5*lt_growth)
    
        cash_flow=cash_flow_df.iloc[-1]['FreeCashFlow']
        
        year_list = [i for i in range(1,21)]
        cashflows = list(calc_cashflow())
        cf_list = [i[0] for i in cashflows]
        dcf_list = [i[1] for i in cashflows]
         
        intrinsic_value = (sum(dcf_list) - total_debt + cash_and_ST_investments)/shares_outstanding
        df = pd.DataFrame.from_dict({'Year Out': year_list, 'Future Value': cf_list, 'Present Value': dcf_list})
        
        fig = px.bar(df,x='Year Out',y=['Future Value','Present Value'],barmode='group',color_discrete_sequence=['navy','paleturquoise'])
        fig.update_layout(title=f'{name} Projected Free Cash Flows',yaxis_title='Free Cash Flow ($)',legend_title='')
        y1 = list(readable_nums(df['Future Value']))
        y2 = list(readable_nums(df['Present Value']))
        texts = [y1,y2]
        for i, t in enumerate(texts):
            fig.data[i].text = t
            fig.data[i].textposition = 'outside'
    
        return fig, intrinsic_value, st_growth, lt_growth,terminal_growth
    

# Testing
if __name__ == '__main__':
    simplifi = Simplifi('ULTA')
    # simplifi.blackscholes()
    # simplifi.get_historical_data(make_ohlc=True)
    # simplifi.ddm_valuation()
    # simplifi.get_capm_return(target_return=0.07)
    # simplifi.run_regression(dte=10,make_viz=True)
    # simplifi.make_stdev_fig()
    # simplifi.make_yahoo_ratings_fig()
    simplifi.dcf_valuation()
    simplifi.make_comp_chart()
