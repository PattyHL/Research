# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:55:51 2023

@author: patri
"""

###PYTHON CHEAT SHEET

    ##GENERAL FUNCTIONALITY
    
    os.getcwd()
    np.random.seed(n) --> This will give you the same sequence of random numbers eachtime you run your code
    
    x += 1 --> x = x + 1
    
    if __name__ == '__main__':
        INDENTED TEST CODE HERE --> only runs test cases when the function is run from the original .py
        
    
    df.iterrows() --> is used to iterate over a pandas Data frame rows in the form of (index, series) pair. This function iterates over the data frame column, it will return a tuple with the column name and content in form of series.

    ##DATA
    
    array.dropna(inplace=True) --> drops all NA in series
    
    series.iloc[-1] += x --> adds "x" to last value of series, eg for bond cashflows it will add the principal to last coupon pmt
    
    np.linspace(start,stop,interval) --> Return evenly spaced numbers over a specified interval.
    
        #list comprehension
        listx = [function(arg1, arg2, argx, etc) for argx in listy]
    
        #append to list
        list.append()
    
    import quandl
    quandl.ApiConfig.api_key = 'XsGZi1kZVoBCCThKf2aY'

    %timeit function() --> test the time to completion for a function

        #Dataframe
        np.empty_like(array_eg) --> creates an empty array with same dimensions as array_eg
        
        df.iloc[-1] --> gives last value of index 
        
        df = df.add_suffix('_some_suffix') --> add suffix to column names
        df = df.add_prefix('some_prefix_') --> add prefix to column names
        #index
        dataframe.index.values --> gives values of index
        
        #rate conversions
        #instantaneous to annualized
        np.expm1(r)
        
        #annualized to instantaneous
        np.log1p(r)
        
    ##DATETIME FUNCTIONS

        #Resampling
            #convert object into datetime object
            df['Column'] = pd.to_datetime(df['Column'], format="%d/%m/%Y")
            
            #if column already exists but not index
            df['Column']=pd.to_datetime(df['Column'])
            
            #set index
            df.set_index('Column',inplace=True)
            
            #convert directly from data import
            df = pd.read_csv('path', index_col='Column',parse_dates=True)
            
            #time resampling
            df.resample(rule='A')
            
            Alias 	Description
                B 	business day frequency
                C 	custom business day frequency (experimental)
                D 	calendar day frequency
                W 	weekly frequency
                M 	month end frequency
                SM 	semi-month end frequency (15th and end of month)
                BM 	business month end frequency
                CBM 	custom business month end frequency
                MS 	month start frequency
                SMS 	semi-month start frequency (1st and 15th)
                BMS 	business month start frequency
                CBMS 	custom business month start frequency
                Q 	quarter end frequency
                BQ 	business quarter endfrequency
                QS 	quarter start frequency
                BQS 	business quarter start frequency
                A 	year end frequency
                BA 	business year end frequency
                AS 	year start frequency
                BAS 	business year start frequency
                BH 	business hour frequency
                H 	hourly frequency
                T, min 	minutely frequency
                S 	secondly frequency
                L, ms 	milliseconds
                U, us 	microseconds
                N 	nanoseconds

        #time shifts
            #shift forwards
            df.shift(periods=x) where x is the number of rows you want the the data to be shifted down
        
            #shift backwards
            df.shift(periods=-x)
            
            #t shift & frequency argument
            df.tshift(freq='M') --> all arguments that share the same month and year will be assigned to the last day of that month
            
            
        #pandas rolling and expanding
            #moving average
            df.rolling(window=7).mean() --> moving average for 7 past periods
            df['Close 30 Day MA']=df['Close'].rolling(window=30).mean() --> add column of 30 day MA to df
            
            #cumulative --> expanding method
            df['Close'].expanding.mean() --> cumulative mean
            
            #bollinger bands
                #close 20 MA
                df['Close: 20 Day MA'] = df['Close'].rolling(20).mean()
                #upper = 20 MA + 2*std(20)
                df['Upper']=df['Close: 20 Day MA']+2*(df['Close'].rolling(20).std())
                #lower = 20 MA + 2*std(20)
                df['Lower']=df['Close: 20 Day MA']-2*(df['Close'].rolling(20).std())            
        
        

    ##TIME SERIES ANALYSIS
        #statsmodels
        df=sm.datasets.macrodata.load_pandas().data
        
        #convert column into time index
        sm.tsa.datetools.dates_from_range('first date','last date')
        
        #Hodrick–Prescott filter --> remove cyclical component of raw data
        sm.tsa.filters.hpfilter(df['realgdp']) --> gives a tuple 
            #use tuple unpacking
            gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp']) 
            df['trend']=gdp_trend
            
        #ETS Decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        result  = seasonal_decompose(df['dataseries'],model='multiplicative')
        result.seasonal.plot() --> plot out seasonal part of data
        result.trend
        result.resid
        result.plot() --> plots observed, trend, seasonal and residual all at once
        
                   
        
            #weakness of SMA 
            # smaller windows lead to more noise
            # always lag by size of window
            # never each peak or trough of data 
            # not informative for futurte behavior 
            # past outliers can skew SMA a lot
            
            #EWMA
            df['EWMA-12'] = df['data'].ewm(span=12).mean() --> 12 month EWMA
            
        #ARIMA
            #initial computations
            times_series = df['Data Column']
            
            #visual representation
            times_series.rolling(12).mean().plot(label='12 Month Rolling Mean')
            times_series.rolling(12).std().plot(label='12 Month Rolling Std')
            times_series.plot()
            
            decomp = season_decompose(time_series)
            fig = decom.plot()
            
            plt.legend()
            
            #stationarity tests
            #augmented dickey fuller test
            #null hypothesis --> non stationary time series 
            
            from statsmodels.tsa.stattools import adfuller --> import augmented dickey fuller test
            
            #augmented Dickey Fuller test
            result=adfuller(df['Column'])
            
            #create function to do DF test on a time series and format the results
            def adf_check(time_series):
                
                result=adfuller(time_series)
                print("Augmented Dickey-Fuller Test")
                labels=['ADF Test Statistic', 'p-value','# of Lags','# of Observations Used']
                
                for value,label in zip(result,labels):
                    print(label+ " : "+str(value))
                    
                if result[1]<=0.05:
                    print("Strong evidence against null hypothesis")
                    print("Reject null hypothesis")
                    print("Data has no unit root and is stationary")
                else:
                    print("Weak evidence against null hypothesis")
                    print("Fail to reject null hypothesis")
                    print("Data has unit root and is non stationary")
                    
                
            #differencing techniques
            df['First difference']=df['Column'].df['Column'].shift(1)
            
            
            #!!! for adf_check need to dropna() for first difference as we loose one row of data 
            adf_check(df['First difference'].dropna())
            
            df['Second difference']=df['First difference']-df['First difference'].shift(1)
            
            #seasonal first difference 
            df['Seasonal first difference']=df['Column'].df['Column'].shift(12) --> if monthly
                
                
            #ACF and PACF
            from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

                #ACF
                fig_acf = plot_acf(df['Column'])
                #PACF
                fig_pacf = plot_pacf(df['Column'])
            
            #implementing ARIMA model
            from statsmodel.tsa.arima_model import ARIMA
            
            model=sm.tsa.statespace.SARIMAX(df['Column'],order=(0,1,0),seasonal_order=(1,1,1,12))
            
            results=model.fit()
            
            print(results.summary())
            
            results.resid --> gives residual value
            
            #make fitted value
            
            df['forecast']=results.predict(start=150, end=168)
            df[['Column','forecast']].plot() --> plots forecast and actual data
            
            #make forecast
                #add on new dates to df: create future dates
                from pandas.tseries.offsets import DateOffset
                
                futures_dates=[df.index[-1]+DateOffset(months-x) for x in range(1,24)] --> add 24 months to the last date of df
                future_df=pd.DataFrame(index=future_dates, columns=df.columns) --> create df of new dates with same amount of columns as original df
                final_df = pd.concat([df,future_df])
                
                #forecast the values for the created 24 months
                final_df['forecast']=results.predict(start=168,end=192)
              
    ##Monte Carlo Simulation
    np.random.normal(mu, std, size)
     
        #Geometric Brownian Motion
        def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
            import numpy as np
            """
            Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
            :param n_years:  The number of years to generate data for
            :param n_paths: The number of scenarios/trajectories
            :param mu: Annualized Drift, e.g. Market Return
            :param sigma: Annualized Volatility
            :param steps_per_year: granularity of the simulation
            :param s_0: initial value
            :return: a numpy array of n_paths columns and n_years*steps_per_year rows
            """
            # Derive per-step Model Parameters from User Specifications
            dt = 1/steps_per_year
            n_steps = int(n_years*steps_per_year) + 1
            # the standard way ...
            # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
            # without discretization error ...
            rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
            rets_plus_1[0] = 1
            ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
            return ret_val
        
        #Cox Ingersoll Ross model --> interest rate random walk 
        def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
            """
            Generate random interest rate evolution over time using the CIR model
            b and r_0 are assumed to be the annualized rates, not the short rate
            and the returned values are the annualized rates as well
            """
            if r_0 is None: r_0 = b 
            r_0 = ann_to_inst(r_0)
            dt = 1/steps_per_year
            num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
            
            shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
            rates = np.empty_like(shock)
            rates[0] = r_0

            ## For Price Generation
            h = np.sqrt(a**2 + 2*sigma**2)
            prices = np.empty_like(shock)
            ####

            def price(ttm, r):
                _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
                _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
                _P = _A*np.exp(-_B*r)
                return _P
            prices[0] = price(n_years, r_0)
            ####
            
            for step in range(1, num_steps):
                r_t = rates[step-1]
                d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
                rates[step] = abs(r_t + d_r_t)
                # generate prices at time t as well ...
                prices[step] = price(n_years-step*dt, rates[step])

            rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
            ### for prices
            prices = pd.DataFrame(data=prices, index=range(num_steps))
            ###
            return rates, prices
 
    
     ##Liability Driven Investing
         #Duration matching
         
         def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
             """
             Returns the series of cash flows generated by a bond,
             indexed by the payment/coupon number
             """
             n_coupons = round(maturity*coupons_per_year)
             coupon_amt = principal*coupon_rate/coupons_per_year
             coupons = np.repeat(coupon_amt, n_coupons)
             coupon_times = np.arange(1, n_coupons+1)
             cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
             cash_flows.iloc[-1] += principal  #adds in the principal for the last CF
             return cash_flows
             


    ##CHARTING
    
        #2 y axes
        fig1, ax1 = plt.subplots(figsize=(12,6))

        resample1=US_Corps_data['USHY_OAS'].loc[sample_start:'2023-01'].resample(rule='Q').mean()
        x2=resample1.index.values
        ax2 = ax1.twinx()
        ax1.plot(x2,y1smpl,'b-', label='US HY OAS')
        ax2.plot(x2,y6, 'g-', label='% of Banks Tightening Standards for C&I Loans')
        ax1.set_ylabel('Bps')
        ax2.set_ylabel('%')
        fig1.legend()
        
        
        
    

    
                
                