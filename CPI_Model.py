# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:26:13 2023

@author: patri
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from fredapi import Fred
import datetime as dt
from sklearn.preprocessing import StandardScaler
from arch.univariate import arch_model
import scipy.stats as stats




fred = Fred(api_key='4af1d8883429260c922d5b7f1b3948cc')
FRED_US_Prices_series = ['US_CPIAUCSL', 'US_CPICORESTICKYOY', 'US_CPICORESTICKMOM', 'US_PCEPILFE', 'US_STICKCPIMOM', 'US_STICKCPIYOY', 'US_JCXFE', 'US_CPIEHOUSE', 'US_CPIEMEDCARE', 'US_CPIETRANS', 'US_CPIEBEV', 'US_CPIEAPPAREL', 'US_CPIEREC', 'US_CPIECOMEDU', 'US_CPIEOTRGS', 'US_PCETRIMYOY']
FRED_US_Prices_source_list = ['CPIAUCSL', 'CORESTICKM159SFRBATL', 'CORESTICKM157SFRBATL', 'PCEPILFE', 'STICKCPIM157SFRBATL', 'STICKCPIM159SFRBATL', 'JCXFE', 'CPIEHOUSE', 'CPIEMEDCARE', 'CPIETRANS', 'CPIEBEV', 'CPIEAPPAREL', 'CPIEREC', 'CPIECOMEDU', 'CPIEOTRGS', 'TRMMEANCPIM159SFRBCLE']
  
FRED_US_Prices_data = list()
for x in FRED_US_Prices_source_list:
    US_Prices_data = fred.get_series(x,index_col=0)
    FRED_US_Prices_data.append(US_Prices_data)
FRED_US_Prices_data=pd.DataFrame(FRED_US_Prices_data).T.set_axis(FRED_US_Prices_series, axis=1)


sample_start=dt.datetime.strptime("2010-01", "%Y-%m")
sample_end=dt.datetime.strptime("2020-01", "%Y-%m")

cpi=FRED_US_Prices_data['US_CPICORESTICKMOM']


plt.plot(cpi['1967-12':])
cpi=cpi['1967-01':]

cpi_ltmean=np.average(cpi)
cpi_ltstdev=np.std(cpi)
cpi_scaled=(cpi-cpi_ltmean)/cpi_ltstdev

plt.plot(cpi_scaled)

#cpi=cpi.loc["2010-01":]


cpi_train=cpi[:sample_end]
cpi_test=cpi[sample_end:]

## acf + pacf

lags = np.arange(1,21,1)

plot_acf(cpi_scaled, lags=lags, auto_ylims=True)
plot_pacf(cpi_scaled, lags=lags, auto_ylims=True)


## AIC + BIC

OUT_ar_ic=np.zeros((20,2))
for order_ar in range(1,21,1):
    ar=ARIMA(cpi_scaled,order=(order_ar,0,0))
    result_ar=ar.fit()
    OUT_ar_ic[order_ar-1,:]=(result_ar.aic/len(cpi_scaled),result_ar.bic/len(cpi_scaled))


## Checking Selected Model

order_ar=13
ar=ARIMA(cpi_scaled,order=(order_ar,0,0))
result_ar=ar.fit()
residuals=result_ar.resid


# ACF
plot_acf(residuals,lags=lags)
plt.show()
# Q-test
OUT_qtest=sm.stats.acorr_ljungbox(residuals, lags=[12], boxpierce=True)
print('p-value Q-test=', OUT_qtest.iloc[:, 3])


##ARCH MODEL

#squared residuals pacf
plot_pacf(residuals**2, lags=lags, auto_ylims=True)
plt.show()
plot_acf(residuals**2, lags=lags, auto_ylims=True)
plt.show()

#model selection: AIC+BIC
OUT_arch_bic=np.zeros((20))
OUT_arch_aic=np.zeros((20))

for order_p in range(1,21,1):
    arch=arch_model(100*residuals, mean='Zero', vol='ARCH', p=order_p, o=0, q=0, rescale=False)
    result_arch=arch.fit()
    OUT_arch_aic[order_p-1]=result_arch.aic/len(cpi_scaled)
    OUT_arch_bic[order_p-1]=result_arch.bic/len(cpi_scaled)

order_p=min(np.where(OUT_arch_bic==min(OUT_arch_bic))[0][0]+1,np.where(OUT_arch_aic==min(OUT_arch_aic))[0][0]+1)
arch=arch_model(residuals*100,p=np.int(order_p),q=0,o=0, vol='ARCH')
result_arch=arch.fit()
print(result_arch.summary())


##RESIDUAL DIAGNOSTIC
#Residuals and standardized residuals
plt.plot(result_arch._resid)
plt.show()
std_arch_resid=result_arch._resid/result_arch._volatility
plt.plot(std_arch_resid)
plt.show()


#Std residuals normality test
print(stats.jarque_bera(std_arch_resid))
print(stats.kurtosis(std_arch_resid))


#Std residuals serial correlation test
OUT_arch_qtest_res=sm.stats.acorr_ljungbox(std_arch_resid, lags=[10], boxpierce=True)
print('p-value Q-test=', OUT_arch_qtest_res.iloc[:, 3])
OUT_arch_qtest_sqres=sm.stats.acorr_ljungbox(std_arch_resid**2, lags=[10], boxpierce=True)
print('p-value Q-test=', OUT_arch_qtest_sqres.iloc[:, 3])

yh=result_arch._resid
plot_pacf(yh,lags=lags, auto_ylims=True)
plt.show()
plot_acf(yh,lags=lags, auto_ylims=True)
plt.show()

plot_pacf(std_arch_resid**2,lags=lags, auto_ylims=True)
plt.show()
plot_acf(std_arch_resid**2,lags=lags, auto_ylims=True)
plt.show()

## Fitted Model vs Actual Values
plt.plot(cpi)
plt.plot(result_ar.fittedvalues)
plt.show()

##Forecast Values
start=dt.datetime.strptime('1967-01', "%Y-%m")
last=cpi.index[-1]
predict=result_ar.predict(start, end=last + np.timedelta64(12,'M'))



## Convert MoM to YoY
cpi_yoy=(1+cpi/100).rolling(window=12).apply(np.prod, raw=True)-1
predict_yoy=(1+predict/100).rolling(window=12).apply(np.prod, raw=True)-1


plt.plot(cpi_yoy)
plt.plot(predict_yoy)