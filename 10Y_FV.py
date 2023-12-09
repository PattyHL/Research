# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:54:18 2023

@author: patri
"""

##Euro Area Macro Monitors

import pandas as pd
import numpy as np
import data_import as data
import datetime as dt
import Fin_Function as fin
from fredapi import Fred
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests

## 10Y Treasury FV Model 

# Causality Tests

fred = Fred(api_key='4af1d8883429260c922d5b7f1b3948cc')
FRED_US_Economy_series = ['US_GDPR', 'US_CPI', 'USDEUR', 'US_10YT', 'TWUSD', 'TWUSD_N', 'US_EFFR', 'US_UI4WMA', 'US_IP', 'US_PAYRLS', 'SP500', 'US_FSI', 'US_10Y2YYC', 'US_3MT', 'US_UR', 'US_Wlsh5K', 'US_CapUt', 'US_5Y5Y', 'US_UMICHINF', 'WTI']
FRED_US_Economy_source_list = ['A191RL1Q225SBEA', 'CPIAUCSL', 'DEXUSEU', 'DGS10', 'DTWEXM', 'DTWEXBGS' ,'FEDFUNDS', 'IC4WSA', 'INDPRO', 'PAYEMS', 'SP500', 'STLFSI', 'T10Y2Y', 'TB3MS', 'UNRATE', 'WILL5000INDFC', 'TCU', 'T5YIFR', 'MICH', 'DCOILWTICO']

FRED_US_Economy_data = list()
for x in FRED_US_Economy_source_list:
    US_Economy_data = fred.get_series(x,index_col=0)
    FRED_US_Economy_data.append(US_Economy_data)

sample_start=dt.datetime.strptime("1986-01", "%Y-%m")
sample_end=dt.datetime.strptime("2023-07", "%Y-%m")

FRED_US_Economy_data=pd.DataFrame(FRED_US_Economy_data).T.set_axis(FRED_US_Economy_series, axis=1)
FRED_US_Economy_data=FRED_US_Economy_data.resample(rule='M').mean()
FRED_US_Economy_data=FRED_US_Economy_data.loc[sample_start:sample_end]

date=dt.datetime.strptime("2006-01", "%Y-%m")
base_value=FRED_US_Economy_data['TWUSD'].loc['2006-01'].iloc[0]

FRED_US_Economy_data['TWUSD_rb']=(FRED_US_Economy_data['TWUSD']/base_value)*100
FRED_US_Economy_data['TWUSD_rb']=FRED_US_Economy_data['TWUSD_rb'].loc[:'2006-01']
FRED_US_Economy_data['TWUSD_rb']=FRED_US_Economy_data['TWUSD_rb'].combine_first(FRED_US_Economy_data['TWUSD_N'])

X = FRED_US_Economy_data[['TWUSD_rb', 'US_UR', 'US_EFFR', 'US_IP', 'US_PAYRLS', 'WTI']]
y = FRED_US_Economy_data['US_10YT']
scale=FRED_US_Economy_data[['US_10YT','TWUSD_rb', 'US_UR', 'US_EFFR', 'US_IP', 'US_PAYRLS', 'WTI']]
#X = sm.add_constant(X)

X[['TWUSD_rb','US_IP', 'US_PAYRLS', 'WTI']]=X[['TWUSD_rb','US_IP', 'US_PAYRLS', 'WTI']].pct_change()
X[['US_UR', 'US_EFFR',]]=X[['US_UR', 'US_EFFR']]-X[['US_UR', 'US_EFFR']].shift(1)

min_max_scaler=preprocessing.MinMaxScaler()
scaler=preprocessing.StandardScaler()

scaled = scaler.fit_transform(scale)

X_scaled = pd.DataFrame(scaled, columns=scale.columns, index=scale.index)


plt.plot(X_scaled)

maxlag=6
variables=['US_10YT','TWUSD_rb', 'US_UR', 'US_EFFR', 'US_IP', 'US_PAYRLS', 'WTI']

X_GC=fin.grangers_causation_matrix(X_scaled, variables)

X_ur=grangercausalitytests(X_scaled[['US_10YT','US_IP']], maxlag=12)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
US_10Y_FVModel = sm.OLS(y,sm.add_constant(X).fit()

# summary=US_10Y_FVModel.summary()
# print(US_10Y_FVModel.summary())
# US_10Y_FV=US_10Y_FVModel.fittedvalues.values
# US_10Y_FV=pd.DataFrame(US_10Y_FV)

# US_10Y_FV.index=y.index

# x1=y
# x2=US_10Y_FV

# plt.plot(x1, label="10-Year")
# plt.plot(x2, label="10-Year FV")
# plt.legend()
# plt.show()

# x3=FRED_US_Economy_data['TWUSD_rb']
# x4=FRED_US_Economy_data['US_UR']
# x5=FRED_US_Economy_data['US_EFFR']

# fig, ax1 = plt.subplots(figsize=(12,6))
# ax2 = ax1.twinx()
# ax1.plot(x3,'b-', label='TWD USD')
# ax2.plot(x4,label='UR')
# ax2.plot(x5,label='Eff FFR')
# ax1.set_ylabel('Index')
# ax2.set_ylabel('%YoY')
# fig.legend()


