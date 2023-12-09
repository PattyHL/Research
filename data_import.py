# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:55:47 2023

@author: patri
"""

import pandas as pd 
import numpy as np
import quandl as qd



#import data function


#MerrilLynch
    
def ML_import():
    import pandas as pd 
    import numpy as np
    import quandl as qd
    qd.ApiConfig.api_key = 'XsGZi1kZVoBCCThKf2aY'
    
    ML_series = ['IG_TR', 'AAA_TR', 'AA_TR ', 'A_TR', 'BBB_TR', 'BB_TR', 'B_TR', 'CCC_TR', 'IG_YLD', 'AAA_YLD', 'AA_YLD', 'A_YLD', 'HY_YLD', 'BBB_YLD', 'BB_YLD', 'B_YLD', 'CCC_YLD']
    ML_source_list =['ML/TRI', 'ML/AAATRI', 'ML/AATRI', 'ML/ATRI', 'ML/BBBTRI', 'ML/BBTRI', 'ML/BTRI', 'ML/CCCTRI', 'ML/USEY', 'ML/AAAEY', 'ML/AAY', 'ML/AEY', 'ML/USTRI', 'ML/BBBEY', 'ML/BBY', 'ML/BEY', 'ML/CCCY']

    ML_data=qd.get(ML_source_list, index_col=0)
    ML_data=ML_data.set_axis(ML_series,axis=1)
    
    return ML_data

def QUANDL_import():
    import pandas as pd
    import numpy as np
    import quandl as qd
    
    qd.ApiConfig.api_key = 'XsGZi1kZVoBCCThKf2aY'
    
    QUANDL_series = ['DEU_GDPR_RB',	'FRA_GDPR_RB',	'ITA_GDPR_RB',	'ESP_GDPR_RB',	'NLD_GDPR_RB',	'SWE_GDPR_RB',	'NOR_GDPR_RB',	'FIN_GDPR_RB',	'DNK_GDPR_RB',	'EUU_GDPR_RB',	'EMU_GDPR_RB',	'DEU_GDPR_YoY',	'FRA_GDPR_YoY',	'ITA_GDPR_YoY',	'ESP_GDPR_YoY',	'NLD_GDPR_YoY',	'SWE_GDPR_YoY',	'NOR_GDPR_YoY',	'FIN_GDPR_YoY',	'DNK_GDPR_YoY',	'EUU_GDPR_YoY',	'EMU_GDPR_YoY']
    QUANDL_source_list =['OECD/NAAG_DEU_GDPVIXOB',	'OECD/NAAG_FRA_GDPVIXOB',	'OECD/NAAG_ITA_GDPVIXOB',	'OECD/NAAG_ESP_GDPVIXOB',	'OECD/NAAG_NLD_GDPVIXOB',	'OECD/NAAG_SWE_GDPVIXOB',	'OECD/NAAG_NOR_GDPVIXOB',	'OECD/NAAG_FIN_GDPVIXOB',	'OECD/NAAG_DNK_GDPVIXOB',	'OECD/NAAG_EUU_GDPVIXOB',	'OECD/NAAG_EMU_GDPVIXOB',	'OECD/NAAG_DEU_GDPG',	'OECD/NAAG_FRA_GDPG',	'OECD/NAAG_ITA_GDPG',	'OECD/NAAG_ESP_GDPG',	'OECD/NAAG_NLD_GDPG',	'OECD/NAAG_SWE_GDPG',	'OECD/NAAG_NOR_GDPG',	'OECD/NAAG_FIN_GDPG',	'OECD/NAAG_DNK_GDPG',	'OECD/NAAG_EUU_GDPG',	'OECD/NAAG_EMU_GDPG']

    QUANDL_data=qd.get(QUANDL_source_list, index_col=0)
    QUANDL_data=QUANDL_data.set_axis(QUANDL_series,axis=1)
    
    return QUANDL_data

    

#FRED

def FRED_import():
    import pandas as pd 
    import numpy as np
    from fredapi import Fred
    fred = Fred(api_key='4af1d8883429260c922d5b7f1b3948cc')
    
    FRED_Corp_series = ['USHY_OAS', 'USBBB_OAS', 'USIG_OAS', 'USCCC_OAS', 'USBB_OAS', 'EUHY_OAS', 'USB_OAS', 'USA_OAS', 'EMIG_OAS', 'USAAA_OAS', 'USAA_OAS', 'ASIAIG_OAS', 'US1Y3YIG_OAS', 'US3Y5YIG_OAS', 'US7Y10YIG_OAS', 'EUEMIG_OAS', 'US5Y7YIG_OAS', 'US15YIG_OAS', 'US10Y15YIG_OAS']
    FRED_Corp_source_list = ['BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLC0A0CM', 'BAMLH0A3HYC', 'BAMLH0A1HYBB', 'BAMLHE00EHYIOAS', 'BAMLH0A2HYB', 'BAMLC0A3CA', 'BAMLEMCBPIOAS', 'BAMLC0A1CAAA', 'BAMLC0A2CAA', 'BAMLEMRACRPIASIAOAS', 'BAMLC1A0C13Y', 'BAMLEMHYHYLCRPIUSOAS', 'BAMLC2A0C35Y', 'BAMLEM3BRRBBCRPIOAS', 'BAMLC3A0C57Y', 'BAMLC8A0C15PY', 'BAMLC7A0C1015Y']
    FRED_US_Economy_series = ['US_GDPR', 'US_CPI', 'USDEUR', 'US_10YT', 'TWUSD', 'US_EFFR', 'US_UI4WMA', 'US_IP', 'US_PAYRLS', 'SP500', 'US_FSI', 'US_10Y2YYC', 'US_3MT', 'US_UR', 'US_Wlsh5K', 'US_CapUt']
    FRED_US_Economy_source_list = ['A191RL1Q225SBEA', 'CPIAUCSL', 'DEXUSEU', 'DGS10', 'DTWEXM', 'FEDFUNDS', 'IC4WSA', 'INDPRO', 'PAYEMS', 'SP500', 'STLFSI', 'T10Y2Y', 'TB3MS', 'UNRATE', 'WILL5000INDFC', 'TCU']
    FRED_US_Prices_series = ['US_CPIAUCSL', 'US_CPICORESTICKYOY', 'US_CPICORESTICKMOM', 'US_PCEPILFE', 'US_STICKCPIMOM', 'US_STICKCPIYOY', 'US_JCXFE', 'US_CPIEHOUSE', 'US_CPIEMEDCARE', 'US_CPIETRANS', 'US_CPIEBEV', 'US_CPIEAPPAREL', 'US_CPIEREC', 'US_CPIECOMEDU', 'US_CPIEOTRGS', 'US_PCETRIMYOY']
    FRED_US_Prices_source_list = ['CPIAUCSL', 'CORESTICKM159SFRBATL', 'CORESTICKM157SFRBATL', 'PCEPILFE', 'STICKCPIM157SFRBATL', 'STICKCPIM159SFRBATL', 'JCXFE', 'CPIEHOUSE', 'CPIEMEDCARE', 'CPIETRANS', 'CPIEBEV', 'CPIEAPPAREL', 'CPIEREC', 'CPIECOMEDU', 'CPIEOTRGS', 'TRMMEANCPIM159SFRBCLE']
    FRED_US_NationalAccts_series = ['US_GDP', 'US_GDPA', 'US_WEI', 'US_CorpNI', 'US_SR', 'US_PDI', 'US_PDIR', 'US_FedAs']
    FRED_US_NationalAccts_source_list = ['GDP', 'GDPA', 'WEI', 'A466RC1A027NBEA', 'PSAVERT', 'NC000335Q', 'NB000335Q', 'WALCL']
    FRED_US_BankReporting_series = ['US_BTS_CI',	'US_BTS_CC',	'US_BTS_Auto',	'US_BTS_SubMtge',	'US_BTS_CRE',	'US_BIW_ConsInst',	'US_BSD_Auto',	'US_BTS_Cons',	'US_BSD_SupMtge',	'US_BSD_CC',	'US_BSD_Cons',	'US_BIS_CC',	'US_MRD_Auto']
    FRED_US_BankReporting_source_list = ['DRTSCILM',	'DRTSCLCC',	'STDSAUTO',	'DRTSSP',	'SUBLPDRCSC',	'DRIWCIL',	'DEMAUTO',	'STDSOTHCONS',	'DRSDSP',	'DEMCC',	'DEMOTHCONS',	'SUBLPDCLCTSNQ',	'SUBLPDCLATDLGNQ']
    
    
    FRED_Corp_data = list()
    for x in FRED_Corp_source_list:
      Corp_data = fred.get_series(x,index_col=0)
      FRED_Corp_data.append(Corp_data)
    FRED_Corp_data=pd.DataFrame(FRED_Corp_data).T.set_axis(FRED_Corp_series, axis=1)
    
    FRED_US_Economy_data = list()
    for x in FRED_US_Economy_source_list:
      US_Economy_data = fred.get_series(x,index_col=0)
      FRED_US_Economy_data.append(US_Economy_data)
    FRED_US_Economy_data=pd.DataFrame(FRED_US_Economy_data).T.set_axis(FRED_US_Economy_series, axis=1)

    FRED_US_Prices_data = list()
    for x in FRED_US_Prices_source_list:
      US_Prices_data = fred.get_series(x,index_col=0)
      FRED_US_Prices_data.append(US_Prices_data)
    FRED_US_Prices_data=pd.DataFrame(FRED_US_Prices_data).T.set_axis(FRED_US_Prices_series, axis=1)

    FRED_US_NationalAccts_data = list()
    for x in FRED_US_NationalAccts_source_list:
      US_NationalAccts_data = fred.get_series(x,index_col=0)
      FRED_US_NationalAccts_data.append(US_NationalAccts_data)
    FRED_US_NationalAccts_data=pd.DataFrame(FRED_US_NationalAccts_data).T.set_axis(FRED_US_NationalAccts_series, axis=1)
    
    FRED_US_BankReporting_data = list()
    for x in FRED_US_BankReporting_source_list:
      US_BankReporting_data = fred.get_series(x,index_col=0)
      FRED_US_BankReporting_data.append(US_BankReporting_data)
    FRED_US_BankReporting_data=pd.DataFrame(FRED_US_BankReporting_data).T.set_axis(FRED_US_BankReporting_series, axis=1)
    
    return FRED_Corp_data, FRED_US_Economy_data, FRED_US_Prices_data, FRED_US_NationalAccts_data




    
        

