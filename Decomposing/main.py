import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt

df=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\Decomposing\\venv\\data\\orders.txt',delimiter='\t',encoding='latin',parse_dates=['orderdate'])

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())

df_sales=df.groupby('orderdate').sum()['totalprice'].to_frame().sort_index()
print(df_sales)
#df_sales.plot()
#plt.show()
#result.plot()

#result.trend.plot()
#plt.show()
#print(result.trend.to_frame())
# what if there's a lot of null value in the trend and residual?
# only include those period without null values
#print(result.trend.isnull().sum())
#print(result.resid)
#print(result.observed)
result=seasonal_decompose(df_sales,model='multiplicative',period=30)
print(result.trend.loc['2011-01-01':'2016-01-01'].var())
print(result.seasonal.loc['2011-01-01':'2016-01-01'].var())
print(result.resid.loc['2011-01-01':'2016-01-01'].var())

WhES=ExponentialSmoothing(df_sales['totalprice'],seasonal_periods=30,
                          trend='add',
                          seasonal='add',
                          use_boxcox=True,
                          damped_trend=True,
                          initialization_method='estimated').fit()
#df_sales['train']=WhES.fittedvalues
f_1=WhES.forecast(240)
print(f_1)
#print(df_sales)
f_1.plot()
plt.show()

