import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\Decomposing\\venv\\data\\seasonality.xlsx',index_col='t')
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt
from statsmodels.tsa.seasonal import seasonal_decompose

'''
simple exponetial smoothing 
SES=SimpleExpSmoothing(df,initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
df['y_pred_2']=SES.fittedvalues
SES=SimpleExpSmoothing(df['y'],initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
df['y_pred_4']=SES.fittedvalues
SES=SimpleExpSmoothing(df['y'],initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
df['y_pred_6']=SES.fittedvalues
print(df)
df.plot()
plt.show()
'''
'''
result=seasonal_decompose(df,model='multiplicative',period=4)
result.plot()
plt.show()
'''
s_list=['add','mul']



for s in s_list:
    WhES=ExponentialSmoothing(df['y'],seasonal_periods=4,trend='add',seasonal=s,
                          use_boxcox=True,initialization_method='estimated').fit()
    df[f'pred_{s}']=WhES.predict(4)
    df[s]=WhES.fittedvalues

print(df)

df[['y','pred_mul']].plot()
plt.show()
