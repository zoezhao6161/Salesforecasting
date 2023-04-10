import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df_tfs=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\df_tfs.csv',parse_dates=['Date'])

#date  to numeric
df_tfs['Date']=pd.to_numeric(df_tfs['Date'])

#type from abc to 123
le = preprocessing.LabelEncoder()
le.fit(df_tfs['Type'].unique())
df_tfs['Type']=le.transform(df_tfs['Type'].to_list())
print(df_tfs['Type'].unique())

#missing value to 0
df_tfs.loc[df_tfs.MarkDown1.isnull(),'MarkDown1']=0
df_tfs.loc[df_tfs.MarkDown2.isnull(),'MarkDown2']=0
df_tfs.loc[df_tfs.MarkDown3.isnull(),'MarkDown3']=0
df_tfs.loc[df_tfs.MarkDown4.isnull(),'MarkDown4']=0
df_tfs.loc[df_tfs.MarkDown5.isnull(),'MarkDown5']=0

def WMAE(ds,expect,predict):
    W= ds.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.sum(W* abs(predict-expect))/np.sum(W)



y=df_tfs['Weekly_Sales']
X=df_tfs.drop(['Weekly_Sales'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

print(f'MAE for Linear Regression - {mean_absolute_error(y_test,y_pred)}')
print(f'WMAE for Linear Regression - {WMAE(X_test,y_test,y_pred)}')

from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)

print(f'MAE for Random Forest - {mean_absolute_error(y_test,y_pred)}')
print(f'WMAE for Random Forest - {WMAE(X_test,y_test,y_pred)}')

importance = rf.feature_importances_
index_top10 =np.argsort(importance)[::-1][:10]
columns_top_10 = [X_train.columns[i] for i in index_top10]
print(columns_top_10)
plt.bar(range(10),importance[index_top10])
plt.xticks(range(10),columns_top_10)
plt.show()