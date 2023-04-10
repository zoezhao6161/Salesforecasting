import pandas as pd
import matplotlib.pyplot as plt

df_t=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\train.csv')
#print(df_t.head())

df_f=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\features.csv')
#print(df_f.head())

df_s=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\stores.csv')
#print(df_s.head())

df_tf=df_t.merge(df_f,on=['Store','Date','IsHoliday'],how='inner')
df_tfs=df_tf.merge(df_s,on=['Store'],how='inner')
print(df_tfs)
print(df_tfs.dtypes)
print(df_tfs.describe())

df_tfs.to_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\df_tfs.csv')

# find missing values

df_tfs.loc[df_tfs.MarkDown1.isnull(),'MarkDown1']=0
df_tfs.loc[df_tfs.MarkDown2.isnull(),'MarkDown2']=0
df_tfs.loc[df_tfs.MarkDown3.isnull(),'MarkDown3']=0
df_tfs.loc[df_tfs.MarkDown4.isnull(),'MarkDown4']=0
df_tfs.loc[df_tfs.MarkDown5.isnull(),'MarkDown5']=0

print(df_tfs.isnull().sum())

print(df_tfs.groupby('Date')['Weekly_Sales'].sum())
#df_tfs.groupby('Date')['Weekly_Sales'].sum().hist()
#plt.show()

import seaborn as sns
#see the distribution of weekly sales in histogram
#sns.displot(df_tfs.groupby('Date')['Weekly_Sales'].sum())

#see the distribution of markdown so many null values
'''
sns.distplot(df_tfs.groupby('Date')['MarkDown1'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown2'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown3'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown4'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown5'].sum())
'''

#group by weekly_sales by type
'''
sns.distplot(df_tfs[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum())
'''
'''
df_tfs[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum().plot()
df_tfs[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum().plot()
df_tfs[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum().plot()
'''
# holiday does not affect the sales of store A but heavily impacted on store  A AND B

plt.show()