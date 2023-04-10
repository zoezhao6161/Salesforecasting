import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import datetime as dt

df_tfs=pd.read_csv('C:\\Users\\zoes\\Documents\\Github\\Salesforecasting\\case study walmart salesforecasting\\df_tfs.csv')
print(df_tfs.dtypes)
#turn boolean true or false to  1 or 0
df_tfs['IsHoliday']=df_tfs['IsHoliday'].apply(lambda x:1 if x== True else 0)
print(df_tfs['IsHoliday'].unique())

#turn type  ABC to numeric
print(df_tfs['Type'].unique())


le = preprocessing.LabelEncoder()
le.fit(df_tfs['Type'].unique())
df_tfs['Type']=le.transform(df_tfs['Type'].to_list())
print(df_tfs['Type'].unique())

#temperature vs weeklysales
print(df_tfs['Temperature'].unique())
sns.jointplot(data=df_tfs, x='Temperature',y='Weekly_Sales')
plt.show()
#there is correlation b/w temperature and weekly sales, so we can transfrom numeric to category
t_10_20=df_tfs[(df_tfs['Temperature']>10) & (df_tfs['Temperature']<=20) ].Weekly_Sales.sum()
t_20_30=df_tfs[(df_tfs['Temperature']>20) & (df_tfs['Temperature']<=30) ].Weekly_Sales.sum()
t_30_40=df_tfs[(df_tfs['Temperature']>30) & (df_tfs['Temperature']<=40) ].Weekly_Sales.sum()
t_40_50=df_tfs[(df_tfs['Temperature']>40) & (df_tfs['Temperature']<=50) ].Weekly_Sales.sum()
t_50_60=df_tfs[(df_tfs['Temperature']>50) & (df_tfs['Temperature']<=60) ].Weekly_Sales.sum()
t_60_70=df_tfs[(df_tfs['Temperature']>60) & (df_tfs['Temperature']<=70) ].Weekly_Sales.sum()
t_70_80=df_tfs[(df_tfs['Temperature']>70) & (df_tfs['Temperature']<=80) ].Weekly_Sales.sum()
t_80_90=df_tfs[(df_tfs['Temperature']>80) & (df_tfs['Temperature']<=90) ].Weekly_Sales.sum()
t_90_100=df_tfs[(df_tfs['Temperature']>90) & (df_tfs['Temperature']<=100) ].Weekly_Sales.sum()

t_list=[t_10_20,t_20_30,t_30_40,t_40_50,t_50_60,t_60_70,t_70_80,t_80_90,t_90_100]
t_df=pd.Series(t_list,index=['t_10_20','t_20_30','t_30_40','t_40_50','t_50_60','t_60_70','t_70_80','t_80_90','t_90_100'])
t_df.sort_values(ascending=True).plot()

t_df= t_df.to_frame()
t_df.reset_index(inplace=True)
print(t_df)

#turn temperature to categorical

df_tfs['t_rank']=np.nan
df_tfs.loc[((df_tfs['Temperature']>90) & (df_tfs['Temperature']<=100)),'t_rank']=0
df_tfs.loc[((df_tfs['Temperature']>30) & (df_tfs['Temperature']<=40)),'t_rank']=1
df_tfs.loc[((df_tfs['Temperature']>40) & (df_tfs['Temperature']<=50)),'t_rank']=2
df_tfs.loc[((df_tfs['Temperature']>50) & (df_tfs['Temperature']<=60)),'t_rank']=3
df_tfs.loc[((df_tfs['Temperature']>60) & (df_tfs['Temperature']<=70)),'t_rank']=4
df_tfs.loc[((df_tfs['Temperature']>70) & (df_tfs['Temperature']<=80)),'t_rank']=5
df_tfs.loc[((df_tfs['Temperature']>80) & (df_tfs['Temperature']<=90)),'t_rank']=6

print(df_tfs[['Temperature','t_rank']])

#how to transform date
df_tfs['Date']=pd.to_datetime(df_tfs['Date'])
df_tfs['Year']=df_tfs['Date'].dt.year
df_tfs['Month']=df_tfs['Date'].dt.month
df_tfs['Week']=df_tfs['Date'].dt.week
df_tfs['Day']=df_tfs['Date'].dt.day

df_tfs[['Date','Year','Month','Day']]

#transform IsHoliday to categorical and calculate correlation with weekly sales
df_holiday=df_tfs[df_tfs['IsHoliday']==1][['IsHoliday','Week']]
df_holiday['Week'].unique()
#how to turn array([ 6, 36, 47, 52]) to 0,1,2,3


x=df_tfs[df_tfs['IsHoliday']==1]['Week'].unique()
x=x.tolist()
x.append(0)
print(x)

le = preprocessing.LabelEncoder()
le.fit(x)
df_tfs['Holiday_Type']=df_tfs['Week'].apply(lambda x :0 if x  not in [6, 36, 47, 52] else x)
df_tfs['Holiday_Type']=le.transform(df_tfs['Holiday_Type'].to_list())
print(df_tfs['Holiday_Type'].unique())

# correlation b/w IsHoliday vs Sales & Holiday_Type vs Sales
cor=df_tfs[['Holiday_Type','IsHoliday','Weekly_Sales']].corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()

# they are the same...

# calculate pre holiday, post holiday
def weeks_pre_holiday(x):
    diff_list = []
    if x['Year'] == 2010:
        for d in [dt.datetime(2010, 12, 31), dt.datetime(2010, 11, 26), dt.datetime(2010, 9, 10),
                  dt.datetime(2010, 2, 12)]:
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))

    if x['Year'] == 2011:
        for d in [dt.datetime(2010, 12, 30), dt.datetime(2010, 11, 25), dt.datetime(2010, 9, 9),
                  dt.datetime(2010, 2, 11)]:
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))

    if x['Year'] == 2012:
        for d in [dt.datetime(2010, 12, 28), dt.datetime(2010, 11, 23), dt.datetime(2010, 9, 7),
                  dt.datetime(2010, 2, 10)]:
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))


df_tfs['weeks_pre_holiday'] = df_tfs.apply(weeks_pre_holiday, axis=1)
print(df_tfs)

cor=df_tfs[['weeks_pre_holiday','Weekly_Sales']].corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()