import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../Data/titanic.csv')
df.head()
df.duplicated()
df.info()

cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns :', cat_col)
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns: ', num_col)

df[cat_col].nunique()
df['Ticket'].unique()[:50]

df1 = df.drop(columns = ['Name', 'Ticket'])
df1.shape
df1.head()
round((df1.isnull().sum()/df1.shape[0])*100,2)

df2 = df1.drop(columns = ['Cabin'])
df2.dropna(subset=['Embarked'], axis=0, inplace=True)
df2.shape

df3 = df2.fillna(df2.Age.mean())
df3.isnull().sum()

plt.boxplot(df3['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

mean = df3['Age'].mean()
std = df3['Age'].std()
lower_bound = mean - std*2
upper_bound = mean + std*2
print('Lower Bound: ',lower_bound)
print('Upper Bound: ',upper_bound)

df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]
df4.info()

X = df4[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df4['Survived']

scaler = MinMaxScaler(feature_range=(0,1))
num_col_ = [col for col in df4.columns if df4[col].dtype != 'object']
df5 = df4
df5[num_col] = scaler.fit_transform(df5[num_col])
df5.head()