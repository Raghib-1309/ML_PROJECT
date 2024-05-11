import pandas as pd
import seaborn as sns
df = sns.load_dataset('mpg')
from sklearn.model_selection import train_test_split
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
X=df[['displacement' , 'horsepower' , 'weight' ,'acceleration']]
X.dropna()
Y=df.mpg
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.15, random_state=42)
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train,y_train)
model.score(X_train, y_train)
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(random_state=0)
model2.fit(X_train,y_train)
model2.score(X_test, y_test)
import pickle
filename= 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))

