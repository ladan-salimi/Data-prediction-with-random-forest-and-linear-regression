####prediction using random forest and linear regression
import pandas as pd
from sklearn.metrics import r2_score
from pandas import DataFrame
from numpy import log
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

df=pd.DataFrame()
#Import as Dataframe
df = pd.read_excel(r'C:\Users\ladan\Desktop\clustring\time series pattern recognition.xlsx',sheet_name=2, parse_dates=['date'], index_col='date')
df.index.freq='MS'
#######################log transformation
dataframe = DataFrame(df.values)
dataframe.columns = ['value']
dataframe['value'] = log(dataframe['value'])
#######################preprocessing to make a suitable form of database
df.columns=['value']
df.plot()
df['value_LastMonth']=df['value'].shift(+1)
df['value_2MonthBack']=df['value'].shift(+2)
df['value_3MonthBack']=df['value'].shift(+3)
df['value_4MonthBack']=df['value'].shift(+4)
df=df.dropna()
#########################prediction using linear and randon forest regression
lin_model=LinearRegression()
model=RandomForestRegressor(n_estimators=150,max_features=4, random_state=1)
x1,x2,x3,x4,y=df['value_LastMonth'],df['value_2MonthBack'],df['value_3MonthBack'],df['value_4MonthBack'],df['value']
x1,x2,x3,x4,y=np.array(x1),np.array(x2),np.array(x3),np.array(x4),np.array(y)
x1,x2,x3,x4,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),x4.reshape(-1,1),y.reshape(-1,1)#reshape(-1, 1) 
final_x=np.concatenate((x1,x2,x3,x4),axis=1)
print(final_x)
####train and test
X_train,X_test,y_train,y_test=final_x[:-80],final_x[-80:],y[:-80],y[-80:]
model.fit(X_train,y_train)
lin_model.fit(X_train,y_train)
####################Random_Forest_Predictions
pred=model.predict(X_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
plt.plot(pred,label='Random_Forest_Predictions')
plt.plot(y_test,label='Actual vales')
plt.legend(loc="upper left")
plt.show()
########################Linear_Regression_Predictions
lin_pred=lin_model.predict(X_test)
plt.plot(lin_pred,label='Linear_Regression_Predictions')
plt.plot(y_test,label='Actual values')
plt.legend(loc="upper left")
plt.show()
#########RMSE
rmse_rf=sqrt(mean_squared_error(pred,y_test))
rmse_lr=sqrt(mean_squared_error(lin_pred,y_test))
print('Mean Squared Error for Random Forest Model is:',rmse_rf)
print('Mean Squared Error for Linear Regression Model is:',rmse_lr)
##################R2 score error
r1 = r2_score(pred,y_test)
r2 = r2_score(lin_pred,y_test)
print('r2 score for Random Forest Model is', r1)
print('r2 score for Random Forest Model is', r2)

