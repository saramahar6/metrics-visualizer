import pandas as pd
import numpy as np

from numba import jit, vectorize, float64, int64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from datetime import datetime
from sklearn.externals import joblib

# Data Loading
data = pd.read_csv('./data/TrainAndValid.csv')
test = pd.read_csv('./data/Test.csv')

# Date Features
tmp_date = data.saledate.apply(lambda x : datetime.strptime(x[:-5], '%m/%d/%Y'))
data['sale_mon'] = tmp_date.dt.month
data['sale_dayofweek'] = tmp_date.dt.dayofweek
data['sale_dayofyear'] = tmp_date.dt.dayofyear
data['sale_year'] = tmp_date.dt.year
data.drop(['saledate'],axis=1,inplace=True)

# Taking Subset of Columns
kept_columns = [
                'YearMade', 
                'sale_mon', 
                'sale_dayofweek',
                'sale_dayofyear',
                'sale_year',
                'fiModelDesc',
                'fiBaseModel',
                'fiProductClassDesc',
                'state',
                'SalePrice'
               ]
data = data[kept_columns]
data['age'] = data.sale_year - data.YearMade

# Encoding Class Description
data.loc[:,'classDesc_1'] = data.fiProductClassDesc.apply(lambda x : x.replace(',','').strip().split('-')[0])
data.loc[:,'classDesc_2'] = data.fiProductClassDesc.apply(lambda x : x.replace(',','').strip().split('-')[1])
data.drop('fiProductClassDesc',axis=1,inplace=True)

for col in ['fiModelDesc','fiBaseModel','state','classDesc_1','classDesc_2']:
    lb = LabelEncoder()
    data.loc[:,col] = lb.fit_transform(data.loc[:,col])

# Building a Random Forest Regressor
data.loc[data.YearMade < 1920, 'YearMade'] = np.median(data.YearMade)
data.loc[data.YearMade > 2012, 'YearMade'] = 2012

train, valid = train_test_split(data, test_size = .2)
y_train = train.SalePrice
y_valid = valid.SalePrice
train.drop('SalePrice',axis=1,inplace=True)
valid.drop('SalePrice',axis=1,inplace=True)

rf = RandomForestRegressor(min_samples_split=15, 
                           n_estimators = 30, 
                           n_jobs=-1)
rf.fit(train, y_train)
y_pred = rf.predict(valid)

# Metrics
print(mean_squared_log_error(y_valid, y_pred))
print('Train Score: %s' % rf.score(train, y_train))
print('Valid Score: %s' % rf.score(valid, y_valid))

# Save the RF
joblib.dump(rf, './data/rf_trained.pkl')

# Load RF
#rf = joblib.load('./data/rf_trained.pkl')