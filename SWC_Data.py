import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")
train = train.sample(20000)

train = train[train.columns.difference(['v21', 'v7', 'v17', 'v14', 'v11', 
                                        'v1', 'v6', 'v5', 'v4', 'v24', 
                                        'v25', 'v26', 'v30', 'v3', 'v22'
                                        'v28', 'v23', 'v8', 'v2', 'v12',
                                        'v10'])]

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)


y_train.target.value_counts()
y_test.target.value_counts()
#y_val.target.value_counts()

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)
#d_Matrix_valid = xgb.DMatrix(X_val, label = y_val)
watchlist = [(d_Matrix_test, 'test')]


outlier_count = 0
for var in X_train.columns:
    var_z = stats.zscore(X_train[var])
    if((len(var_z[var_z > 3.0]) > 0) or(len(var_z[var_z < -3.0]) > 0)):
        #print("Feature with Outliers:",var)
        outlier_count = outlier_count + 1
print("Total Number of features that has outliers:", outlier_count)



