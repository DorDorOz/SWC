import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import seaborn as sn
from scipy.optimize import minimize
import scipy.sparse


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_val, y_train, y_val = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.5, random_state = 0, stratify = y_train)


y_train.target.value_counts()
y_test.target.value_counts()

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_val = xgb.DMatrix(X_val, label = y_val)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)

watchlist = [(d_Matrix_val, 'val')]



##xgb.__version__
##XGB_1
xgb_param_1 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 2,
        'min_child_weight' : 12,
        'subsample' : 0.6,
        'colsample_bytree' : 0.6,
        'colsample_bynode' : 0.6,
        'colsample_bylevel' : 0.6,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'nthread' : -1,
        'num_class': 9} 

xgb_model_1 = xgb.train(params = xgb_param_1, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 10000,
                        early_stopping_rounds = 25,
                        evals = watchlist)

xgb_model_1_prob = xgb_model_1.predict(d_Matrix_test)

xgb_model_1_log_loss = log_loss(y_test, xgb_model_1_prob)
print('xgb_model_1_log_loss: ', xgb_model_1_log_loss)
###########################################################################








