import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression
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

X_train, X_val, y_train, y_val = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.5, random_state = 0)

#y_train.target.value_counts()
#y_test.target.value_counts()

#####################################################################################################################
d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_val = xgb.DMatrix(X_val, label = y_val)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)

watchlist = [(d_Matrix_val, 'val')]

##XGB_1
xgb_param_1 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 10,
        'colsample_bytree' : 0.4,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 1,
        'alpha' : 0.01,
        'lambda' : 1,
        'gamma' : 0,
        'min_child_weight' : 5,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'nthread' : -1,
        'num_class': 9} 
xgb_model_1 = xgb.train(params = xgb_param_1, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 10000,
                        early_stopping_rounds = 50,
                        evals = watchlist)

xgb_model_1_prob = xgb_model_1.predict(d_Matrix_test)

###########################################################################################################################

lgbm_train = lgbm.Dataset(X_train, label = y_train)
lgbm_val = lgbm.Dataset(X_val, label = y_val)
lgbm_test = lgbm.Dataset(X_test, label = y_test)

lgbm_params_1 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.1,
          'num_leaves': 256,
          'max_depth' : 10,
          'min_data_in_leaf' : 52,
          'min_sum_hessian_in_leaf' : 0.001,
          'min_gain_to_split' : 0,
          'lambda_l1' : 0.8,
          'lambda_l2' : 0,
          'bagging_fraction' : 0.6,
          'bagging_freq' : 10,
          'feature_fraction': 0.6,
          'max_delta_step' : 1}

lgbm_model_1 = lgbm.train(params = lgbm_params_1, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True,
                          num_boost_round = 4000,
                          early_stopping_rounds = 50)

lgbm_model_1_prob = lgbm_model_1.predict(X_test)

#########################################################################################################################

xgb_model_1_prob = pd.DataFrame(xgb_model_1_prob)
lgbm_model_1_prob = pd.DataFrame(lgbm_model_1_prob)

lgbm_model_1_prob.columns = ['9','10','11','12','13','14','15','16','17']
#lgbm_model_2_prob.columns = ['18','19','20','21','22','23','24','25','26']
#lgbm_model_3_prob.columns = ['27','28','29','30','31','32','33','34','35']
#lr_model_1_prob.columns = ['36','37','38','39','40','41','42','43','44']

train_2 = pd.concat([xgb_model_1_prob,
                     lgbm_model_1_prob,
                     pd.DataFrame(y_test).reset_index(drop = True)], axis = 1) 

train_2_values = train_2[train_2.columns.difference(['target'])]
train_2_target = train_2[['target']]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_2_values, train_2_target, 
                                                            test_size = 0.2, random_state = 0, stratify = train_2_target)

d_Matrix_train_2 = xgb.DMatrix(X_train_2, label = y_train_2)
d_Matrix_test_2 = xgb.DMatrix(X_test_2, label = y_test_2)
watchlist_2 = [(d_Matrix_test_2, 'test')]

###########################################################################################################################
xgb_param_2 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 8,
        'colsample_bytree' : 0.4,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 0.8,
        'alpha' : 0,
        'lambda' : 1.5,
        'gamma' : 0.5,
        'min_child_weight' : 1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'nthread' : -1,
        'num_class': 9} 
xgb_model_2 = xgb.train(params = xgb_param_2, 
                        dtrain = d_Matrix_train_2, 
                        num_boost_round = 5000,
                        early_stopping_rounds = 50,
                        evals = watchlist_2)

xgb_model_2_prob = xgb_model_2.predict(d_Matrix_test_2)











