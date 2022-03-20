import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils import class_weight
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.externals import joblib

train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_val, y_train, y_val = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0, stratify = y_train)

y_train.target.value_counts()
y_test.target.value_counts()


lgbm_train = lgbm.Dataset(X_train, label = y_train)
lgbm_test = lgbm.Dataset(X_test, label = y_test)


##[1254]  valid_0's multi_logloss: 0.458806
lgbm_params_1 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.01,
          'max_bin': 55,
          'bin_construct_sample_cnt ': 5,
          'max_depth': 16,
          'num_leaves': 32,
          'reg_alpha' : 1,
          'reg_lambda' : 1,
          'min_child_weight' : 3,
          'min_child_samples' : 4,
          'feature_fraction': 0.95,
          'bagging_fraction': 0.9,
          'bagging_freq': 3}

lgbm_model_1 = lgbm.train(params = lgbm_params_1, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True,
                          num_boost_round = 1254,
                          early_stopping_rounds = 300)

lgbm_model_1_prob = lgbm_model_1.predict(X_test)

lgbm_model_1_log_loss = log_loss(y_test, lgbm_model_1_prob)
print('lgbm_model_1_log_loss: ', lgbm_model_1_log_loss)
################################################################################
##[4771]  valid_0's multi_logloss: 0.454218
lgbm_params_2 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.01,
          'max_depth': 16,
          'reg_alpha' : 0,
          'reg_lambda' : 0,
          'max_delta_step' : 1,
          'min_child_weight' : 5,
          'min_child_samples' : 25,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 3
          }
lgbm_model_2 = lgbm.train(params = lgbm_params_2, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True,
                          num_boost_round = 10000,
                          early_stopping_rounds = 100)


lgbm_model_2_prob = lgbm_model_2.predict(X_test)

lgbm_model_2_log_loss = log_loss(y_test, lgbm_model_2_prob)
print('lgbm_model_2_log_loss: ', lgbm_model_2_log_loss)

################################################################################
##[3273]  valid_0's multi_logloss: 0.447198
lgbm_params_3 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.01,
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
          'max_delta_step' : 1
          }
lgbm_model_3 = lgbm.train(params = lgbm_params_3, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True,
                          num_boost_round = 10000,
                          early_stopping_rounds = 50)


lgbm_model_3_prob = lgbm_model_3.predict(X_test)

lgbm_model_3_log_loss = log_loss(y_test, lgbm_model_3_prob)
print('lgbm_model_3_log_loss: ', lgbm_model_3_log_loss)
################################################################################
##[5205]  valid_0's multi_logloss: 0.455283
lgbm_params_4 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.01,
          'max_depth' : 10,
          'min_gain_to_split' : 0,
          'lambda_l1' : 0,
          'lambda_l2' : 0,
          'bagging_fraction' : 0.4,
          'bagging_freq' : 2,
          'feature_fraction': 0.4,
          'max_delta_step' : 1
          }
lgbm_model_4 = lgbm.train(params = lgbm_params_4, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True, num_boost_round = 5205, 
                          early_stopping_rounds = 100)


lgbm_model_4_prob = lgbm_model_4.predict(X_test)

lgbm_model_4_log_loss = log_loss(y_test, lgbm_model_4_prob)
print('lgbm_model_4_log_loss: ', lgbm_model_4_log_loss)
################################################################################
##[671]   valid_0's multi_logloss: 0.454850
lgbm_params_5 = {
          'task': 'train',
          'objective': 'multiclass',
		  'metric': 'multi_logloss',
          'num_class' : 9,
          'eta': 0.05,
          'max_bin': 55,
          'bin_construct_sample_cnt ': 5,
          'max_depth': 16,
          'num_leaves': 64,
          'reg_alpha' : 1,
          'reg_lambda' : 1,
          'min_child_weight' : 3,
          'min_child_samples' : 4,
          'feature_fraction': 0.95,
          'bagging_fraction': 0.9,
          'bagging_freq': 3}

lgbm_model_5 = lgbm.train(params = lgbm_params_5, 
                          train_set = lgbm_train,
                          valid_sets = lgbm_test,
                          verbose_eval = True,
                          num_boost_round = 671,
                          early_stopping_rounds = 300)

lgbm_model_5_prob = lgbm_model_5.predict(X_test)

lgbm_model_5_log_loss = log_loss(y_test, lgbm_model_5_prob)
print('lgbm_model_5_log_loss: ', lgbm_model_5_log_loss)
################################################################################
joblib.dump(lgbm_model_1, 'lgbm_model_1.pkl')
gbm_pickle = joblib.load('lgbm_model_1.pkl')


log_loss(y_val, gbm_pickle.predict(X_val))




joblib.dump(lgbm_model_1, 'lgbm_model_1.pkl')
joblib.dump(lgbm_model_2, 'lgbm_model_2.pkl')
joblib.dump(lgbm_model_3, 'lgbm_model_3.pkl')
joblib.dump(lgbm_model_4, 'lgbm_model_4.pkl')
joblib.dump(lgbm_model_5, 'lgbm_model_5.pkl')








































