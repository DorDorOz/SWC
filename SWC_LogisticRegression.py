import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import  XGBClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']]

X_train, X_val, y_train, y_val = train_test_split(train_values, train_target, test_size = 0.05, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0)

##LR_1
lr_1 = LogisticRegression(random_state = 0, C = 100, solver = 'lbfgs', penalty = 'l2', multi_class = 'auto', max_iter = 2000)
lr_1.fit(X_train, y_train.values.ravel())

lr_1_proba_val = pd.DataFrame(lr_1.predict_proba(X_val))
lr_1_proba_test = pd.DataFrame(lr_1.predict_proba(X_test))

print('log_model_1_log_loss_val: ', log_loss(y_val, lr_1_proba_val))
print('log_model_1_log_loss_test: ', log_loss(y_test, lr_1_proba_test))

##LR_2
lr_2 = LogisticRegression(random_state = 0, C = 100, solver = 'liblinear', penalty = 'l2', multi_class = 'auto', max_iter = 2000)
lr_2.fit(X_train, y_train.values.ravel())

lr_2_proba_val = pd.DataFrame(lr_2.predict_proba(X_val))
lr_2_proba_test = pd.DataFrame(lr_2.predict_proba(X_test))

print('log_model_2_log_loss_val: ', log_loss(y_val, lr_2_proba_val))
print('log_model_2_log_loss_test: ', log_loss(y_test, lr_2_proba_test))

##SVC_1
svc_1 = SVC(kernel = 'rbf', C = 1, gamma = 'auto', probability = True)
svc_1.fit(X_train, y_train.values.ravel())

svc_1_proba_val = pd.DataFrame(svc_1.predict_proba(X_val))
svc_1_proba_test = pd.DataFrame(svc_1.predict_proba(X_test))

print('svc_1_log_loss_val: ', log_loss(y_val, svc_1_proba_val))
print('svc_1_log_loss_test: ', log_loss(y_test, svc_1_proba_test))

##KNN_1
knn_1 = KNeighborsClassifier(n_neighbors = 119)
knn_1.fit(X_train, y_train.values.ravel())

knn_1_proba_val = pd.DataFrame(knn_1.predict_proba(X_val))
knn_1_proba_test = pd.DataFrame(knn_1.predict_proba(X_test))

print('knn_1_log_loss_val: ', log_loss(y_val, knn_1_proba_val))
print('knn_1_log_loss_test: ', log_loss(y_test, knn_1_proba_test))

##RF_1
rf_1 = RandomForestClassifier(random_state = 0, n_estimators = 500, n_jobs = -1)
rf_1.fit(X_train, y_train.values.ravel())

rf_1_proba_val = pd.DataFrame(rf_1.predict_proba(X_val))
rf_1_proba_test = pd.DataFrame(rf_1.predict_proba(X_test))

print('rf_1_log_loss_val: ', log_loss(y_val, rf_1_proba_val))
print('rf_1_log_loss_test: ', log_loss(y_test, rf_1_proba_test))

##RF_2
rf_2 = RandomForestClassifier(random_state = 0, n_estimators = 500, n_jobs = -1, max_depth = 16)
rf_2.fit(X_train, y_train.values.ravel())

rf_2_proba_val = pd.DataFrame(rf_2.predict_proba(X_val))
rf_2_proba_test = pd.DataFrame(rf_2.predict_proba(X_test))

print('rf_2_log_loss_val: ', log_loss(y_val, rf_2_proba_val))
print('rf_2_log_loss_test: ', log_loss(y_test, rf_2_proba_test))

##XGB_1
xgb_1 = XGBClassifier(random_state = 0, n_estimators = 100, n_jobs = -1, max_depth = 10, min_child_weight = 10)
xgb_1.fit(X_train, y_train.values.ravel())

xgb_1_proba_val = pd.DataFrame(xgb_1.predict_proba(X_val))
xgb_1_proba_test = pd.DataFrame(xgb_1.predict_proba(X_test))

print('xgb_1_log_loss_val: ', log_loss(y_val, xgb_1_proba_val))
print('xgb_1_log_loss_test: ', log_loss(y_test, xgb_1_proba_test))

##XGB_2
xgb_2 = XGBClassifier(random_state = 0, n_estimators = 100, n_jobs = -1, max_depth = 8, min_child_weight = 12, subsample = 0.8)
xgb_2.fit(X_train, y_train.values.ravel())

xgb_2_proba_val = pd.DataFrame(xgb_2.predict_proba(X_val))
xgb_2_proba_test = pd.DataFrame(xgb_2.predict_proba(X_test))

print('xgb_2_log_loss_val: ', log_loss(y_val, xgb_2_proba_val))
print('xgb_2_log_loss_test: ', log_loss(y_test, xgb_2_proba_test))

##LGBM_1
lgbm_1 = lgbm_1 = LGBMClassifier(objective = 'multiclass')
lgbm_1.fit(X_train, y_train.values.ravel())

lgbm_1_proba_val = pd.DataFrame(lgbm_1.predict_proba(X_val))
lgbm_1_proba_test = pd.DataFrame(lgbm_1.predict_proba(X_test))

print('lgbm_1_log_loss_val: ', log_loss(y_val, lgbm_1_proba_val))
print('lgbm_1_log_loss_test: ', log_loss(y_test, lgbm_1_proba_test))

##LGBM_2
lgbm_2 = lgbm_2 = LGBMClassifier(objective = 'multiclass', max_depth = 8, num_leaves = 128)
lgbm_2.fit(X_train, y_train.values.ravel())

lgbm_2_proba_val = pd.DataFrame(lgbm_2.predict_proba(X_val))
lgbm_2_proba_test = pd.DataFrame(lgbm_2.predict_proba(X_test))

print('lgbm_2_log_loss_val: ', log_loss(y_val, lgbm_2_proba_val))
print('lgbm_2_log_loss_test: ', log_loss(y_test, lgbm_2_proba_test))

############################################################################
ens_df = pd.concat([lr_1_proba_test, lr_2_proba_test,
                    knn_1_proba_test,
                    svc_1_proba_test,
                    rf_1_proba_test, rf_2_proba_test,
                    xgb_1_proba_test, xgb_2_proba_test,
                    lgbm_1_proba_test, lgbm_2_proba_test], axis = 1)

##meta_Model
meta = LGBMClassifier(objective = 'multiclass', max_depth = 2, num_leaves = 32, 
                      min_child_weight = 10, n_estimators = 400, learning_rate = 0.05)

num_folds = 5
folds = KFold(n_splits = num_folds, random_state = 0, shuffle = False)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(ens_df, y_test)):
    X_train_fold, y_train_fold = ens_df.iloc[trn_idx], y_test.iloc[trn_idx]
    X_valid_fold, y_valid_fold = ens_df.iloc[val_idx], y_test.iloc[val_idx]
    
    meta.fit(X_train_fold.values, y_train_fold.values.ravel())
    y_pred_meta = meta.predict_proba(X_valid_fold.values)
    print('Fold', fold_, 'oof-LogLoss_meta: ', log_loss(y_valid_fold, pd.DataFrame(y_pred_meta)))














