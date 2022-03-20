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
from sklearn.utils import shuffle as shff
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import seaborn as sn


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)


y_train.target.value_counts()
y_test.target.value_counts()
y_val.target.value_counts()

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)
d_Matrix_valid = xgb.DMatrix(X_val, label = y_val)

watchlist = [(d_Matrix_test, 'test')]

#compute_sample_weight(class_weight = 'balanced', y = y_train)
#compute_sample_weight(class_weight = 'balanced', y = y_test)


xgb_model_1 = xgb.Booster({'nthread':-1}) #init model
xgb_model_1.load_model("xgb_model_1.model")
xgb_model_2 = xgb.Booster({'nthread':-1}) #init model
xgb_model_2.load_model("xgb_model_2.model")
xgb_model_3 = xgb.Booster({'nthread':-1}) #init model
xgb_model_3.load_model("xgb_model_3.model")
xgb_model_4 = xgb.Booster({'nthread':-1}) #init model
xgb_model_4.load_model("xgb_model_4.model")
lgbm_model_1 = joblib.load('lgbm_model_1.pkl')


xgb_model_1_prob = xgb_model_1.predict(d_Matrix_valid)
xgb_model_1_log_loss = log_loss(y_val, xgb_model_1_prob)
print('xgb_model_1_log_loss: ', xgb_model_1_log_loss)

xgb_model_2_prob = xgb_model_2.predict(d_Matrix_valid)
xgb_model_2_log_loss = log_loss(y_val, xgb_model_2_prob)
print('xgb_model_2_log_loss: ', xgb_model_2_log_loss)

xgb_model_3_prob = xgb_model_3.predict(d_Matrix_valid)
xgb_model_3_log_loss = log_loss(y_val, xgb_model_3_prob)
print('xgb_model_3_log_loss: ', xgb_model_3_log_loss)

xgb_model_4_prob = xgb_model_4.predict(d_Matrix_valid)
xgb_model_4_log_loss = log_loss(y_val, xgb_model_4_prob)
print('xgb_model_4_log_loss: ', xgb_model_4_log_loss)

lgbm_model_1_prob = lgbm_model_1.predict(X_val)
lgbm_model_1_log_loss = log_loss(y_val, lgbm_model_1_prob)
print('lgbm_model_1_log_loss: ', lgbm_model_1_log_loss)







log_loss(y_val, (xgb_model_1_prob * 1.2 + xgb_model_2_prob * 1.2 + xgb_model_3_prob + xgb_model_4_prob + lgbm_model_1_prob * 2) / 5)


xgb_model_1_pred = pd.DataFrame(xgb_model_1_prob)

xgb_model_2_pred = pd.DataFrame(xgb_model_2_prob)

xgb_model_4_pred = pd.DataFrame(xgb_model_4_prob)

lgbm_model_1_pred = pd.DataFrame(lgbm_model_1_prob)

xgb_model_3_pred = pd.concat([pd.DataFrame(xgb_model_3_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

models_preds = pd.concat([lgbm_model_1_pred,
                              xgb_model_1_pred, 
                              xgb_model_2_pred,
                              xgb_model_4_pred,
                              xgb_model_3_pred ], axis = 1) 


models_preds.columns = ['1', '2', '3', '4', '5', '6', '7' ,'8', '9',
                            '10', '11', '12', '13', '14', '15', '16', '17', '18',
                            '19', '20','21', '22', '23', '24', '25', '26', '27',
                            '28', '29','30', '31', '32', '33', '34', '35', '36',
                            '37', '38', '39', '40', '41', '42', '43', '44', '45',
                            'target']


models_preds_train_values = models_preds[models_preds.columns.difference(['target'])]
models_preds_train_target = models_preds[['target']]


X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(models_preds_train_values, 
                                                    models_preds_train_target, 
                                                    test_size = 0.3, 
                                                    random_state = 0)


d_Matrix_train_p = xgb.DMatrix(X_train_p, label = y_train_p)
d_Matrix_test_p = xgb.DMatrix(X_test_p, label = y_test_p)

watchlist_ens = [(d_Matrix_test_p, 'test')]




xgb_param_ens = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 2,
        'colsample_bytree' : 0.4,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 0.8,
        'gamma' : 1,
        'alpha' : 0.01,
        'lambda' : 0,
        'min_child_weight' : 5,
        'max_delta_step' : 1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_ens = xgb.train(params = xgb_param_ens, 
                        dtrain = d_Matrix_train_p, 
                        num_boost_round = 100000,
                        early_stopping_rounds = 25,
                        evals = watchlist_ens)

#0.481572

##########################################################################


lgbm_model_1_pred_2 = pd.concat([pd.DataFrame(lgbm_model_1_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

xgb_model_1_pred_2 = pd.concat([pd.DataFrame(xgb_model_1_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

xgb_model_2_pred_2 = pd.concat([pd.DataFrame(xgb_model_2_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

xgb_model_3_pred_2 = pd.concat([pd.DataFrame(xgb_model_3_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

xgb_model_4_pred_2 = pd.concat([pd.DataFrame(xgb_model_4_prob), 
                              pd.DataFrame(y_val).reset_index(drop = True)], axis = 1) 

models_preds_2 = pd.concat([xgb_model_1_pred_2, 
                              xgb_model_2_pred_2, 
                              xgb_model_3_pred_2,
                              xgb_model_4_pred_2,
                              lgbm_model_1_pred_2], axis = 0) 


models_preds_2.columns = ['1', '2', '3', '4', '5', '6', '7' ,'8', '9', 'target']


models_preds_2_train_values = models_preds_2[models_preds_2.columns.difference(['target'])]
models_preds_2_train_target = models_preds_2[['target']]


X_train_p_2, X_test_p_2, y_train_p_2, y_test_p_2 = train_test_split(models_preds_2_train_values, 
                                                    models_preds_2_train_target, 
                                                    test_size = 0.3, 
                                                    random_state = 0)


d_Matrix_train_p_2 = xgb.DMatrix(X_train_p_2, label = y_train_p_2)
d_Matrix_test_p_2 = xgb.DMatrix(X_test_p_2, label = y_test_p_2)

watchlist_ens_2 = [(d_Matrix_test_p_2, 'test')]


xgb_param_ens_2 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 8,
        'colsample_bytree' : 0.8,
        'subsample' : 1,
        'min_child_weight' : 3,
        'max_delta_step' : 0,
        'gamma' : 0,
        'alpha' : 0,
        'lambda' : 1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_ens_2 = xgb.train(params = xgb_param_ens_2, 
                        dtrain = d_Matrix_train_p_2, 
                        num_boost_round = 100000,
                        early_stopping_rounds = 25,
                        evals = watchlist_ens_2)

#test-mlogloss:0.456035


xgb_model_ens_2_prob = xgb_model_ens_2.predict(d_Matrix_test_p_2)

xgb_model_ens_2_log_loss = log_loss(y_test_p_2, xgb_model_ens_2_prob)
print('ens_1_log_loss: ', xgb_model_ens_2_log_loss)



cm_xgb_model_1 = confusion_matrix(y_test_p_2, np.asarray([np.argmax(line) for line in xgb_model_ens_2_prob]))

plt.figure(figsize = (15, 10))
sn.heatmap(cm_xgb_model_1, annot = True, fmt='.5g').set(xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                        yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.xlabel('Predict')
plt.ylabel('Actual')

























































