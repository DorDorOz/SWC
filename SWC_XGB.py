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

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)

y_train.target.value_counts()
y_test.target.value_counts()

#X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
#X_test = preprocessing.MinMaxScaler().fit_transform(X_test)

#d_Matrix_train = xgb.DMatrix(scipy.sparse.csr_matrix(X_train), label = y_train)
#d_Matrix_test = xgb.DMatrix(scipy.sparse.csr_matrix(X_test), label = y_test)

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)


watchlist = [(d_Matrix_test, 'test')]



##xgb.__version__
##XGB_1
##[782]   test-mlogloss:0.451880
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
                        early_stopping_rounds = 25,
                        evals = watchlist)

xgb_model_1_prob = xgb_model_1.predict(d_Matrix_test)

xgb_model_1_log_loss = log_loss(y_test, xgb_model_1_prob)
print('xgb_model_1_log_loss: ', xgb_model_1_log_loss)
###########################################################################
##XGB_10
##[214]   test-mlogloss:0.452173
xgb_param_10 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 2,
        'max_leaves' : 2,
        'colsample_bytree' : 0.4,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 1,
        'alpha' : 0,
        'lambda' : 1,
        'gamma' : 0,
        'min_child_weight' : 5,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_10 = xgb.train(params = xgb_param_10, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 10000,
                        early_stopping_rounds = 100,
                        evals = watchlist)

xgb_model_10_prob = xgb_model_10.predict(d_Matrix_test)

xgb_model_10_log_loss = log_loss(y_test, xgb_model_10_prob)
print('xgb_model_10_log_loss: ', xgb_model_10_log_loss)
###########################################################################
##XGB_8
##[82]    test-mlogloss:0.494282
xgb_param_8 = {
        'eta' : 0.5,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 0,
        'max_leaves' : 48,
        'colsample_bytree' : 1,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 1,
        'alpha' : 0,
        'lambda' : 1,
        'gamma' : 0,
        'min_child_weight' : 5,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_8 = xgb.train(params = xgb_param_8, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 100000,
                        early_stopping_rounds = 25,
                        evals = watchlist)

xgb_model_8_prob = xgb_model_8.predict(d_Matrix_test)

xgb_model_8_log_loss = log_loss(y_test, xgb_model_8_prob)
print('xgb_model_8_log_loss: ', xgb_model_8_log_loss)
###########################################################################
#xgb_model_1.save_model('xgb_model_3.model')
#bst = xgb.Booster({'nthread':-1}) #init model
#bst.load_model("xgb_model_2.model")
#print(log_loss(y_test, bst.predict(d_Matrix_test)))

#cm_xgb_model_1 = confusion_matrix(y_val, np.asarray([np.argmax(line) for line in xgb_model_3_prob]))
#
#plt.figure(figsize = (15, 10))
#sn.heatmap(cm_xgb_model_1, annot = True, fmt='.5g').set(xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
#                                                        yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
#
#plt.xlabel('Predict')
#plt.ylabel('Actual')



#################################################################
#param_grid = { 
#        'n_estimators' : [100],
#        'eta': [0.1],
#        'gamma' : [0],
#        'max_depth' : [12],
#        'colsample_bytree': [0.4],
#        'subsample' : [1],
#        'min_child_weight' : [3, 5, 8],
#        'num_class': [9],
#        'eval_metric' : ['mlogloss'],
#        'early_stopping_rounds' : [25],
#        'eval_set' : [(X_test, y_test)],
#        'objective': ['multi:softprob'],
#        'tree_method' : ['hist'],
#        'grow_policy' : ['lossguide']}
#
#xgb_model = XGBClassifier(random_state = 0, n_jobs = -1, verbose = 10)
#
#CV_model = GridSearchCV(estimator = xgb_model,
#                        param_grid = param_grid,
#                        n_jobs = -1,
#                        cv = StratifiedKFold(n_splits = 5, random_state = 0),
#                        verbose = 10)
#
#CV_model.fit(X_train, y_train.values.ravel(), verbose = 10)
#print(CV_model.best_estimator_)
#print(CV_model.best_score_)
##############################################################################
##XGB_2
##[1491]  test-mlogloss:0.455033
xgb_param_2 = {
        'eta' : 0.05,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 16,
        'max_leaves' : 32,
        'colsample_bylevel' : 0.4,
        'subsample' : 1,
        'min_child_weight' : 10,
        'alpha' : 2.1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_2 = xgb.train(params = xgb_param_2, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 1492,
                        early_stopping_rounds = 300,
                        evals = watchlist)

xgb_model_2_prob = xgb_model_2.predict(d_Matrix_test)

xgb_model_2_log_loss = log_loss(y_test, xgb_model_2_prob)
print('xgb_model_2_log_loss: ', xgb_model_2_log_loss)



####################################################################
##XGB_3
##[1008]  test-mlogloss:0.464229
xgb_param_3 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 8,
        'max_leaves' : 36,
        'min_child_weight' : 3,
        'alpha' : 4,
        'lambda' : 3,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_3 = xgb.train(params = xgb_param_3, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 1009,
                        early_stopping_rounds = 50,
                        evals = watchlist)

xgb_model_3_prob = xgb_model_3.predict(d_Matrix_test)

xgb_model_3_log_loss = log_loss(y_test, xgb_model_3_prob)
print('xgb_model_3_log_loss: ', xgb_model_3_log_loss)

####################################################################
##XGB_4
##[5579]  test-mlogloss:0.456630
xgb_param_4 = {
        'eta' : 0.01,
        'tree_method' : 'hist',
        'grow_policy' : 'lossguide',
        'max_depth' : 16,
        'max_leaves' : 32,
        'colsample_bylevel' : 0.4,
        'subsample' : 1,
        'min_child_weight' : 10,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_4 = xgb.train(params = xgb_param_4, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 5580,
                        early_stopping_rounds = 50,
                        evals = watchlist)

xgb_model_4_prob = xgb_model_4.predict(d_Matrix_test)

xgb_model_4_log_loss = log_loss(y_test, xgb_model_4_prob)
print('xgb_model_4_log_loss: ', xgb_model_4_log_loss)



##########################################################################
##XGB_5
##[1473]  test-mlogloss:0.453967
xgb_param_5 = {
        'eta' : 0.05,
        'tree_method' : 'hist',
        'gamma' : 0.2,
        'max_depth' : 6,
        'colsample_bytree' : 0.8,
        'colsample_bylevel' : 0.6,
        'colsample_bynode' : 0.8,
        'subsample' : 0.8,
        'min_child_weight' : 3,
        'max_delta_step' : 1,
        'lambda' : 1,
        'alpha' : 1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_5 = xgb.train(params = xgb_param_5, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 1474,
                        early_stopping_rounds = 150,
                        evals = watchlist)

xgb_model_5_prob = xgb_model_5.predict(d_Matrix_test)

xgb_model_5_log_loss = log_loss(y_test, xgb_model_5_prob)
print('xgb_model_5_log_loss: ', xgb_model_5_log_loss)
##########################################################################
##XGB_6
##[1638]  test-mlogloss:0.458783
xgb_param_6 = {
        'eta' : 0.05,
        'tree_method' : 'hist',
        'gamma' : 0,
        'max_depth' : 24,
        'max_leaves' : 48,
        'colsample_bytree' : 1,
        'colsample_bylevel' : 0.4,
        'colsample_bynode' : 1,
        'subsample' : 1,
        'min_child_weight' : 20,
        'lambda' : 1,
        'alpha' : 4,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_6 = xgb.train(params = xgb_param_6, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 1639,
                        early_stopping_rounds = 35,
                        evals = watchlist)

xgb_model_6_prob = xgb_model_6.predict(d_Matrix_test)

xgb_model_6_log_loss = log_loss(y_test, xgb_model_6_prob)
print('xgb_model_6_log_loss: ', xgb_model_6_log_loss)
##########################################################################
##XGB_7
##[473]   test-mlogloss:0.453416
xgb_param_7 = {
        'eta' : 0.1,
        'tree_method' : 'hist',
        'gamma' : 0.06,
        'max_depth' : 8,
        'colsample_bytree' : 0.8,
        'colsample_bylevel' : 0.6,
        'colsample_bynode' : 0.8,
        'subsample' : 0.8,
        'min_child_weight' : 3,
        'max_delta_step' : 1,
        'lambda' : 1,
        'alpha' : 1,
        'objective': 'multi:softprob',
        'eval_metric' : 'mlogloss',
        'num_class': 9} 
xgb_model_7 = xgb.train(params = xgb_param_7, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 474,
                        early_stopping_rounds = 35,
                        evals = watchlist)

xgb_model_7_prob = xgb_model_7.predict(d_Matrix_test)

xgb_model_7_log_loss = log_loss(y_test, xgb_model_7_prob)
print('xgb_model_7_log_loss: ', xgb_model_7_log_loss)
##########################################################################

xgb_model_1_log_loss = log_loss(y_test, xgb_model_1_prob)
print('xgb_model_1_log_loss: ', xgb_model_1_log_loss)

xgb_model_2_log_loss = log_loss(y_test, xgb_model_2_prob)
print('xgb_model_2_log_loss: ', xgb_model_2_log_loss)

xgb_model_3_log_loss = log_loss(y_test, xgb_model_3_prob)
print('xgb_model_3_log_loss: ', xgb_model_3_log_loss)

xgb_model_4_log_loss = log_loss(y_test, xgb_model_4_prob)
print('xgb_model_4_log_loss: ', xgb_model_4_log_loss)

xgb_model_5_log_loss = log_loss(y_test, xgb_model_5_prob)
print('xgb_model_5_log_loss: ', xgb_model_5_log_loss)

xgb_model_6_log_loss = log_loss(y_test, xgb_model_6_prob)
print('xgb_model_6_log_loss: ', xgb_model_6_log_loss)

xgb_model_7_log_loss = log_loss(y_test, xgb_model_7_prob)
print('xgb_model_7_log_loss: ', xgb_model_7_log_loss)

lgbm_model_1_log_loss = log_loss(y_test, lgbm_model_1_prob)
print('lgbm_model_1_log_loss: ', lgbm_model_1_log_loss)

lgbm_model_2_log_loss = log_loss(y_test, lgbm_model_2_prob)
print('lgbm_model_2_log_loss: ', lgbm_model_2_log_loss)

lgbm_model_3_log_loss = log_loss(y_test, lgbm_model_3_prob)
print('lgbm_model_3_log_loss: ', lgbm_model_3_log_loss)

lgbm_model_4_log_loss = log_loss(y_test, lgbm_model_4_prob)
print('lgbm_model_4_log_loss: ', lgbm_model_4_log_loss)

lgbm_model_5_log_loss = log_loss(y_test, lgbm_model_5_prob)
print('lgbm_model_5_log_loss: ', lgbm_model_5_log_loss)








clfs = []
clfs.append(xgb_model_1)
clfs.append(xgb_model_2)
#clfs.append(xgb_model_3)
clfs.append(xgb_model_4)
clfs.append(xgb_model_5)
clfs.append(xgb_model_6)
clfs.append(xgb_model_7)
clfs.append(lgbm_model_1)
clfs.append(lgbm_model_2)
clfs.append(lgbm_model_3)
clfs.append(lgbm_model_4)
clfs.append(lgbm_model_5)

predictions = []
predictions.append(xgb_model_1_prob)
predictions.append(xgb_model_2_prob)
#predictions.append(xgb_model_3_prob)
predictions.append(xgb_model_4_prob)
predictions.append(xgb_model_5_prob)
predictions.append(xgb_model_6_prob)
predictions.append(xgb_model_7_prob)
predictions.append(lgbm_model_1_prob)
predictions.append(lgbm_model_2_prob)
predictions.append(lgbm_model_3_prob)
predictions.append(lgbm_model_4_prob)
predictions.append(lgbm_model_5_prob)



def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y_test, final_prediction)

starting_values = [0.01] * len(predictions)
constraints = ({'type':'eq','fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(predictions)

res = minimize(log_loss_func, starting_values, method = 'Nelder-Mead', constraints = constraints, bounds = bounds)

print('Ensemble Score: {best_score}'.format(best_score = res['fun']))
print('Best Weights: {weights}'.format(weights = res['x']))


nealmenderDFTest = pd.DataFrame(xgb_model_1_prob * 0.45513348 + 
                    xgb_model_2_prob * 0.2400976 + 
                    xgb_model_4_prob * -0.10813321 + 
                    xgb_model_5_prob * -0.09191581 +
                    xgb_model_6_prob * -0.03484713 +
                    xgb_model_7_prob * 0.48791544 + 
                    lgbm_model_1_prob * -0.2406361 +
                    lgbm_model_2_prob * 0.148275 +
                    lgbm_model_3_prob * 0.22020494 +
                    lgbm_model_4_prob * 0.12458667 +
                    lgbm_model_5_prob * 0.30119606)

slsqpDFTest = pd.DataFrame(xgb_model_1_prob * 0.14754187 + 
                    xgb_model_2_prob * 0.07633769 + 
                    xgb_model_4_prob * 0.09308489 + 
                    xgb_model_5_prob * 0.07152513 +
                    xgb_model_6_prob * 0.09603376 +
                    xgb_model_7_prob * 0.0682989 + 
                    lgbm_model_1_prob * 0.07623538 +
                    lgbm_model_2_prob * 0.03716155 +
                    lgbm_model_3_prob * 0.09512566 +
                    lgbm_model_4_prob * 0.12067015 +
                    lgbm_model_5_prob * 0.11798501)

simpleAVGDFTest = pd.DataFrame((xgb_model_1_prob + 
                    xgb_model_2_prob + 
                    xgb_model_4_prob + 
                    xgb_model_5_prob +
                    xgb_model_6_prob +
                    xgb_model_7_prob + 
                    lgbm_model_1_prob +
                    lgbm_model_2_prob +
                    lgbm_model_3_prob +
                    lgbm_model_4_prob +
                    lgbm_model_5_prob)/11)


print(log_loss(y_test, nealmenderDFTest))
print(log_loss(y_test, simpleAVGDFTest))
print(log_loss(y_test, slsqpDFTest))
print(log_loss(y_test, (nealmenderDFTest + simpleAVGDFTest + slsqpDFTest) / 3))












#xgb_model_1.save_model('xgb_model_1.model')
#xgb_model_2.save_model('xgb_model_2.model')
#xgb_model_3.save_model('xgb_model_3.model')
#xgb_model_4.save_model('xgb_model_4.model')
#xgb_model_5.save_model('xgb_model_5.model')
#xgb_model_6.save_model('xgb_model_6.model')
#xgb_model_7.save_model('xgb_model_7.model')
#xgb_model_8.save_model('xgb_model_8.model')











