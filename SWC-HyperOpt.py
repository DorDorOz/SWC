import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy.optimize import minimize
import scipy.sparse
from sklearn.decomposition import PCA
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

#pca = PCA(n_components = 30)
#pca.fit_transform(train_values) 
#train_values = pd.DataFrame(pca.fit_transform(train_values))

X_train, X_val, y_train, y_val = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.5, random_state = 0)

xgboost_space = {
            'learning_rate ' : hp.quniform('eta', 0.01, 0.1, 0.01), 
            'max_depth' : hp.choice('max_depth', np.arange(2, 14, dtype = int)),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 12, 1),
            'subsample' : hp.quniform('subsample', 0.4, 1, 0.05),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.4, 1, 0.05),
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.4, 1, 0.05),
            'colsample_bynode' : hp.quniform('colsample_bynode', 0.4, 1, 0.05),
            'gamma' : hp.quniform('gamma', 0, 1, 0.05),
            'reg_alpha' :  hp.quniform('reg_alpha', 0, 1, 0.1),
            'reg_lambda' : hp.quniform('reg_lambda', 1, 2, 0.1),
            'objective' : 'multi:softprob',
            'num_class' : 9,
            'nthread' : -1,
            'tree_method' : 'hist',
            'grow_policy' : 'lossguide'}

best_score = 1.0

def objective(space):
    
    global best_score
    xgb_model = XGBClassifier(**space,
                              n_estimators = 10000,
                              n_jobs = -1)   
    
    xgb_model.fit(X_train, y_train.values.ravel(), 
                  eval_metric = 'mlogloss',
                  early_stopping_rounds = 50,
                  eval_set = [(X_val, y_val.values.ravel())],
                  verbose = False)
    score = log_loss(y_test, xgb_model.predict_proba(X_test))
    if (score < best_score):
        best_score = score
    
    return score

start = time.time()
trials = Trials()
best = fmin(objective, 
            space = xgboost_space, 
            algo = tpe.suggest, 
            max_evals = 100, 
            trials = trials)

print("Hyperopt search took %.2f seconds for 100 candidates" % ((time.time() - start)))
print(-best_score, best)



#{'colsample_bylevel': 0.6000000000000001, 
# 'colsample_bynode': 0.6000000000000001, 
# 'colsample_bytree': 0.6000000000000001, 
# 'eta': 0.02, 
# 'gamma': 0.1, 
# 'max_depth': 8, 
# 'min_child_weight': 3.0, 
# 'reg_alpha': 0.6000000000000001, 
# 'reg_lambda': 2.0, 
# 'subsample': 0.9}



















