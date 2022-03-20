import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import scipy.sparse
from sklearn.decomposition import PCA


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")
#train = train.sample(frac = 0.2, random_state = 0)

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)
 
lgbm_1 = LGBMClassifier(random_state = 0, objective = 'multiclass')
xgb_1 = XGBClassifier(random_state = 0, objective = 'multi:softprob')

num_folds = 5
folds = StratifiedKFold(n_splits = num_folds, random_state = 0, shuffle = False)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    X_train_fold, y_train_fold = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
    X_valid_fold, y_valid_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    lgbm_1.fit(X_train_fold.values, y_train_fold.values.ravel())
    y_pred_lgbm_1 = lgbm_1.predict_proba(X_valid_fold.values)
    
    xgb_1.fit(X_train_fold.values, y_train_fold.values.ravel())
    y_pred_xgb_1 = xgb_1.predict_proba(X_valid_fold.values)
    
    print('Fold', fold_, 'oof-LogLoss_lgbm_1    : ', log_loss(y_valid_fold, pd.DataFrame(y_pred_lgbm_1)))
    print('Fold', fold_, 'oof-LogLoss_xgb_1     : ', log_loss(y_valid_fold, pd.DataFrame(y_pred_xgb_1)))















