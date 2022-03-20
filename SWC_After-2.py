import numpy as np
import pandas as pd
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

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)

lgbm_1 = LGBMClassifier(objective = 'multiclass', 
                        max_depth = 16, 
                        num_leaves = 32, 
                        reg_alpha = 1,
                        reg_lambda = 1,
                        min_child_weight = 3,
                        min_child_samples = 4,
                        feature_fraction = 0.95,
                        bagging_fraction = 0.9,
                        bagging_freq = 3,
                        random_state = 0, 
                        num_class = 9, 
                        n_estimators = 100, 
                        n_jobs = -1)


xgb_1 = XGBClassifier(objective = 'multi:softprob', 
                      learning_rate = 0.1,
                      max_depth = 10, 
                      colsample_bytree = 0.4,
                      reg_alpha =  0.01,
                      reg_lambda = 1,
                      gamma = 0,
                      min_child_weight = 5,
                      random_state = 0, 
                      num_class = 9, 
                      n_estimators = 100, 
                      n_jobs = -1,
                      silent = False)

lr_1 = LogisticRegression(C = 0.5, max_iter = 2000, solver = 'lbfgs', multi_class = 'auto')

xgb_1.fit(X_train, y_train.values.ravel(), verbose = True)
y_pred = xgb_1.predict_proba(X_test.values)
print('LogLoss: ', log_loss(y_test, y_pred))






















