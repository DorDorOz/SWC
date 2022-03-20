import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
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


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1


#pca = PCA(n_components = 50)
#pca.fit_transform(train_values) 
#plt.plot(pca.explained_variance_ratio_)
#plt.xlabel('Number of Components')
#plt.ylabel('Variance')
#plt.show()
#train_values = pd.DataFrame(pca.fit_transform(train_values))

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)

y_train.target.value_counts()
y_test.target.value_counts()

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)

watchlist = [(d_Matrix_test, 'test')]











