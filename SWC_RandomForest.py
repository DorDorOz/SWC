import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn

train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']]

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0)


##Random_Forest_1
rf_model_1 = RandomForestClassifier(random_state = 0, 
                                    n_estimators = 200,
                                    min_impurity_decrease = 0,
                                    min_samples_leaf = 1,
                                    min_samples_split = 2,
                                    min_weight_fraction_leaf = 0,
                                    n_jobs = -1,
                                    verbose = 1)
rf_model_1 = CalibratedClassifierCV(rf_model_1, method = 'isotonic', cv = 5)
rf_model_1.fit(X_train, y_train.values.ravel())

rf_model_1_prob = rf_model_1.predict(X_test)
rf_model_1_proba = rf_model_1.predict_proba(X_test)

rf_model_1_log_loss = log_loss(y_test, rf_model_1_proba)
print('rf_model_1_log_loss: ', rf_model_1_log_loss)

cm_rf_model_1 = confusion_matrix(y_test, rf_model_1_prob)

plt.figure(figsize = (15, 10))
sn.heatmap(cm_rf_model_1, annot = True, fmt='.5g').set(xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                        yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.xlabel('Predict')
plt.ylabel('Actual')



##Random_Forest_2
param_grid = { 
    'n_estimators': [100],
    'min_samples_leaf' : [1, 2, 4, 6],
    'max_features' : ['auto'],
    'min_impurity_decrease' : [0.0, 0.1, 0.2],
    'min_weight_fraction_leaf' : [0.0, 0.2, 0.4],
    'min_samples_split' : [2, 4],
    'criterion' : ['gini']}

CV_rf_model_2 = GridSearchCV(estimator = RandomForestClassifier(random_state = 0),
                             param_grid = param_grid, 
                             cv = 2, 
                             n_jobs = -1,
                             verbose = 10)

CV_rf_model_2.fit(X_train, y_train.values.ravel())
CV_rf_model_2.best_params_


rf_model_2 = RandomForestClassifier(random_state = 0, 
                                    n_estimators = 8000,
                                    criterion = 'gini',
                                    max_features = 'auto',
                                    verbose = 1,
                                    n_jobs = -1)


rf_model_2.fit(X_train, y_train.values.ravel())

rf_model_2_prob = rf_model_2.predict(X_test)
rf_model_2_proba = rf_model_2.predict_proba(X_test)

rf_model_2_log_loss = log_loss(y_test, rf_model_2_proba)
print('rf_model_2_log_loss: ', rf_model_2_log_loss)

cm_rf_model_2 = confusion_matrix(y_test, rf_model_2_prob)

plt.figure(figsize = (15, 10))
sn.heatmap(cm_rf_model_2, annot = True, fmt='.5g').set(xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                        yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.xlabel('Predict')
plt.ylabel('Actual')





