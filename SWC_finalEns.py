import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
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
from sklearn.externals import joblib



testFinal = pd.read_csv("D:/Python/data/SWC/testdata.csv")

d_Matrix_testFinal = xgb.DMatrix(testFinal)

xgb_1_p = xgb_model_1.predict(d_Matrix_testFinal)
xgb_2_p = xgb_model_2.predict(d_Matrix_testFinal)
xgb_4_p = xgb_model_4.predict(d_Matrix_testFinal)
xgb_5_p = xgb_model_5.predict(d_Matrix_testFinal)
xgb_6_p = xgb_model_6.predict(d_Matrix_testFinal)
xgb_7_p = xgb_model_7.predict(d_Matrix_testFinal)


lgbm_1_p = lgbm_model_1.predict(testFinal)
lgbm_2_p = lgbm_model_2.predict(testFinal)
lgbm_3_p = lgbm_model_3.predict(testFinal)
lgbm_4_p = lgbm_model_4.predict(testFinal)
lgbm_5_p = lgbm_model_5.predict(testFinal)






nelderMeadDF = pd.DataFrame(xgb_1_p * 0.45513348 + 
                         xgb_2_p * 0.2400976 + 
                         xgb_4_p * -0.10813321 + 
                         xgb_5_p * -0.09191581 +
                         xgb_6_p * -0.03484713 +
                         xgb_7_p * 0.48791544 + 
                         lgbm_1_p * -0.2406361 +
                         lgbm_2_p * 0.148275 +
                         lgbm_3_p * 0.22020494 +
                         lgbm_4_p * 0.12458667 +
                         lgbm_5_p * 0.30119606)

nelderMeadDF.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7' ,'c8', 'c9']
nelderMeadDF.to_csv(r'D:/Python/data/SWC/results_NelderMead.csv', index = False)

slsqpDF = pd.DataFrame(xgb_1_p * 0.14754187 + 
                    xgb_2_p * 0.07633769 + 
                    xgb_4_p * 0.09308489 + 
                    xgb_5_p * 0.07152513 +
                    xgb_6_p * 0.09603376 +
                    xgb_7_p * 0.0682989 + 
                    lgbm_1_p * 0.07623538 +
                    lgbm_2_p * 0.03716155 +
                    lgbm_3_p * 0.09512566 +
                    lgbm_4_p * 0.12067015 +
                    lgbm_5_p * 0.11798501)

slsqpDF.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7' ,'c8', 'c9']
slsqpDF.to_csv(r'D:/Python/data/SWC/results_SLSQP.csv', index = False)

avgDF = pd.DataFrame((xgb_1_p + 
                         xgb_2_p + 
                         xgb_4_p + 
                         xgb_5_p +
                         xgb_6_p +
                         xgb_7_p + 
                         lgbm_1_p +
                         lgbm_2_p +
                         lgbm_3_p +
                         lgbm_4_p +
                         lgbm_5_p )/11)

avgDF.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7' ,'c8', 'c9']
avgDF.to_csv(r'D:/Python/data/SWC/results_AVG.csv', index = False)




allavgDF = pd.DataFrame((nelderMeadDF + slsqpDF + avgDF) / 3 )
allavgDF.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7' ,'c8', 'c9']
allavgDF.to_csv(r'D:/Python/data/SWC/results_ALLAVG.csv', index = False)





log_loss(['1', '2', '2', '3'],
        [[1, 0, 0], 
         [0, 1, 0], 
         [0, 1, 0], 
         [-0.07, 0.1, 0.83]])





