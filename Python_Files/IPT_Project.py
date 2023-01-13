# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:02:31 2022

@author: semih
"""

# Kütüphaneler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# -----------------------------------------------------------------------------

dataset_original = pd.read_csv("Exam_Data_Set.csv")
df = dataset_original.copy()


# Veri ön işleme süreci
X = df.drop(['math score'], axis=1)
reg_y = df['math score']

X = pd.concat([X, pd.get_dummies(X['gender']), pd.get_dummies(X['race/ethnicity']), pd.get_dummies(X['lunch'])], axis=1)

categoricial_to_nums = {
                        'some high school'          :   1,
                        'some college'              :   2,
                        'high school'               :   3,
                        'associate\'s degree'       :   4,
                        'bachelor\'s degree'        :   5,
                        'master\'s degree'          :   6,
                        'completed'                 :   1,
                        'none'                      :   0
                        }
X = X.replace(categoricial_to_nums)
nll = X.isna().sum()

del X['gender'], X['race/ethnicity'], X['lunch']
# -----------------------------------------------------------------------------

# Veriyi train ve testa olarak ayırma
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(X, reg_y, test_size=0.2, random_state=0)
# -----------------------------------------------------------------------------

def Calculate(score):
    if score < 60:
        return "Başarısız"
    else:
        return "Başarılı"
    
from sklearn.preprocessing import LabelEncoder
y = df.apply(lambda x: Calculate(x['math score']),axis=1)
csn = {'Başarılı' : 1,
       'Başarısız' : 0}
y = y.replace(csn)
y_str = df.apply(lambda x: Calculate(x['math score']), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 0)



# CLASSIFICATION MODELS
import Classification.IPT_CLS_KNN_Algorithm as clsKnnAlgorithm
cls_knn_values = clsKnnAlgorithm.KNN_Algorithm_Semih(X_train, X_test, y_train, y_test, 100)

import Classification.IPT_CLS_DT_Algorithm as dtAlgorithm
cls_dt_values = dtAlgorithm.Decision_Tree_Algorithm(X_train, X_test, y_train, y_test, y_str, df)


import Classification.IPT_CLS_LR_Algorithm as lrAlgorithm
cls_lr_values = lrAlgorithm.Logistic_Regression_Algorithm(X_train, X_test, y_train, y_test)


import Classification.IPT_CLS_NB_Algorithm as nbAlgorithm
cls_nb_values = nbAlgorithm.NB_Algorithm(X_train, y_train, X_test, y_test)
"""


# REGRESYON Modeller

import IPT_KNN_Algorithm as knnAlgorithm
knn_values = knnAlgorithm.KNN_Algorithm_Semih(X_train, X_test, y_train, y_test)


import IPT_DecisionTree_Algorithm as dtAlgorithm
dt_values = dtAlgorithm.Decision_Tree_Semih(X_train, X_test, y_train, y_test)


import IPT_LogisticRegression_Algorithm as lrAlgorithm
lr_values = lrAlgorithm.Linear_Regression_Semih(X_train, X_test, y_train, y_test)


import IPT_SVM_Algorithm as svmAlgorithm
svm_values = svmAlgorithm.Support_Vector_Machine_Semih(X_train, X_test, y_train, y_test)


import IPT_NN_Algorithm as nnAlgorithm
nn_values = nnAlgorithm.Nural_Network_Semih(X_train, X_test, y_train, y_test)
"""
# -----------------------------------------------------------------------------










































