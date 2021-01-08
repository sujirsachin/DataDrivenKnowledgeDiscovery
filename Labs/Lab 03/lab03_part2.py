# Author : Sachin Mohan Sujir
# Lab03-Classification Part2


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import pandas
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def printScores(model):
    print("Training Accuracy Score: {}".format(model.score(X_train,y_train.ravel())))
    model_pred = model.predict(X_test)
    print("Test Accuracy Score for this model : {}".format(accuracy_score(y_test, model_pred)))



#  (a)  read csv
data = pd.read_csv('./Default.csv', header=0)



# (b) extracting predictors
feature_vectors = ['balance','income']
X = data[feature_vectors].values
y = data[['default']].values


# (c)  splitting dataset into test and train
X_train = X[0:8000, :]
X_test = X[8000:,:]
y_train = y[0:8000,:]
y_test = y[8000:,:]


# (d) Logistic Regression model
logRegressionModel = LogisticRegression().fit(X_train, y_train.ravel())
print("Model: Logistic Regression: ")
printScores(logRegressionModel)
print("")
# (e) Linear Discriminant Analysis model
LDAModel = LinearDiscriminantAnalysis().fit(X_train, y_train.ravel())
print("Model: LDA: ")
printScores(LDAModel)
print("")


# (f) KNN model
KNN1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train.ravel())
print("KNN with K as 1: ")
printScores(KNN1)
print("")
KNN5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train.ravel())
print("KNN with K as 5: ")
printScores(KNN5)
print("")
KNN50 = KNeighborsClassifier(n_neighbors=50).fit(X_train, y_train.ravel())
print("KNN with K as 50: ")
printScores(KNN50)
print("")

KNN100 = KNeighborsClassifier(n_neighbors=100).fit(X_train, y_train.ravel())
print("KNN with K as 100: ")
printScores(KNN100)
print("")


# Question(g)
QDAModel = QuadraticDiscriminantAnalysis().fit(X_train, y_train.ravel())
print("Model: QDA: ")
printScores(QDAModel)














