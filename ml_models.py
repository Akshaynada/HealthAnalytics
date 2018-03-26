# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.txt')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)



"""
walking: 0
standing: 1
sitting: 2
laying down: 3
"""


# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Unit variance, mean zero 
sc = StandardScaler()
# Thats to transform the train  to scale it 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
# default = 3 folds 
scores = cross_val_score(svm_model_linear,X_train,y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

# Random Forest CLassifer
from sklearn.ensemble import RandomForestClassifier
# 10 decision trees , random state = 42 is seed 
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

accuracy = classifier.score(X_test,y_test)
cm = confusion_matrix(y_test,predictions)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X_train,y_train,cv=10)

from sklearn.tree import DecisionTreeClassifier

#Decision Tree classifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
 
# creating a confusion matrix
accuracy = dtree_model.score(X_test,y_test)
cm = confusion_matrix(y_test, dtree_predictions)
#k=fold cross validation
scores = cross_val_score(dtree_model,X_train,y_train,cv=10)



