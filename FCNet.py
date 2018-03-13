

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sensor_group2.csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder



from keras.utils.np_utils import to_categorical



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)


"""
walking: 0
standing: 1
sitting: 2
laying down: 3
"""
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()


classifier.add(Dense(units=2, kernel_initializer="uniform",activation="relu",input_shape=(X_train.shape[1],)))
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=4,kernel_initializer="uniform",activation="softmax"))

classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


classifier.fit(X_train,y_train,batch_size=256,epochs=10)






# Predicting the Test set results



loss, accuracy = classifier.evaluate(X_test, y_test)
print(accuracy)
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = new_prediction > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN
#use k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

""" k fold cross validation steps"""
def build_classifier():
    classifier = Sequential()


    classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu",input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1,verbose=1)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()


    classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu",input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {"batch_size": [25,32],"epochs": [100,500],"optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,verbose=1,n_jobs=1)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_