# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 03:47:57 2021

@author: Amarnadh Tadi
"""
import pandas as pd
import numpy as np

data=pd.read_excel(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign9\Coca_Rating_Ensemble.xlsx")
data.isnull().sum()
data.dropna(inplace=True)
password_tuple = np.array(data)
password_tuple
import random
random.shuffle(password_tuple)
y = [labels[1] for labels in password_tuple ]
x = [labels[0] for labels in password_tuple ]
def word_char(inputs):
    a= []
    for i in inputs:
        a.append(i)
    return a
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(tokenizer = word_char)
x =vect.fit_transform(x)
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
##Bagging technique
params = {'n_estimators': [40, 42], 'base_estimator__max_leaf_nodes':[10, 15], 'base_estimator__max_depth':[4, 5, 6]}
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(base_estimator=dt, oob_score=True, random_state=1) #n_estimators=70, random_state=1)
# Grid Search to determine best parameters
from sklearn.model_selection import GridSearchCV
bc_grid = GridSearchCV(estimator=bc, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
bc_grid.fit(x_train,y_train)
best_params = bc_grid.best_params_
print(best_params)

## passing best parameters to the baggong classifier

best_clf = BaggingClassifier(base_estimator=dt,oob_score=True, random_state=1)
best_clf.set_params(**best_params)
best_clf.fit(x_train,y_train)


from sklearn.metrics import accuracy_score,confusion_matrix
##Evaluation on training data set
confusion_matrix(y_train,best_clf.predict(x_train))
accuracy_score(y_train,best_clf.predict(x_train))

#Evaluation on test data
confusion_matrix(y_test,best_clf.predict(x_test))
accuracy_score(y_test,best_clf.predict(x_test))

#ADA boosting classifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(dt,random_state=1,learning_rate=0.02)
params1={'n_estimators': [2000,4000,5000], 'base_estimator__max_leaf_nodes':[10, 15], 'base_estimator__max_depth':[4, 5, 6]}
ada_grid = GridSearchCV(estimator=ada_clf, param_grid=params1, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
ada_grid.fit(x_train,y_train)
best_params = ada_grid.best_params_
print(best_params)

## passing best parameters to the baggong classifier

best_clf_ada = AdaBoostClassifier(dt,random_state=1,learning_rate=0.02)
best_clf_ada.set_params(**best_params)
best_clf_ada.fit(x_train,y_train)
##Evaluation on training data
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_train,best_clf_ada.predict(x_train))
accuracy_score(y_train,best_clf_ada.predict(x_train))
##Evaluation on test data
confusion_matrix(y_test,best_clf_ada.predict(x_test))
accuracy_score(y_test,best_clf_ada.predict(x_test))

##Gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))



##stacking classification
#bagging classifier,AdaBoost,GradientBoost

base_learners=[best_clf,best_clf_ada,boost_clf]
cancer_data.columns
array=cancer_data.values
x=array[:,1:31]
y=array[:,0]
train_x, train_y = x[:1200], y[:1200]
test_x, test_y = x[1200:], y[1200:]

# Create variables to store meta data and the targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))
from sklearn.model_selection import KFold
# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
    # Train each learner on the K-1 folds and create meta data for the Kth fold
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(train_x[train_indices], train_y[train_indices])
        predictions = learner.predict_proba(train_x[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
    meta_index += len(test_indices)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc =accuracy_score(test_y, learner.predict(test_x))
    base_acc.append(acc)
test_meta_data = test_meta_data.transpose()

# Fit the bagging model on the train set and evaluate it on the test set
best_clf.fit(meta_data, meta_targets)
ensemble_predictions = best_clf.predict(test_meta_data)

acc = accuracy_score(test_y, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')

##voting classifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
estimators = [best_clf,best_clf_ada,boost_clf]
model=VotingClassifier(estimators)
model.fit(x_train,y_train)
results=cross_val_score(model,x,y,cv=KF)
