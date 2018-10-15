#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:00:23 2018

@author: guowei
"""

#import all packages used in this program
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#import warnings
#warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


#getdata and column names
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data'
                      ,header = None)

df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash', 
                   'Alcalinity of ash', 'Magnesium', 
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#split the data into trainning and test set
X = df_wine.iloc[:,1:].values
y = df_wine.iloc[:,0:1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=1)

#build a RF model
in_sample_scores = []
out_of_sample_scores = []
number_of_estimators = []

for num_estimators in range(10, 300, 20):
    forest = RandomForestClassifier(criterion = 'gini',
                                    max_depth = 4,
                                    n_estimators = num_estimators, 
                                    random_state = 1)

    forest.fit(X_train, y_train.ravel())
    cv_socres = cross_val_score(forest, X_train, y_train.ravel(), cv=10, scoring ='accuracy')
    y_pred = forest.predict(X_test)
    test_scores = accuracy_score(y_test, y_pred) 
    
    number_of_estimators.append(num_estimators)
    in_sample_scores.append(cv_socres.mean())
    out_of_sample_scores.append(test_scores)
 
#report scores into a dataframe
scores = pd.DataFrame({'number_of_estimators':number_of_estimators,
                      'in_sample_scores':in_sample_scores,
                      'out_of_sample_scores':out_of_sample_scores})
  
scores_desc = scores.describe()
scores_stat = pd.concat([scores,scores_desc])
scores_stat = scores_stat.drop(['count', 'min','25%', '50%','75%', 'max'])
scores_stat.loc['mean','number_of_estimators'] = '--'
scores_stat.loc['std','number_of_estimators'] = '--'    
     
#output the importance of each feature
forest = RandomForestClassifier(criterion = 'gini',
                                max_depth = 4,
                                n_estimators = 190, 
                                random_state = 1)

forest.fit(X_train, y_train.ravel())


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df_wine.columns[1:]
for f in range(X_train.shape[1]): 
    print("%2d) %-*s %f" % (f + 1, 30,
         feat_labels[indices[f]],
         importances[indices[f]])) 
    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]), importances[indices],
            align='center')    
    plt.xticks(range(X_train.shape[1]),feat_labels,rotation = 90)
plt.show()


#plot the Accuracy score VS n_estimators map
plt.plot(scores['number_of_estimators'],scores['in_sample_scores'],marker = 'o')
plt.title('Accuracy score VS n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy score')
plt.show()



###
print("My name is Wei Guo")
print("My NetID is: weiguo6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
