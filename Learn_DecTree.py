# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:27:10 2017

@author: hum
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import zipfile
# 压缩节省空间
z = zipfile.ZipFile('d:\project\data\Internet Advertisements Data Set\\ad-dataset.zip')
df = pd.read_csv(z.open(z.namelist()[1]), header=None, low_memory=False)
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
# The last column describes the targets
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e == 'ad.' else 0 for e in response_variable_column]
X = df.loc[:, list(explanatory_variable_columns)]

X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline = Pipeline([
#('clf', DecisionTreeClassifier(criterion='entropy'))
('clf', RandomForestClassifier(criterion='entropy'))
])

parameters = {
'clf__n_estimators': (5, 10, 20, 50),
'clf__max_depth': (150, 155, 160),
'clf__min_samples_split': (2, 3, 4),
'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)
print('最佳效果：%0.3f' % grid_search.best_score_)
print('最优参数：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))

#决策树导出及图形
from sklearn import tree
clf=DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)
with open("adv.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

import os
os.unlink('adv.dot')

import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("adv.pdf") 


from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

 