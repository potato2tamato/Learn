# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:29:16 2017

@author: hum
"""

#逻辑回归Link函数
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
import numpy as np
plt.figure()
plt.axis([-6, 6, 0, 1])
plt.grid(True)
X = np.arange(-6,6,0.1)
y = 1 / (1 + np.e ** (-X))
plt.plot(X, y, 'b-');


#逻辑回归-垃圾邮件分类
    #1.Tfidf文本向量
    #2.文本向量逻辑回归预测        
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('d:\project\data\SMSSpamCollection', delimiter='\t', header=None) 
print(df.head())
print('含spam短信数量：', df[df[0] == 'spam'][0].count())
print('含ham短信数量：', df[df[0] == 'ham'][0].count())

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i, prediction in enumerate(predictions[-5:]):
    print('预测类型：%s. 信息：%s' % (prediction, X_test_raw.iloc[i]))
#模型评估    
    #准确率
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('准确率：',np.mean(scores), scores)
    #精准率：P=TP/(TP+FP)
    #召回度：R=TP/(TP+FN)
    #f1   : 1/F1+1/F1=1/P+1/R
from sklearn import preprocessing
lb = preprocessing.LabelEncoder()
lb.fit(y_train)
lb.inverse_transform([0,1])
precisions = cross_val_score(classifier, X_train, lb.transform(y_train), cv=5, scoring='precision')
print('精确率：', np.mean(precisions), precisions)
recalls = cross_val_score(classifier, X_train, lb.transform(y_train), cv=5, scoring='recall')
print('召回率：', np.mean(recalls), recalls)
fls = cross_val_score(classifier, X_train, lb.transform(y_train), cv=5, scoring='f1')
print('综合评价指标：', np.mean(fls), fls)

    #ROC AUC
from sklearn.metrics import roc_curve, auc
predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(lb.transform(y_test), predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

    #超参数网格搜索
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score
pipeline = Pipeline([
('vect', TfidfVectorizer(stop_words='english')),
('clf', LogisticRegression())
])
parameters = {
'vect__max_df': (0.25, 0.5, 0.75),
'vect__stop_words': ('english', None),
'vect__max_features': (2500, 5000, 10000, None),
'vect__ngram_range': ((1, 1), (1, 2)),
'vect__use_idf': (True, False),
'vect__norm': ('l1', 'l2'),
'clf__penalty': ('l1', 'l2'),
'clf__C': (0.01, 0.1, 1, 10),
} 
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
grid_search.fit(X_train_raw, y_train)
print('最佳效果：%0.3f' % grid_search.best_score_)
print('最优参数组合：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test_raw)
print('准确率：', accuracy_score(y_test, predictions))
print('精确率：', precision_score(lb.transform(y_test), lb.transform(predictions)))
print('召回率：', recall_score(lb.transform(y_test), lb.transform(predictions)))


#多分类模型
import zipfile
    #压缩节省空间
z = zipfile.ZipFile('d:\project\data\Sentiment Analysis on Movie Reviews\\train.tsv.zip')
df = pd.read_csv(z.open(z.namelist()[0]), header=0, delimiter='\t')
df.head()
df.count()
df.Phrase.head(10)
df.Sentiment.describe()
df.Sentiment.value_counts()
df.Sentiment.value_counts()/df.Sentiment.count()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
pipeline = Pipeline([
('vect', TfidfVectorizer(stop_words='english')),
('clf', LogisticRegression())
])
parameters = {
'vect__max_df': (0.25, 0.5),
'vect__ngram_range': ((1, 1), (1, 2)),
'vect__use_idf': (True, False),
'clf__C': (0.1, 1, 10),
}
import zipfile
X, y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('最佳效果：%0.3f' % grid_search.best_score_)
print('最优参数组合：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

#模型评估
predictions = grid_search.predict(X_test)
print('准确率：', accuracy_score(y_test, predictions))
print('混淆矩阵：', confusion_matrix(y_test, predictions))
print('分类报告：', classification_report(y_test, predictions))



