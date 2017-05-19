# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

#X-Y点图模板
def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt
plt = runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.plot(X, y, 'k.')
plt.show()


from sklearn.linear_model import LinearRegression
# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)
print('预测一张12英寸匹萨价格：$%.2f' % model.predict([12])[0])


#拟合模型画出结果
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt = runplt()
plt.plot(X, y, 'k.')
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
plt.show()



#模型评估
X = [[6,2],[8,1],[10,0],[14,2],[18,0]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]    
    #残差图
model = LinearRegression()
model.fit(X, y)
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')
plt.show()
    #R方
print('验证r-squared: %.6f' % model.score(X_test, y_test))
print('r-squared: %.6f' % model.score(X, y))
    #参数t检验
    
    
    
    #F-检验：(对比剔除某些参数前后F值是否显著)
        #SSE(FM)=(y实际值-全模型预测y)^2
        #SSE(RM)=(y实际值-简化模型预测y)^2
        #k-简化模型k个参数 p-全模型p+1个参数
    #F=[SSE(RM)-SSE(FM)]/(p+1-k) / SSE(FM)/(n-p-1) 
    
    #SST=SSE+SSR SST=(y实际值-y均值)^2 SSE=(y实际值-预测y)^2 SSR=(预测y-y均值)^2
    #复相关系数-R^2=SSR/SST=1-SSE/SST
    #

    #R方 参数t检验 F-检验 AIC BIC Durbin-Watson(自相关度量，比如时间空间自相关)...
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
model = sm.OLS(y,X)
results = model.fit()
print(results.params)
print(results.summary())
y_fitted = results.fittedvalues
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(X, y, 'o', label='data')
ax.plot(X, y_fitted, 'r--.',label='OLS')
ax.legend(loc='best')
plt.show()
    
#模型参数选择
    #stepwise：无实现，自开发参考http://planspace.org/20150423-forward_selection_with_statsmodels/
        #1)第p个参数选择规则：取是Cp或RMSp最小的参数
            #RMSp=SSEp/(n-p) 
            #Cp准则 SSEp/SSEq+(2p-n)
         #2)参数选择停止规则(可选以下之一的调整)：
            #参数t-检验绝对值<=t0.05(n-p)停止(较严格)
            #参数t-检验绝对值<=1停止(较宽松)
   
   #问题
       #1)共线性会使FS-前向增加和BE-后向删除的参数选择结果不同
    

#二次回归
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt = runplt()
    #x^2:x平方
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.show()
print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('一元线性回归 r-squared', regressor.score(X_test, y_test))
print('二次回归 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))

#VIF
    #http://blog.sina.com.cn/s/blog_6e59e3730100vvdh.html
    #运用岭回归-岭迹删除变量，可依次参考如下准则删除
        #1.删除系数稳定但绝对值很小的变量，因为岭回归处理的是标准化的数据，故不同系数的数值大小可直接比较
        #2.删除系数不稳定二而无预测能力的变量，即趋于0的不稳定系数
        #3.删除一个或多个系数不稳定的变量，用剩下来的p个变量建立回归方程
    
#岭回归：多重共线性问题-增加系数惩罚项
#多重共性性
#https://www.zhihu.com/question/55089869?sort=created
#从多变量回归的变量选择来说，普通的多元线性回归要做的是变量的剔除和筛选，而岭回归是一种shrinkage的方法，就是收缩。这是什么意思呢，
#比如做普通线性回归时候，如果某个变量t检验不显著，我们通常会将它剔除再做回归，如此往复（stepwise)，最终筛选留下得到一个我们满意回归方程，
#但是在做岭回归的时候，我们并没有做变量的剔除，而是将这个变量的系数beta向0”收缩“，使得这个变量在回归方程中的影响变的很小。 
#于普通的多元线性回归相比，岭回归的变化更加smooth，或者说continuous。
#从这点上来说活，岭回归只是shrinkage methods中的一种，大家常说的lasso回归（貌似叫套索回归）其实也属于这种方法。

from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
reg.coef_
reg.intercept_ 

#岭迹

#梯度下降回归
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    #标准化
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('交叉验证R方值:', scores)
print('交叉验证R方均值:', np.mean(scores))
regressor.fit_transform(X_train, y_train)
print('测试集R方值:', regressor.score(X_test, y_test))








#方差
import numpy as np
print('残差平方和: %.2f' % np.mean((model.predict(X) - y) ** 2))
print('方差: %.2f' % np.var([6, 8, 10, 14, 18], ddof=1))
print('协方差: %.2f' % np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1])
#最小二乘法-矩阵求解    
from numpy.linalg import inv
from numpy import dot, transpose
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
print(dot(inv(dot(transpose(X), X)), dot(transpose(X), y)))
