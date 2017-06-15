# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:34:54 2016
Kaggle
@author: wangx3
"""
import numpy as np
import pandas as pd
import sklearn.datasets as sds
import sklearn.model_selection as skm
import sklearn.preprocessing as spr
import sklearn.feature_extraction as skfe
import sklearn.feature_extraction.text as skft
import sklearn.linear_model as sklin
import sklearn.svm as svm
import sklearn.naive_bayes as skb
import sklearn.neighbors as skn
import sklearn.tree as skt
import sklearn.ensemble as ske
import sklearn.metrics as skmt
import sklearn.cluster as skc
import sklearn.decomposition as skd
import matplotlib.pyplot as plt


# 监督学习模型
# 分类
# 逻辑回归，随机梯度参数估计
# cancer data用Pandas从互联网读取指定数据
# 创建特征列表
column_names = ['Sampe Code Number', 
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 
                'Marginal Adhesion', 
                'Single Epothelial Cell Size', 
                'Bare Nuclei', 
                'Bland Chromatin', 
                'Normal Nucleoli', 
                'Mitoses', 
                'Class']
server = 'https://archive.ics.uci.edu'
folder = '/ml/machine-learning-databases/breast-cancer-wisconsin'
file = '/breast-cancer-wisconsin.data'
source = server + folder + file
data = pd.read_csv(source, names=column_names)
data = data.replace('?', np.nan)    # 用np.nan替代？号
data = data.dropna(how='any')    # 丢弃有缺失值的记录
# 分割数据，随机采样25%用于测试，75%用于训练
# 分割函数被移到model_selection模块
x = data[column_names[1:10]]
y = data[column_names[10]]
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
# 标准化数据，方差为1，均值为0, zero mean and unit variance
ss = spr.StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 线性模型
# 初始化LR和SGDC
lr = sklin.LogisticRegression()
sgdc = sklin.SGDClassifier()
# 调用fit函数和训练数据训练模型
lr.fit(x_train, y_train)
sgdc.fit(x_train, y_train)
# 用训练好的模型对测试数据进行预测，预测结果保存在y_predict中
lr_y_predict = lr.predict(x_test)
sgdc_y_predict = sgdc.predict(x_test)
# 对预测结果进行评价
print('Accuracy of LR Classifier:', lr.score(x_test, y_test))
print(skmt.classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print('Accuracy of SGD Classifier:', sgdc.score(x_test, y_test))
print(skmt.classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

# 支持向量机
# Handwriting Digits
digits = sds.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
ss = spr.StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
lsvc = svm.LinearSVC()
lsvc.fit(x_train, y_train)
lsvc_y_predict = lsvc.predict(x_test)
print('Accuracy of Linear SVC Classifier:', lsvc.score(x_test, y_test))
print(skmt.classification_report(y_test, lsvc_y_predict, target_names=digits.target_names.astype(str)))

# 朴素贝叶斯
# 新闻文本
news = sds.fetch_20newsgroups(subset='all')
x = news.data
y = news.target
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
vec = skft.CountVectorizer()    #向量化
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
mnb = skb.MultinomialNB()
mnb.fit(x_train, y_train)
mnb_y_predict = mnb.predict(x_test)
print('Accuracy of Naive Bayes Classifier:', mnb.score(x_test, y_test))
print(skmt.classification_report(y_test, mnb_y_predict, target_names=news.target_names))


# KNN
# iris data
iris = sds.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
ss = spr.StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
knc = skn.KNeighborsClassifier()    # 初始化KNN分类器
knc.fit(x_train, y_train)
knc_y_predict = knc.predict(x_test)
print('Accuracy of KNN Classifier:', knc.score(x_test, y_test))
print(skmt.classification_report(y_test, knc_y_predict, target_names=iris.target_names))

# 决策树分类, 集成模型
# titanic data
server = 'http://biostat.mc.vanderbilt.edu'
folder = '/wiki/pub/Main/DataSets'
file = '/titanic.txt'
source = server + folder + file
titanic = pd.read_csv(source)
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
x['age'].fillna(x['age'].mean(), inplace=True)    # 对空值进行填充， 有问题
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
vec = skfe.DictVectorizer(sparse=False)    # 向量化
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.fit_transform(x_test.to_dict(orient='record'))
dtc = skt.DecisionTreeClassifier()    # 初始化决策树分类器
dtc.fit(x_train, y_train)
dtc_y_predict = dtc.predict(x_test)
rfc = ske.RandomForestClassifier()    # 随机森林
rfc.fit(x_train, y_train)
rfc_y_predict = dtc.predict(x_test)
gbc = ske.GradientBoostingClassifier()    # 梯度提升决策树
gbc.fit(x_train, y_train)
gbc_y_predict = gbc.predict(x_test)
print('Accuracy of Decision Tree Classifier:', dtc.score(x_test, y_test))
print(skmt.classification_report(y_test, dtc_y_predict, target_names=['died', 'survived']))
print('Accuracy of Random Forest Classifier:', rfc.score(x_test, y_test))
print(skmt.classification_report(y_test, rfc_y_predict, target_names=['died', 'survived']))
print('Accuracy of Gradient Boosting Classifier:', rfc.score(x_test, y_test))
print(skmt.classification_report(y_test, rfc_y_predict, target_names=['died', 'survived']))

# 回归
# 线性回归和随机梯度回归
boston = sds.load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = skm.train_test_split(x, y, test_size=0.25, random_state=33)
ss_x = spr.StandardScaler()
# ss_y = spr.StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
# y_train = ss_y.fit_transform(y_train)
# y_test = ss_y.transform(y_test)
# 线性回归
lr = sklin.LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
# 随机梯度回归
sgdr = sklin.SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)
# 支持向量机回归
# 线性核函数
lsvr = svm.SVR(kernel='linear')
lsvr.fit(x_train, y_train)
lsvr_y_predict = lsvr.predict(x_test)
# 多项式核函数
psvr = svm.SVR(kernel='poly')
psvr.fit(x_train, y_train)
psvr_y_predict = psvr.predict(x_test)
# 径向基核函数
rbf_svr = svm.SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)

# K近邻回归
# 预测方式是平均回归
uni_knr = skn.KNeighborsRegressor(weights='uniform')
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)
# 预测方式是距离加权平均
dis_knr = skn.KNeighborsRegressor(weights='distance')
dis_knr.fit(x_train, y_train)
dis_knr_y_predict = dis_knr.predict(x_test)

# 回归树
dtr = skt.DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)

# 集成回归
# 随机森林回归
rfr = ske.RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)
# 极端随机森林回归
etr = ske.ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)
# 梯度渐近回归
gbr = ske.GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)

# 回归性能测评
print('The Defaut Measurement of Linear Regression:', lr.score(x_test, y_test))
print('The R-Squared of Linear Regression:', skmt.r2_score(y_test, lr_y_predict))
print('The MSE of Linear Regression:', skmt.mean_squared_error(y_test, lr_y_predict))
print('The MAE of Linear Regression:', skmt.mean_absolute_error(y_test, lr_y_predict))

print('The Defaut Measurement of SGD Regression:', sgdr.score(x_test, y_test))
print('The R-Squared of SGD Regression:', skmt.r2_score(y_test, sgdr_y_predict))
print('The MSE of SGD Regression:', skmt.mean_squared_error(y_test, sgdr_y_predict))
print('The MAE of SGD Regression:', skmt.mean_absolute_error(y_test, sgdr_y_predict))

print('The Defaut Measurement of LSVR Regression:', lsvr.score(x_test, y_test))
print('The R-Squared of LSVR Regression:', skmt.r2_score(y_test, lsvr_y_predict))
print('The MSE of LSVR Regression:', skmt.mean_squared_error(y_test, lsvr_y_predict))
print('The MAE of LSVR Regression:', skmt.mean_absolute_error(y_test, lsvr_y_predict))

print('The Defaut Measurement of PSVR Regression:', psvr.score(x_test, y_test))
print('The R-Squared of PSVR Regression:', skmt.r2_score(y_test, psvr_y_predict))
print('The MSE of PSVR Regression:', skmt.mean_squared_error(y_test, psvr_y_predict))
print('The MAE of PSVR Regression:', skmt.mean_absolute_error(y_test, psvr_y_predict))

print('The Defaut Measurement of RBFSVR Regression:', rbf_svr.score(x_test, y_test))
print('The R-Squared of RBFSVR Regression:', skmt.r2_score(y_test, rbf_svr_y_predict))
print('The MSE of RBFSVR Regression:', skmt.mean_squared_error(y_test, rbf_svr_y_predict))
print('The MAE of RBFSVR Regression:', skmt.mean_absolute_error(y_test, rbf_svr_y_predict))

print('The Defaut Measurement of Uniform-KNeighbor Regression:', uni_knr.score(x_test, y_test))
print('The R-Squared of Uniform-KNeighbor Regression:', skmt.r2_score(y_test, uni_knr_y_predict))
print('The MSE of Uniform-KNeighbor Regression:', skmt.mean_squared_error(y_test, uni_knr_y_predict))
print('The MAE of Uniform-KNeighbor Regression:', skmt.mean_absolute_error(y_test, uni_knr_y_predict))

print('The Defaut Measurement of Distance-KNeighbor Regression:', dis_knr.score(x_test, y_test))
print('The R-Squared of Distance-KNeighbor Regression:', skmt.r2_score(y_test, dis_knr_y_predict))
print('The MSE of Distance-KNeighbor Regression:', skmt.mean_squared_error(y_test, dis_knr_y_predict))
print('The MAE of Distance-KNeighbor Regression:', skmt.mean_absolute_error(y_test, dis_knr_y_predict))

print('The Defaut Measurement of Decision Tree Regression:', dtr.score(x_test, y_test))
print('The R-Squared of Decision Tree Regression:', skmt.r2_score(y_test, dtr_y_predict))
print('The MSE of Decision Tree Regression:', skmt.mean_squared_error(y_test, dtr_y_predict))
print('The MAE of Decision Tree Regression:', skmt.mean_absolute_error(y_test, dtr_y_predict))

print('The Defaut Measurement of Random Forest Regression:', rfr.score(x_test, y_test))
print('The R-Squared of Random Forest Regression:', skmt.r2_score(y_test, rfr_y_predict))
print('The MSE of Random Forest Regression:', skmt.mean_squared_error(y_test, rfr_y_predict))
print('The MAE of Random Forest Regression:', skmt.mean_absolute_error(y_test, rfr_y_predict))


print('The Defaut Measurement of Extra Tree Regression:', etr.score(x_test, y_test))
print('The R-Squared of Extra Tree Regression:', skmt.r2_score(y_test, etr_y_predict))
print('The MSE of Extra Tree Regression:', skmt.mean_squared_error(y_test, etr_y_predict))
print('The MAE of Extra Tree Regression:', skmt.mean_absolute_error(y_test, etr_y_predict))


print('The Defaut Measurement of Gradient Boosting Regression:', gbr.score(x_test, y_test))
print('The R-Squared of Gradient Boosting Regression:', skmt.r2_score(y_test, gbr_y_predict))
print('The MSE of Gradient Boosting Regression:', skmt.mean_squared_error(y_test, gbr_y_predict))
print('The MAE of Gradient Boosting Regression:', skmt.mean_absolute_error(y_test, gbr_y_predict))


# 无监督学习模型
# 聚类 - K均值法
server = 'https://archive.ics.uci.edu'
folder = '/ml/machine-learning-databases/optdigits'
file = '/optdigits.tra'
source = server + folder + file
digits_train = pd.read_csv(source, header=None)
file = '/optdigits.tes'
source = server + folder + file
digits_test = pd.read_csv(source, header=None)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = skc.KMeans(n_clusters=10)
kmeans.fit(x_train)
kmeans_y_pred = kmeans.predict(x_test)

print(skmt.adjusted_rand_score(y_test, kmeans_y_pred))

# 特征降维
# 主成分分析
estimator = skd.PCA(n_components=2)
x_pca = estimator.fit_transform(x_train)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    px = x_pca[:, 0][y_train.as_matrix()==i]
    py = x_pca[:, 1][y_train.as_matrix()==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(np.arange(0,10).astype(str))
plt.xlabel('First')
plt.ylabel('Second')
plt.show()

