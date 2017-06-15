# -*- coding: utf-8 -*-
"""
Spyder Editor

Python for Predictive
Penalty Linear Regression
Ensemble Methods

"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import sklearn.tree as skt
from sklearn.metrics import roc_curve, auc, confusion_matrix


server = 'http://archive.ics.uci.edu/ml/machine-learning-databases'

sonar_url = '/undocumented/connectionist-bench/sonar/sonar.all-data'
abalone_url = '/abalone/abalone.data'
wine_url = '/wine-quality/winequality-red.csv'
glass_url = '/glass/glass.data'
# 数据源
sonar_url = server + sonar_url
abalone_url = server + abalone_url
wine_url = server + wine_url
glass_url = server + glass_url


# 用Pandas直接把CSV数据读入DataFrame
# 数据源不提供header,用'V'作为前缀创建header
# df1 = pd.read_csv(sonar_url, header=None, prefix='V')
# df2 = pd.read_csv(abalone_url, header=None, prefix='V')
# 为每个字段header命名
# df2.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight',
#               'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
# df3 = pd.read_csv(glass_url, header=None, prefix='V')
# df3.columns = ['Id', 'Ri', 'Na', 'Mg', 'Al', 'Si',
#               'K', 'Ca', 'Ba', 'Fe', 'Type']
# 以数据源的第零行作为header,字段以';'作为分隔符
df4 = pd.read_csv(wine_url, header=0, sep=';')


# 除去非数值化的列, 行对应的axis=0, 列对应的axis=1, 原来的df1保持不变
# df = df1.drop(['V60'], axis=1)
# df = df2.drop(['Sex'], axis=1)
# df = df3
df = df4

c_list = list(df.columns)
c_length = len(df.columns)
nrows = len(df)
summary = df.describe()
m = summary.loc['mean']
s = summary.loc['std']

# 四分位间距归一化，也可以用sklearn的模块StandardScalar
for c in c_list:
    df[c] = (df[c] - m[c]) / s[c]
# 把DataFrame转换成Numpy的Array
array = df.values
plt.figure()
plt.boxplot(array)
plt.show()


# 相关系数可视化, 生成DataFrame格式的相关系数矩阵
# corr是DataFrame的一个函数
corMat = df.corr()
plt.figure()
plt.pcolor(corMat)
plt.show()


# 平行坐标图
plt.figure()
for i in range(nrows):
    dataRow = df.iloc[i, :]
    dataRow.plot()
plt.show()

"""
以Sonar的数据为例演示数据数据分析
"""
"""
# 建立属性xList和标签label
xList = df
labels = []

# 转换标签列到数值
for i in range(nrows):
    if df1.iloc[i, -1] == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
# 划分训练集和测试集

x_train, x_test, y_train, y_test = skm.train_test_split(xList, labels, test_size=0.25, random_state=33)
 
# 训练线性模型
sonarModel = skl.LinearRegression()
sonarModel.fit(x_train, y_train)
trainPrediction = sonarModel.predict(x_train)

# confusion matrix
# cnf_matrix = confusion_matrix(trainPrediction, y_train)
# 建立ROC curve模型性能评估
fpr, tpr, th = roc_curve(y_train, trainPrediction)   # 计算评估曲线
roc_auc = auc(fpr, tpr)   # 计算auc面积
# 展示评估图形
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve(area = %0.2f)' % roc_auc)   # 画ROC曲线
plt.plot([0, 1], [0, 1], 'k-')   # 画对角线
plt.show()
"""


"""
以wine的数据为例演示
"""

xList = df.drop(['quality'], axis=1)
labels = df[['quality']]

# 二元决策树
wineTree = skt.DecisionTreeRegressor(max_depth=8)    # 树的深度
wineTree.fit(xList, labels)
wine_dot = skt.export_graphviz(wineTree, out_file=None)
yHat = wineTree.predict(xList)
labels['yHat'] = yHat



    