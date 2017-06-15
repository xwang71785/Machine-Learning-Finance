# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:18:23 2017
Bayes PyMC2
@author: wangx3
"""
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

count_data = np.loadtxt('txtdata.csv')   # 导入数据
n_count_data = len(count_data)   # 采集数据信息，行数


# 设置模型参数
alpha = 1.0 / count_data.mean()
# 设置随机变量
lambda_1 = pm.Exponential('lambda_1', alpha)   # 指数分布
lambda_2 = pm.Exponential('lambda_2', alpha)
tau = pm.DiscreteUniform('tau', lower=0, upper=n_count_data)    # 离散均匀分布
    
@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out
# 用样本数据作为定值来初始化一个随机变量，由此把样本数据加入到模型  
observation = pm.Poisson('obs', lambda_, value=count_data, observed=True)
# 把所有创建的随机变量都打包进Model类
model = pm.Model([observation, lambda_1, lambda_2, tau])

mcmc = pm.MCMC(model)
trace = mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

'''

'''

p = pm.Uniform('p', lower=0, upper=1)
p_true = 0.05
N = 2000
occurrences = pm.rbernoulli(p_true, N)   # 生成N个伯努利分布的随机数
print(occurrences)
print(occurrences.sum())

# 指明观察到的数据是符合伯努利分布的
obs = pm.Bernoulli('obs', p, value=occurrences, observed=True)   
# 用观察到的数据建立一个伯努利模型
mcmc = pm.MCMC([p, obs])
# 运行模型估计
mcmc.sample(20000, 1000)

plt.plot(figsize=(16, 9))
plt.vlines(p_true, 0, 90, linestyle='--')
# mcmc.trace('p')[:]来提取模型模拟出来的参数
plt.hist(mcmc.trace('p')[:], bins=35, normed=True)
plt.show()






