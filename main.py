import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
# import openpyxl

ER_train = pd.read_excel('D:/MLProject/ER.xlsx', sheet_name='training')
ER_test = pd.read_excel('D:/MLProject/ER.xlsx', sheet_name='test')
Molecular_train = pd.read_excel('D:/MLProject/Molecular.xlsx', sheet_name='training')
Molecular_test = pd.read_excel('D:/MLProject/Molecular.xlsx', sheet_name='test')
ADMET_train = pd.read_excel('D:/MLProject/ADMET.xlsx', sheet_name='training')
ADMET_test = pd.read_excel('D:/MLProject/ADMET.xlsx', sheet_name='test')

# 1. 数据预处理
# 1.1 缺失值处理
ER_train_sum = ER_train.isnull().sum()
ER_test_sum = ER_test.isnull().sum()
Molecular_train_sum = Molecular_train.isnull().sum()
Molecular_test_sum = Molecular_test.isnull().sum()
ADMET_train_sum = ADMET_train.isnull().sum()
missing = Molecular_train.columns[Molecular_train.isnull().any()].tolist()
print('缺失值共%i条' % len(missing))
print('------')

# 1.2 数据筛选
print(len(list(Molecular_train.columns.values)))
neg_list = list(Molecular_train.drop(columns='SMILES').columns.values)
for item in neg_list:
    a = pd.DataFrame(Molecular_train[item].value_counts()).iloc[0]/len(Molecular_train)
    if a[0] > 0.73:
        Molecular_train = Molecular_train.drop(columns=item)
print(len(list(Molecular_train.columns.values)))
print('------')
data = Molecular_train.drop(columns=['SMILES'])

# 1.3 异常值处理
# 箱型图 可视化图像
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1)
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
data_1 = Molecular_train.iloc[:, 1]
data_1.plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
# data_2 = Molecular_train.iloc[:, 2]
# data_2.plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
# data_3 = Molecular_train.iloc[:, 3]
# data_3.plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
# data_4 = Molecular_train.iloc[:, 4]
# data_4.plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
# data_5 = Molecular_train.iloc[:, 2]
# data_5.plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
plt.show()

# for i in range(1, 6):
#     data[i] = Molecular_train.iloc[:, i]
#     data[i].plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
#     plt.show()

data_desc = data.describe()
print(data_desc)
print('------')
# 基本统计量(Molecular_train)

ER_desc = ER_train.drop(columns='SMILES').describe()
print(ER_desc)
print('------')
# 基本统计量(ER_train)

neg = list(data.columns.values)
for i in neg:
    q1 = data_desc[i].iloc[4]
    q3 = data_desc[i].iloc[6]
    EI = q3 - q1
    mi = q1 - 1.5 * EI
    ma = q3 + 1.5 * EI
    # 计算分位差

    error = data[i][np.array((data[i] < mi) | (data[i] > ma))]
    data[i][np.array((data[i] < mi) | (data[i] > ma))] = data[i].mean()
    print('异常值共%i条' % len(error))
    # data_new = data[i][np.array((data[i] < mi) | (data[i] > ma))]
    # print('剔除异常值之后的数据共%i条' % len(data_new))
    # print('------')
    # 筛选出异常值error,剔除异常值之后的数据data

# data_new = data[i][(data[i] >= mi) & (data[i] <= ma)]
# print('剔除异常值后的数据 \n', data_new)
# 筛选出异常值error,剔除异常值之后的数据data_new
print('------')

# 散点图 可视化图像
data_1_desc = data_1.describe()
q1 = data_1_desc.iloc[4]
q3 = data_1_desc.iloc[6]
EI = q3 - q1
mi = q1 - 1.5 * EI
ma = q3 + 1.5 * EI

error_1 = data_1[np.array((data_1 < mi) | (data_1 > ma))]
data_1_new = data_1[(data_1 >= mi) & (data_1 <= ma)]
plt.scatter(data_1_new.index, data_1_new, marker='.', alpha=0.3)
plt.scatter(error_1.index, error_1, color='r', marker='.', alpha=0.5)
plt.grid()

# data_2_desc = data_2.describe()
# q1 = data_2_desc.iloc[4]
# q3 = data_2_desc.iloc[6]
# EI = q3 - q1
# mi = q1 - 1.5 * EI
# ma = q3 + 1.5 * EI
#
# error_2 = data_2[np.array((data_2 < mi) | (data_2 > ma))]
# data_2_new = data_2[(data_2 >= mi) & (data_2 <= ma)]
# plt.scatter(data_2_new.index, data_2_new, marker='.', alpha=0.3)
# plt.scatter(error_2.index, error_2, color='r', marker='.', alpha=0.5)
# plt.grid()

# data_3_desc = data_3.describe()
# q1 = data_3_desc.iloc[4]
# q3 = data_3_desc.iloc[6]
# EI = q3 - q1
# mi = q1 - 1.5 * EI
# ma = q3 + 1.5 * EI
#
# error_3 = data_3[np.array((data_3 < mi) | (data_3 > ma))]
# data_3_new = data_3[(data_3 >= mi) & (data_3 <= ma)]
# plt.scatter(data_3_new.index, data_3_new, marker='.', alpha=0.3)
# plt.scatter(error_3.index, error_3, color='r', marker='.', alpha=0.5)
# plt.grid()

# data_4_desc = data_4.describe()
# q1 = data_4_desc.iloc[4]
# q3 = data_4_desc.iloc[6]
# EI = q3 - q1
# mi = q1 - 1.5 * EI
# ma = q3 + 1.5 * EI
#
# error_4 = data_4[np.array((data_4 < mi) | (data_4 > ma))]
# data_4_new = data_4[(data_4 >= mi) & (data_4 <= ma)]
# plt.scatter(data_4_new.index, data_4_new, marker='.', alpha=0.3)
# plt.scatter(error_4.index, error_4, color='r', marker='.', alpha=0.5)
# plt.grid()

# data_5_desc = data_5.describe()
# q1 = data_5_desc.iloc[4]
# q3 = data_5_desc.iloc[6]
# EI = q3 - q1
# mi = q1 - 1.5 * EI
# ma = q3 + 1.5 * EI
#
# error_5 = data_5[np.array((data_5 < mi) | (data_5 > ma))]
# data_5_new = data_5[(data_5 >= mi) & (data_5 <= ma)]
# plt.scatter(data_5_new.index, data_5_new, marker='.', alpha=0.3)
# plt.scatter(error_5.index, error_5, color='r', marker='.', alpha=0.5)
# plt.grid()

plt.show()

# 1.4 重复值处理
dupl = Molecular_train[Molecular_train.duplicated()]
print('重复值共%i条' % len(dupl))
print('------')
# ER_train.duplicated(subset='pIC50', keep='first')

# 2. 数据分布
Molecular_desc = data.describe()
print(Molecular_desc)
print('------')
# 基本统计量(Molecular)

ER_desc = ER_train.drop(columns='SMILES').describe()
print(ER_desc)
print('------')
# 基本统计量(ER)

ADMET_desc = ADMET_train.drop(columns='SMILES').describe()
print(ER_desc)
print('------')
# 基本统计量(ADMET)

# Molecular_df = pd.DataFrame(Molecular_desc)
# Molecular_df.to_excel("D:\\MLProject\\Excel\\molecular_desc.xlsx", encoding='utf-8', index=False)
# ER_df = pd.DataFrame(ER_desc)
# ER_df.to_excel("D:\\MLProject\\Excel\\er_desc.xlsx", encoding='utf-8', index=False)
# Molecular_df = pd.DataFrame(ADMET_desc)
# Molecular_df.to_excel("D:\\MLProject\\Excel\\admet_desc.xlsx", encoding='utf-8', index=False)
# Excel导出(Molecular_desc,ER_desc,A)

# 3. 特征选择:随机森林
neg_list = list(Molecular_train.drop(columns='SMILES').columns.values)
forest = RandomForestRegressor(n_estimators=100, random_state=0)
forest.fit(preprocessing.scale(data), ER_train.drop(columns='SMILES'))
y_pred = forest.predict(preprocessing.scale(Molecular_test[neg_list]))
print(sklearn.metrics.mean_squared_error(y_pred, ER_test.drop(columns='SMILES')))
print('------')

importances = forest.feature_importances_
features_list = Molecular_train.columns.values
sorted_idx = np.argsort(importances)
features_optimistic = features_list[sorted_idx[0:30]]
print(features_optimistic)
data_new = data[features_optimistic]
forest2 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=15)
forest2.fit(preprocessing.scale(data_new), ER_train.drop(columns='SMILES'))
y_pred2 = forest2.predict(preprocessing.scale(Molecular_test[features_optimistic]))
print(sklearn.metrics.mean_squared_error(y_pred2, ER_test.drop(columns='SMILES')))

plt.plot(range(1, 51), y_pred2, label='pred', linewidth=3)
plt.plot(range(1, 51), ER_test['pIC50'], label='test', linewidth=3)
plt.legend()
plt.show()
print('------')

# 4. MLP
Molecular_train = pd.read_excel('D:/MLProject/Molecular.xlsx', sheet_name='training')
Molecular_test = pd.read_excel('D:/MLProject/Molecular.xlsx', sheet_name='test')
ADMET_train = pd.read_excel('D:/MLProject/ADMET.xlsx', sheet_name='training')
ADMET_test = pd.read_excel('D:/MLProject/ADMET.xlsx', sheet_name='test')

Molecular_train = preprocessing.scale(Molecular_train.drop(columns='SMILES'))
Molecular_test = preprocessing.scale(Molecular_test.drop(columns='SMILES'))
scorelist = []

for i in range(1, 6):
    mlp = MLPClassifier(activation='logistic', learning_rate_init=0.2, hidden_layer_sizes=100, max_iter=350, random_state=3, solver='adam')
    mlp.fit(Molecular_train, ADMET_train.iloc[:, i])
    ADMET_test = mlp.predict(Molecular_test)
    scorelist.append(ADMET_test)

print(scorelist)
# 输出数组
print(np.average(np.array(scorelist)))
# 输出平均值
