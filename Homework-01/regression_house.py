import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #注释：内置的拆分数据函数
from sklearn.linear_model import LinearRegression #注释：线性回归模型
from sklearn.metrics import mean_squared_error , r2_score #注释：回归预测评估函数，MSE，R2
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 通过数据集的源地址读取Boston房价数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
print(raw_df.head())
# 数据分为多行显示，每一行包含不完整的记录, 需要合并两行才能形成完整的记录。如下图。
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# 注释：raw_df.values,即将DataFrame转变为numpy
# 注释：raw_df.values[::2,:]指选取raw_df数组中所有偶数行的所有列
# 注释：raw_df.values[::2,:2]指选取raw_df数组中所有奇数行，前两行
# 注释：np.hstack(),为numpy中水平堆叠函数，将上述两个数组进行拼接，列方向
target = raw_df.values[1::2, 2]
# 注释：选取raw_df数组中奇数行的第三列，索引为2，将目标值存储在target中
# 拼接特征data和预测目标target
complete_data = np.column_stack([data, target])
# 注释：numpy中用于拼接的函数，拼接后，data在前面，target在最后
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 创建DataFrame，方便数据分析和特征处理
boston = pd.DataFrame(complete_data, columns=columns)
# 注释：将numpy类型的complete_data转变为DataFrame类型，并添加列名
# print(boston.head())

#查看数据字段，数据类型等
print(boston.info())
#描述性统计
print(boston.describe())
#对每个字段画直方图
boston.hist(bins=20, figsize=(20,15))
#注释：bins 参数指定了直方图的区间数量即20，bins 值越大，区间越窄，直方图越能反映数据的细节
plt.show()
# 计算相关矩阵
# corr()是 Pandas 库中的一个方法，用于计算数据框（DataFrame）中各列之间的相关系数矩阵。
# 相关系数用于衡量两个变量之间的线性关系，其值介于 -1 和 1 之间
correlation_matrix = boston.corr()
# 可视化相关矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
#注释：heatmap是绘制热力图，annot=True指会在热力图的方格子上显示数值
# fmt=".2f"指的是小数点位数，cmap指颜色映射，coolwarm指红蓝表示数值，蓝表示负，红表示正
plt.show()
#查看与目标变量相关性最高的特征变量
correlation_with_target=correlation_matrix['MEDV'].sort_values(ascending=False)
#注释：对MEDV列进行降序排列
print(correlation_with_target)
#分析房间数RM与房价的关系
x1=boston['RM']
y1=boston['MEDV']
plt.scatter(x1,y1)
plt.xlabel('平均房间数RM')
plt.ylabel('房价中位数MEDV')
plt.title('RM vs MEDV')
plt.show()
#分析人口中地位较低人群的百分比LSTATRM与房价的关系
x2=boston['LSTAT']
y2=boston['MEDV']
plt.scatter(x2,y2)
plt.xlabel('地位较低人群的百分比LSTATRM')
plt.ylabel('房价中位数MEDV')
plt.title('LSTAT vs MEDV')
plt.show()
boston.boxplot(column=['RM'])
plt.show()
boston.loc[boston['RM']>8,'RM']=8 #!!
#注释：将RM列数值超过8的替换为8
##线性模型
#拆分特征和目标变量
x= boston.drop(columns='MEDV',axis=1)
y= boston['MEDV']
#数据分割，拆分出模型训练集和测试数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
#注释：将数据集分为训练集和测试集，
#注释：test_size=0.1即测试集数据占比为10%，训练集为90%，random_state=42即用于控制数据分割的随机性
#创建模型
model=LinearRegression()
#调用训练模型参数，传入训练数据集
model.fit(x_train,y_train)
#模型预测，传入测试数据集，输出预测值
y_pred=model.predict(x_test)
# 通过预测值和测试值来计算均方误差
mse = mean_squared_error(y_test,y_pred)
# 通过预测值和测试值来计算决定系数
r2=r2_score(y_test,y_pred)
# print(f"均方误差 (MSE): {np.sqrt(mse)}")
# print(f"决定系数 (R^2): {r2}")
#画散点图判断预测值与测试值偏差
plt.scatter(y_test,y_pred)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('实际价格 vs 预测价格')
#辅助线
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()
