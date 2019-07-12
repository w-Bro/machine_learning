
# k近邻算法实现

import numpy as np
import pandas as pd
# 引入sklearn里的数据集，iris鸢尾花
from sklearn.datasets import load_iris
# 切分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
# 计算分类预测的准确率
from sklearn.metrics import accuracy_score


def l1_distance(a, b):
    """
    曼哈顿距离
    :param a: a可以是矩阵也可以是向量
    :param b: b只能是向量
    :return:
    """
    # axis=1 a的每一行都减去b， 最后保存成1列，否则会加成一个数
    return np.sum(np.abs(a - b), axis=1)


def l2_distance(a, b):
    """
    欧式距离
    :param a:
    :param b:
    :return:
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


# 分类器实现
class KNN(object):
    def __init__(self, n_neighbours=1, dist_func=l1_distance):
        self.n_neighbours = n_neighbours
        self.dist_func = dist_func
        
    # 训练模型方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        
    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组, 指定数据类型
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 遍历数据点
        for i, x_test in enumerate(x):
            
            # x_test跟所有训练数据计算距离
            distances = self.dist_func(self.x_train, x_test)
            
            # 距离按照从近到远排序，取出索引值
            nn_index = np.argsort(distances)
            
            # 选取最近的k个点，保存他们对应的分类类别 ravel转为一维数组
            nn_y = self.y_train[nn_index[: self.n_neighbours]].ravel()
            
            # 统计类别中出现频率最高的值，赋给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))
            
        return y_pred


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['class'] = iris.target
    # 将class列0， 1， 2改为对应的类型名称
    df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
    # df的统计信息
    # print(df.describe())
    
    x = iris.data
    # 一维转二维
    y = iris.target.reshape(-1, 1)
    # x 150行4列 y 150行1列
    print(x.shape, y.shape)
    
    # 划分训练集和测试集 test_size测试集比例 random_state随机选择 stratify按照y进行等比例分配
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)
    
    print(x_train.shape, y_train.shape)
    
    knn = KNN(n_neighbours=3)
    # 训练模型
    knn.fit(x_train, y_train)
    
    # 结果list
    results = []
    for p in [1, 2]:
        knn.dist_func = l1_distance if p == 1 else l2_distance
        
        # 考虑不同的k取值，步长为2，尽量避免偶数
        for k in range(1, 10, 2):
            knn.n_neighbours = k
            
            # 测试数据做预测
            y_pred = knn.predict(x_test)
            # 求出预测准确率
            accuracy = accuracy_score(y_test, y_pred)
            # print('预测准确率：', accuracy)
            results.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])
            
    df = pd.DataFrame(results, columns=['k', '距离函数', '预测准确率'])
    print(df)