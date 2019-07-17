
import numpy as np
from matplotlib import pyplot as plt

# 从sklearn生成聚类数据
from sklearn.datasets.samples_generator import make_blobs
# scipy距离函数，默认计算欧式距离
from scipy.spatial.distance import cdist


class K_Means(object):
    def __init__(self, n_cluster=6, max_iter=300, centroids=[]):
        # 初始化参数， 参数n_cluster(K)、迭代次数max_iter、初始质心centroids
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.centroids = np.array(centroids, dtype=np.float)
        
    def fit(self, data):
        # 训练模型的方法，传入原始数据
        
        # 如没有初始质心，随机选取data中的一个
        if len(self.centroids) == 0:
            self.centroids = data[np.random.randint(0, len(data), self.n_cluster), :]
            
        # 开始迭代
        for i in range(self.max_iter):
            # 1. 计算距离矩阵,得到100*6的矩阵
            distance = cdist(data, self.centroids)
            
            # 2. 对距离按由近到远排序，结果是100*1的一列
            c_index = np.argmin(distance, axis=1)
            
            # 3. 对每一类数据进行均值计算，更新质点坐标
            for j in range(self.n_cluster):
                # 排除掉没有出现在c_index里的类别
                if j in c_index:
                    # 选出所有类别是i的点，取data里面坐标的均值，更新第i个质心
                    self.centroids[j] = np.mean(data[c_index == j], axis=0)
            
    def predict(self, samples):
        # 预测方法
        # 计算距离矩阵，选取距离最近的质心的类别
        distance = cdist(samples, self.centroids)
        c_index = np.argmin(distance, axis=1)
        
        return c_index
    

def plotKmeans(x, y, centroids, subplot, title):
    # 分配子图
    plt.subplot(subplot)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    # 画出质心图
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array(range(6)), s=100)
    plt.title(title)
    
    
if __name__ == '__main__':
    # 100个样本点 按照6个中心点生成类别
    x, y = make_blobs(n_samples=100, centers=6, random_state=1234, cluster_std=0.6)
    
    # 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
    
    kmeans = K_Means(max_iter=300, centroids=np.array([[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6]]))
    
    plt.figure(figsize=(16, 6))
    plotKmeans(x, y, kmeans.centroids, 121, 'Initial State')
    
    # 开始聚类
    kmeans.fit(x)
    plotKmeans(x, y, kmeans.centroids, 122, 'Final State')
    
    # 预测新数据点的类别
    x_new = np.array([[0, 0], [10, 7]])
    y_pred = kmeans.predict(x_new)
    print(kmeans.centroids)
    print(y_pred)
    
    plt.scatter(x_new[:, 0], x_new[:, 1], s=100, c='black')
    plt.show()