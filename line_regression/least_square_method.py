
import numpy as np
import matplotlib.pyplot as plt


# 最小二乘法
def least_square_method():
    # 导入数据
    points = np.genfromtxt('data.csv', delimiter=',')
    
    # 提取两列数据，分别作为x,y
    # 取第一行第一列points[0, 0] 取所有行的第一列 points[0 :, 0] 等价于 points[:, 0]
    x = points[:, 0]
    y = points[:, 1]

    # 用plt画出散点图
    plt.scatter(x, y)
    
    w, b = fit(points)
    
    print('w is:', w)
    print('b is:', b)
    
    cost = compute_cost(w, b, points)
    
    print('cost is:', cost)
    
    # 针对每一个x，计算出预测的y值
    pred_y = w * x + b
    
    # c是颜色
    plt.plot(x, pred_y, c='r')
    plt.show()
    

def compute_cost(w, b, points):
    """
    损失函数(是系数的函数)
    https://i.loli.net/2019/07/11/5d26a3c6594ba82985.jpg
    :param w: 参数
    :param b: 参数
    :param points: 数据
    :return:
    """
    total_cost = 0
    M = len(points)
    
    # 逐点计算平方损失误差，求平均值
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2
    
    return total_cost / M


def average(data):
    """
    求均值
    :param data:
    :return:
    """
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num


def fit(points):
    """
    核心拟合函数
    :param points:
    :return:
    """
    M = len(points)
    x_bar = average(points[:, 0])
    
    sum_yx = 0
    sum_x2 = 0
    
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
        
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))
    
    sum_delta = 0
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)
    b = sum_delta / M
    
    return w, b


if __name__ == '__main__':
    
    least_square_method()