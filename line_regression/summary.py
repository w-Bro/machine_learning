
# 使用机器学习库调用线性回归方法
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    lr = LinearRegression()

    points = np.genfromtxt('data.csv', delimiter=',')

    x = points[:, 0]
    y = points[:, 1]

    # 用plt画出散点图
    plt.scatter(x, y)
    
    # 将x, y由一维变为二维矩阵
    x_new = x.reshape(-1, 1)
    y_new = y.reshape(-1, 1)
    
    # 拟合
    lr.fit(x_new, y_new)
    
    # 从训练好的模型中提取系数和截距
    # 系数
    w = lr.coef_[0][0]
    # 截距
    b = lr.intercept_[0]
    
    print('w is:', w)
    print('b is:', b)

    # 针对每一个x，计算出预测的y值
    pred_y = w * x + b

    # c是颜色
    plt.plot(x, pred_y, c='r')
    plt.show()