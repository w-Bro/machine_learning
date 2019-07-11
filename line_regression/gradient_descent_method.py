
import numpy as np
import matplotlib.pyplot as plt


# 梯度下降法
def gradient_descent_method():
    # 导入数据
    points = np.genfromtxt('data.csv', delimiter=',')

    # 模型的超参数，可以调整
    # 步长
    alpha = 0.0001
    # 起始点
    initial_w = 0
    initial_b = 0
    # 迭代次数
    num_iter = 10
    
    w, b, cost_list = grad_desc(points, initial_w, initial_b, alpha, num_iter)
    
    print('w is:', w)
    print('b is:', b)

    cost = compute_cost(w, b, points)

    print('cost is:', cost)
    
    # 以list的下标作为x
    plt.plot(cost_list)
    plt.show()

    x = points[:, 0]
    y = points[:, 1]

    # 用plt画出散点图
    plt.scatter(x, y)
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


def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    """
    梯度下降算法函数
    :param points:
    :param initial_w:
    :param initial_b:
    :param alpha:
    :param num_iter:
    :return:
    """
    w = initial_w
    b = initial_b
    # 保存所有的损失函数值，用来显示下降的过程
    cost_list = []
    
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, points))
        w, b = step_grad_desc(w, b, alpha, points)
    
    return [w, b, cost_list]


def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)
    
    # 每个点代入公式求和
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y
    
    # 用公式求当前梯度
    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b
    
    # 梯度下降，更新当前的w和b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b
    
    return updated_w, updated_b


if __name__ == '__main__':
    gradient_descent_method()