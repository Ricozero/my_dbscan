"""
参考代码：https://github.com/chrisjmccormick/dbscan/blob/master/dbscan.py
"""

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_digits

def DBSCAN(X, eps, min_pts):
    """
    对X进行DBSCAN聚类，-1是噪声，簇从0开始编号。
    """
    labels = [0] * len(X) # 用0表示未标记
    c = 0 # 当前簇编号

    # 寻找新的核心点，并进行簇生长，从1开始编号
    for p in range(len(X)):
        if not (labels[p] == 0):
           continue
        neighbors = regionQuery(X, p, eps)
        if len(neighbors) < min_pts:
            labels[p] = -1 # -1表示噪声
        else:
           c += 1
           growCluster(X, labels, p, neighbors, c, eps, min_pts)

    # 调整为从0开始编号
    for i in range(len(labels)):
        if labels[i] > 0:
            labels[i] -= 1

    return labels


def growCluster(X, labels, p, neighbors, c, eps, min_pts):
    """
    从点p生长簇c。
    """
    labels[p] = c

    # 把neighbors作为队列进行搜索，寻找连通的核心点
    i = 0
    while i < len(neighbors):
        q = neighbors[i]
        if labels[q] == -1:
           labels[q] = c
        elif labels[q] == 0:
            labels[q] = c
            q_neighbors = regionQuery(X, q, eps)
            if len(q_neighbors) >= min_pts:
                neighbors = neighbors + q_neighbors
        i += 1

def regionQuery(X, p, eps):
    """
    寻找点p在eps半径内的点。
    """
    neighbors = []
    for j in range(0, len(X)):
        if np.linalg.norm(X[p] - X[j]) < eps:
           neighbors.append(j)
    return neighbors

data_no = 102
if data_no == 1:
    X, y = make_blobs(300, centers=[[-5, -7], [5, 5], [-4, 3], [1, 0], [8, -3]])
elif data_no == 2:
    X1, y1 = make_blobs(300, centers=[[-5, -7], [5, 5], [-4, 3]])
    X2, y2 = make_blobs(300, centers=[[1, -1]], cluster_std=1.5)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2+y1.max()+1))
elif data_no == 3:
    X1, y1 = make_blobs(300, centers=[[-5, -7], [5, 5], [-4, 3]])
    X2, y2 = make_blobs(50, centers=[[1, -1]], cluster_std=2)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2+y1.max()+1))
elif data_no == 101:
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    labels = DBSCAN(X, 0.5, 4)
elif data_no == 102:
    digits = load_digits(4)
    X = digits["data"]
    X = X / X.max()
    y = digits["target"]
    labels = DBSCAN(X, 1.2, 6)

if data_no < 100:
    labels = DBSCAN(X, 1, 4)
    labels = np.array(labels)

    plt.subplot(121)
    plt.title('Original Dataset')
    for i in range(y.max() + 1):
        plt.scatter(X[np.where(y==i), 0], X[np.where(y==i), 1])

    plt.subplot(122)
    plt.title('Clustering Result')
    plt.scatter(X[np.where(labels==-1), 0], X[np.where(labels==-1), 1], c='#000')
    for i in range(labels.max() + 1):
        plt.scatter(X[np.where(labels==i), 0], X[np.where(labels==i), 1])

    plt.show()
else:
    labels = np.array(labels)
    m = np.zeros((y.max() + 1, labels.max() + 1), np.int)
    for i in range(y.size):
        if labels[i] == -1:
            continue
        m[y[i], labels[i]] += 1
    print(sum(labels == -1))
    print(m)