import sys
import numpy as np

class SVD(object):
    """
    矩阵分解实现
    Example:
            >>> lam = 0.0001
            >>> alpha = 0.0001
            >>> step = 100
            >>> k = 20
            >>> data_file = sys.argv[1]
            >>> vector_file = sys.argv[2]
            >>> svd = SVD(lam, alpha, k)
            >>> svd.train(data_file, step)
            >>> svd.save(vector_file)
    """

    def __init__(self, lam = 0.01, alpha = 0.01, k = 2):
        """
        初始化
        Args:
            lam: 正则化参数lambda
            alpha: 学习率
            k: 隐因子个数
        """
        self._item_feature = {}
        self._user_feature = {}
        self._factors = k
        self._lambda = lam
        self._alpha = alpha

    def train(self, data_file, step = 100):
        """
        读入coo格式矩阵，每一行为一个非零元素：行，列，评分
        Args:
            data_file: 矩阵数据文件
            step: 迭代步数
        Returns:
            无
        """
        for i in range(step):
            with open(data_file, 'r') as matrix_train:
                for line in matrix_train:
                    u, i, r = line.strip().split()
                    # 初始化用户隐因子向量
                    if u not in self._user_feature:
                        self._user_feature[u] = np.random.rand(self._factors)
                    # 初始化物品隐因子向量
                    if i not in self._item_feature:
                        self._item_feature[i] = np.random.rand(self._factors)
                    # 计算预测分数
                    predict = np.dot(self._user_feature[u], self._item_feature[i])
                    # 计算误差
                    error = float(r) - predict
                    # 更新用户的隐因子向量（注意这里采用了向量化计算）
                    self._user_feature[u] = (self._user_feature[u]
                                            + self._alpha * (
                                                error * self._item_feature[i]
                                                - self._lambda * self._user_feature[u]
                                            ))
                    # 更新物品的隐因子向量
                    self._item_feature[i] = (self._item_feature[i]
                                            + self._alpha * (
                                                error * self._user_feature[u]
                                                - self._lambda * self._item_feature[i]
                                            ))
    def save(self, file_name):
        with open("%s_user.vec" % file_name, 'w') as factor_output:
            for u in self._user_feature:
                vec = ','.join([str(v) for v in self._user_feature[u]])
                factor_output.write("%s %s\n" % (u, vec))
        with open("%s_item.vec" % file_name, 'w') as factor_output:
            for i in self._item_feature:
                vec = ','.join([str(v) for v in self._item_feature[i]])
                factor_output.write("%s %s\n" % (i, vec))


if __name__ == "__main__":
    lam = 0.0001
    alpha = 0.0001
    step = 100
    k = 20
    data_file = sys.argv[1]
    vector_file = sys.argv[2]
    svd = SVD(lam, alpha, k)
    svd.train(data_file, step)
    svd.save(vector_file)
