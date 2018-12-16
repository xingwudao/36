import numpy as np
import networkx as nx
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc

class Item(object):
    """
    Item 类，两个元素：
    _id -- Item ID
    _feature -- 特征向量
    """

    def __init__(self, item_id, feature, priority):
        self._id = item_id
        self._feature = feature
        self._priority = priority

    def pull(self):
        return int(np.random.rand() < self._priority)

    @property
    def id(self):
        return self._id

    @property
    def x(self):
        return self._feature

    @property
    def feature(self):
        return self._feature

class User(object):
    """
    User 类， 单个用户的诸多参数，同LinUCB中的参数
    """
    def __init__(self, d, lambda_, uid):
        '''
        personal parameters:
        '''
        self._d = d
        self._reward = 0
        self._A = lambda_*np.identity(n = self._d)
        self._b = np.zeros(self._d)
        self._AI = np.linalg.inv(self._A)
        self._theta = np.zeros(self._d)
        self._I = lambda_ * np.identity(n = d)
        '''
        cluster parameters:
        '''
        self._avg_A = self._A
        self._avg_b = self._b
        self._avg_AI = np.linalg.inv(self._avg_A)
        self._avg_theta = np.dot(self._avg_AI, self._avg_b)

    def update(self, x, reward, alpha_2):
        self._A += np.outer(x,x)
        self._b += x * reward
        self._AI = np.linalg.inv(self._A)
        self._theta = np.dot(self._AI, self._b)

    def update_cluster_avg_parameters(self, clusters, uid, graph, users):
        self._avg_A = self._I
        self._avg_b = np.zeros(self._d)
        for i in range(len(clusters)):
            if clusters[i] == clusters[uid]:
                self._avg_A += (users[i]._A - self._I)
                self._avg_b += users[i]._b
        self._avg_AI = np.linalg.inv(self._avg_A)
        self._avg_theta = np.dot(self._avg_AI, self._avg_b)

    def predict(self, alpha, x, trials):
        expected_reward = np.dot(self._avg_theta, x)
        bound = np.sqrt(np.dot(np.dot(x, self._avg_AI),  x))
        ucb = expected_reward + alpha * bound * np.sqrt(math.log10(trials + 1))
        return ucb

class Cofiba(object):
    def __init__(self, d, alpha, alpha_2, lambda_, user_count, item_count):
        self._trials = 0
        self._d = d
        self._alpha = alpha
        self._alpha_2 = alpha_2
        self._item_count = item_count
        self._user_count = user_count
        self._users = [User(d, lambda_, i) for i in range(user_count)]
        self._item_graph = np.ones([self._item_count, self._item_count])
        cluster_count, item_clusters = cc(csr_matrix(self._item_graph))
        self._cluster_count_of_item = cluster_count
        self._item_clusters = item_clusters
        self._user_graph = []
        self._user_clusters = []
        for i in range(self._cluster_count_of_item):
            self._user_graph.append(np.ones([user_count, user_count]))
            _, user_clusters = cc(csr_matrix(self._user_graph[i]))
            self._user_clusters.append(user_clusters)
  
    @property
    def alpha(self):
        return self._alpha

    @property
    def trials(self):
        return self._trials

    @property
    def alpha_2(self):
        return self._alpha_2

    @property
    def users(self):
        return self._users

    def get_item_cluster(self, item):
        return self._item_clusters[item]

    def select(self, items, uid):
        max_exptected_reward = -1
        item_selected = None
        for item in items:
            item_cluster_index = self._item_clusters[item._id]
            self.update_user_clusters(uid, item._feature, item_cluster_index)
            user = self._users[uid]
            user_cluster = self._user_clusters[item_cluster_index]
            user.update_cluster_avg_parameters(user_cluster,
                                               uid,
                                               self._user_graph,
                                               self._users)
            ucb_prob = self._users[uid].predict(self.alpha,
                                                item._feature,
                                                self._trials)
            if max_exptected_reward < ucb_prob:
                item_selected = item.id
                feature = item._feature
                max_exptected_reward = ucb_prob
        self._trials += 1
        return item_selected

    def update_user_clusters(self, uid, feature, item_cluster_index):
        n = len(self._users)
        user_expected_reward = np.dot(self._users[uid]._theta, feature)
        user_cb = np.sqrt(np.dot(np.dot(feature, self._users[uid]._AI),  feature)) 
        trials_alpha = np.sqrt(np.log10(self._trials + 1))
        for j in range(n):
            theta = self._users[j]._theta
            AI = self._users[j]._AI
            expected_reward = np.dot(theta, feature)
            user_cb = np.sqrt(np.dot(np.dot(feature, AI), feature))
            center_distance = math.fabs(user_expected_reward - expected_reward)
            bounds = self.alpha_2 * (user_cb + user_cb) * trials_alpha
            if center_distance > bounds:
                self._user_raph[item_cluster_index][uid][j] = 0
                self._user_graph[item_cluster_index][j][uid] = 0

        user_graph = csr_matrix(self._user_graph[item_cluster_index])
        cluster_count, user_clusters = cc(user_graph)
        self._user_clusters[item_cluster_index] = user_clusters
        return cluster_count

    def update_item_clusters(self, uid, item_selected, items):
        m = self._item_count
        n = self._user_count
        user_neighbor = {}
        item_cluster_index = self.get_item_cluster(item_selected)
        trials_alpha = np.sqrt(np.log10(self._trials + 1))
        AI1 = self._users[uid]._AI
        for item in items:
            if self._item_graph[items[item_selected].id][item._id] == 1:
                user_neighbor[item._id] = np.ones([n,n])
                for i in range(n):
                    theta1 = self._users[uid]._theta
                    feature = item._feature
                    theta2 = self._users[i]._theta
                    center_distance = math.fabs(np.dot(theta1, feature)
                                     - np.dot(theta2, feature))
                    AI2 = self._users[i]._AI
                    expected_reward1 = np.sqrt(np.dot(np.dot(feature, AI1), feature))
                    expected_reward2 = np.sqrt(np.dot(np.dot(feature, AI2), feature))
                    bounds = self.alpha_2 * (expected_reward1 + expected_reward2)
                    bounds *= trials_alpha
                    if center_distance > bounds:
                        user_neighbor[item._id][uid][i] = 0
                        user_neighbor[item._id][i][uid] = 0
                if not np.array_equal(user_neighbor[item._id],
                                      self._user_graph[item_cluster_index]):
                    self._item_graph[item_selected._id][item._id] = 0
                    self._item_graph[item._id][item_selected._id] = 0
        item_graph = csr_matrix(self._item_graph)
        self._cluster_count_of_item, self._item_clusters = cc(item_graph)

        self._user_graph = []
        self._user_clusters = []
        n = self._user_count
        for i in range(self._cluster_count_of_item):
            self._user_graph.append(np.ones([n, n]))
            cluster_count, user_clusters = cc(csr_matrix(self._user_graph[i]))
            self._user_clusters.append(user_clusters)
        return self._cluster_count_of_item

if __name__ == '__main__':
    lambda_ = 0.1
    alpha  = 0.2
    d = 5
    alpha_2 = 2.0
    item_count = 100
    user_count = 100
    items = [Item(i, np.random.rand(d), np.random.rand())
                for i in range(item_count)]
    cofiba = Cofiba(d = d,
                    alpha = alpha,
                    alpha_2 = alpha_2,
                    lambda_ = lambda_,
                    user_count = user_count,
                    item_count = item_count)
    uid = 5

    item_selected = cofiba.select(items, uid)
    reward = items[item_selected].pull()
    cofiba.users[uid].update(items[item_selected].feature,
                             reward,
                             alpha_2)
    cofiba.update_user_clusters(uid,
                                items[item_selected].feature,
                                cofiba.get_item_cluster(item_selected))
    cofiba.update_item_clusters(uid,
                                item_selected,
                                items)
