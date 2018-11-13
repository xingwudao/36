import sys

def get_distance(user_lists1, user_lists2):
    """
    计算两个稀疏向量的平均距离和共同元素数
    Args:
        user_lists1: 向量1，[(id, value)] 
        user_lists2: 向量2，[(id, value)] 
    Returns:
        distance, common
    """

    list1 = sorted(user_lists1, key = lambda x: x[0])
    list2 = sorted(user_lists2, key = lambda x: x[0])
    index1 = 0
    index2 = 0
    distance = 0.0
    common = 0
    while index1 < len(list1) and index2 < len(list2):
        if list1[index1][0] < list2[index2][0]:
            index1 += 1
        elif list1[index1][0] > list2[index2][0]:
            index2 += 1
        else:
            distance += (list1[index1][1] - list2[index2][1])
            common += 1
            index1 += 1
            index2 += 1
    return distance/common, common

class SlopeOne(object):
    """
    slopeone算法实现：
    1. 首先读入COO矩阵
    2. 然后计算Item之间的平均距离
    3. 计算推荐结果
    
    >>> recommender = SlopeOne(data_file_name)
    >>> itemlists = recommender.recommend(1)
    >>> print(itemlists)
    """

    def __init__(self, data_file_name):
        """
        初始化算法实例，读入COO文件
        args:
            data_file_name: coo格式矩阵，行是用户，列是物品，元素是评分
        """
        self._user_records = {}
        self._item_records = {}
        self._distance = {}
        with open(data_file_name, 'r') as data_file:
            for line in data_file:
                fields = line.strip().split()
                if len(fields) < 3:
                    continue
                user = int(fields[0])
                item = int(fields[1])
                value = float(fields[2])
                if user not in self._user_records:
                    self._user_records[user] = []
                if item not in self._item_records:
                    self._item_records[item] = []
                self._user_records[user].append((item, value))
                self._item_records[item].append((user, value))
        self._compute_distance_of_items()

    def _compute_distance_of_items(self):
        item_count = len(self._item_records)
        items = sorted(list(self._item_records.keys()))
        item_count = len(items)
        for item1 in range(item_count):
            for item2 in range(item1+1, item_count):
                item_id1 = items[item1]
                item_id2 = items[item2]
                distance, common = get_distance(
                    self._item_records[item_id1], 
                    self._item_records[item_id2])
                if item_id1 not in self._distance:
                    self._distance[item_id1] = {}
                self._distance[item_id1][item_id2] = (distance, common)
   #     for item1 in self._distance:
   #         for item2 in self._distance[item1]:
   #             print("%d %d %0.2f(%d)" % (item1, item2,
   #                                        self._distance[item1][item2][0],
   #                                        self._distance[item1][item2][1]))
        
    def recommend(self, user_id, top = -1):
        """
        为输入用户计算推荐结果：
        计算用暴力办法， 遍历所有物品，逐一用slopeone算法计算推荐分数
        Args:
            user_id: 用户ID
            top: 输出推荐结果个数，默认全部
        returns:
            [(item, score)]
        """

        if user_id not in self._user_records:
            return []
        
        recommended = []
        for item1 in self._item_records:
            sum1 = 0.0
            sum2 = 0.0;
            for item2 in self._user_records[user_id]:
                if item1 == item2[0]:
                    continue
                value = item2[1]
                min_item_id = item1 if item1 < item2[0] else item2[0]
                max_item_id = item1 if item1 > item2[0] else item2[0]
                distance, common = self._distance[min_item_id][max_item_id]
                direction = 1
                if min_item_id != item1:
                    direction = -1
                sum1 += (value + distance) * direction * common 
                sum2 += common
            recommended.append((item1, sum1/sum2))
        
        return sorted(recommended, key = lambda x: x[1], reverse = True)[:top]

if __name__ == "__main__":
    recommender = SlopeOne(sys.argv[1])
    userid = 1
    itemlists = recommender.recommend(userid)
    for item in itemlists:
        print(item)
