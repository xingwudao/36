import numpy as np
import sys
import random

"""
输入：
    1. 事件名称和分数的列表
        [('1', 0.6), ('0', 0.1)]
    2. 作为正样本看待的事件名称
输出：
    AUC
"""
def compute_auc(event_with_score, positive = '1'):
   # 按照分数降序排序
   sorted_samples = sorted(event_with_score,
                           key = lambda x : x[1], reverse = True)
   # 给每一个样本增加一个排序值，第一位是n，第二位是n-1，以此类推
   # 同时计算正样本的个数M和负样本个数N
   rank = len(sorted_samples)
   M = 0
   N = 0
   positive_samples_rank = []
   last_score = None 
   same_score_ranks = []
   same_score_is_positive = 0
   for sample in sorted_samples:
       if last_score is not None:
           if last_score != sample[1]:
               same_score_ranks = np.full(same_score_is_positive,
                                          np.mean(same_score_ranks))
               positive_samples_rank.extend(same_score_ranks)
               same_score_ranks = []
               same_score_is_positive = 0
       last_score = sample[1]
       same_score_ranks.append(rank)
       if sample[0] == positive:
           M += 1
           same_score_is_positive += 1
       else:
           N += 1
       rank -= 1
   if len(same_score_ranks) > 0:
       same_score_ranks = np.full(same_score_is_positive,
                                  np.mean(same_score_ranks))
       positive_samples_rank.extend(same_score_ranks)

   # 计算AUC值（正样本分数大于负样本分数的概率）
   auc = np.sum(positive_samples_rank) - M * (M + 1)/2
   if N == 0 or M == 0:
       return 0.5
   return auc / (M * N)
