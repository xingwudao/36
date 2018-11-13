#!/usr/bin/python3.5
# -*- coding:utf-8 -*-

import math
from gensim.corpora.dictionary import Dictionary

def compute_entropy(total_samples, category_distribution):
    system_entropy = 0.0
    for cate in category_distribution:
        prob_cate = category_distribution[cate]/len(total_samples)
        system_entropy += -1.0 * prob_cate * math.log(prob_cate)
    return system_entropy


class InfoGain(object):
    """
    计算标签的信息增益，输入的是标注了类别的语料库，
    计算信息增益首先计算全局的信息熵，然后计算每个标
    签的条件熵，相减就是信息增益。
    Example：
        >>> ig = InfoGain(corpus_file)
        >>> ig.compute()
        >>> ig.save(ig_file)
        >>> print(ig['word'])  # 查询一个词的信息增益
    """
    def __init__(self, corpus_file):
        """
        Args:
            corpus_file -- 语料文件，第一列是类别，后面都是标签
        """
        corpus = []
        categories = []
        self._category_distribution = {}  # 统计各个类别的样本数
        self._words_cate = {}  # 统计每个词（标签、特征）下的类别样本数
        self._words_sample_count = {}
        self._info_gain = {}
        with open(corpus_file, 'r') as documents:
            for line in documents:
                words = line.strip().split()
                if len(words) <= 1:
                    continue
                categories.append(words[0])
                corpus.append(words[1:])
                if words[0] not in self._category_distribution:
                    self._category_distribution[words[0]] = 0
                self._category_distribution[words[0]] += 1

                # 统计词（标签、特征）和类别的共现次数，用于计算条件熵
                for word in set(words[1:]):
                    if word not in self._words_cate:
                        self._words_cate[word] = {}
                        self._words_sample_count[word] = 0
                    if words[0] not in self._words_cate[word]:
                        self._words_cate[word][words[0]] = 0
                    self._words_cate[word][words[0]] += 1
                    self._words_sample_count[word] += 1

        self._common_dictionary = Dictionary(corpus)
        self._corpus = corpus
        self._categories = categories

    def compute(self):
        """
        计算所有词（标签、特征）的信息增益。首先计算全局的信息熵。
        """
        system_entropy = compute_entropy(len(self._corpus),
                                         self._category_distribution)
        # 计算每个词的条件熵
        for word in self._common_dictionary.keys():
            category_distribution = {}
            if word not in self._words_cate:
                continue
            # 出现该词（标签、特征）的类别分布信息熵
            entropy1 = compute_entropy(self._words_sample_count[word],
                                       self._words_cate[word])
            for cate in self._category_distribution:
                category_distribution[cate] = self._category_distribution[cate]
                if cate in self._words_cate[word]:
                    category_distribution[cate] -= self._words_cate[word]
            # 未出现该词（标签、特征）的类别分布信息熵
            entropy2 = (compute_entropy(len(self._corpus)
                                        - self._words_sample_count[word],
                                        category_distribution))
            # 该词（标签、特征）的条件熵
            condition_entropy = (self._words_sample_count[word] * entropy1/len(self._corpus)
                                 + (len(self._corpus) - self._words_sample_count[word])
                                 * entropy2/len(self._corpus))
            # 信息增益
            info_gain = system_entropy - condition_entropy
            self._info_gain[word] = info_gain


    def save(self, ig_file_name, sort=False):
        """
        保存到文件，格式为：词 信息增益
        Args:
            ig_file_name -- 文件路径
            sort -- 是否按照信息增益从高到低排序后输出，默认不排序
        """
        with open(ig_file_name, 'w') as ig_file:
            if not sort:
                for word in self._info_gain:
                    ig_file.write("%s %.2f\n" % (word, self._info_gain[word]))
            else:
                for item in sorted(self._info_gain.items(), key=lambda x: x[1], reverse=True):
                    ig_file.write("%s %.2f\n" % (item[0], item[1]))

    def __get_item__(self, word):
        if word not in self._info_gain:
            return 0.0
        return self._info_gain[word]

class CHI(object):
    """
    卡方校验为每个类别选择特征。计算公式为：

    $$ \chi^{2}(W_{i}, C_{j}) = \frac{N(A(N + A - Q - M)-(M-A)(Q-A))^{2}}{MQ(N-M)(N-Q)} $$

    A：类别 $C_{j}$中包含词 $W_{i}$的数量；
    M: 类别 $C_{j}$ 的语料数量，也就是A+C的值；
    Q：包含词 $W_{i}$的语料数量，也就是A+B；
    N：全部语料数量

    Example：
        >>> chi = CHI(corpus_file)
        >>> chi.compute()
        >>> chi.save(corpus_file)
    """

    def __init__(self, corpus_file):
        """
        Args:
            corpus_file -- 语料文件，第一列是类别，后面都是标签
        """
        self._A = {} # 标签和类别共现统计
        self._M = {} # 类别统计量
        self._Q = {} # 标签统计量
        self._N = 0  #所有语料数量
        self._chi = {} # 卡方值计算结果

        with open(corpus_file, 'r') as corpus:
            for line in corpus:
                sentence = line.strip().split()
                if len(sentence) < 2:
                    continue
                cate = sentence[0]
                if cate not in self._M:
                    self._M[cate] = 0
                self._M[cate] += 1
                if cate not in self._A:
                    self._A[cate] = {}
                self._N += 1
                for tag in sentence[1:]:
                    if tag not in self._A[cate]:
                        self._A[cate][tag] = 0
                    if tag not in self._Q:
                        self._Q[tag] = 0
                    self._A[cate][tag] += 1
                    self._Q[tag] += 1

    def compute(self):
        """计算每个类别下每个标签的卡方值
        计算过程为：
        1.遍历类别的语料样本统计量
        2.遍历每个类别下标签的统计量
        3. 计算卡方值
        Args:
            无
        Return:
            无
        """
        for cate in self._M:
            m = self._M[cate]
            temp0 = self._N - m
            for tag in self._A[cate]:
                q = self._Q[tag]
                a = self._A[cate][tag]
                temp1 = a * (temp0 + a - q) - (q - a) * (m - a)
                chi = (temp1 * temp1) / (m * q * temp0 * (self._N - q))
                self._chi[cate].append((tag, chi))

    def save(self, chi_file, top=-1):
        """保存类别下的top个标签，输出到文件，格式为：类别 标签 卡方值
        Args:
            chi_file - 文件名
            top - 每个类别下保存的标签数（按照卡方值从高到低），默认全部保存
        Return:
            无
        """
        with open(chi_file, 'w') as chi_file_out:
            for cate in self._chi:
                for tag_chi in sorted(self._chi[cate], key=lambda x: x[1], reverse=True)[: top]:
                    chi_file_out.write("%s %s %.2f\n" % (cate, tag_chi[0], tag_chi[1]))
