#!/usr/bin/python3.5
# -*- coding:utf-8 -*-

import sys
import numpy as np

class TextRank(object):
    def __init__(self, window, damping, iter_num):
        """
        初始化textrank算法。初始化参数：
        Args:
            window -- 窗口长度
            dampling -- 阻尼系数
            iter_num -- 迭代次数
        """
        self.window = window
        self.damping = damping
        self.iter_num = iter_num
        self.word_id = {}
        self.id_word = []
        self.sentences_id = []

        # 停用词
        self.black_words = ['没有', '不能', '不会', '还有', '不让', '点儿', '知道', '了解', '出来', '不是']

    def extract_keywords(self, article):
        """
        抽取关键词接口。
        Args:
            article -- 文本
        return:
            keywords -- 关键词列表，带权重 
        example:
        >>> data = read_file(sys.argv[1])
        >>> tr = TextRank(8, 0.85, 10)
        >>> keywords = tr.extract_keywords(data)
        >>> print(keywords)
        """
        # 切分句子，句子已经提前分好词，这里把词转换成ID
        self._cut_sentence(article)
        # 构建窗口内的共现矩阵
        self._create_matrx(self.sentences_id)
        # 利用共现矩阵迭代权重
        self._caculate_rank()
        # 返回top10关键词
        return self._get_keywords()

    def _cut_sentence(self, article):
        sentences = article.split("\n")
        for sentence in sentences:
            words = sentence.split()
            if len(words) < self.window:
                continue
            s = []
            for word in words:
                if len(word) == 1 or word in self.black_words:
                    continue
                if word not in self.word_id:
                    self.word_id[word] = len(self.word_id)
                    self.id_word.append(word)
                self.sentences_id.append(self.word_id[word])

    def _normailize_matrix(self):
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                if sum > 0:
                    self.matrix[i][j] /= sum
                else:
                    self.matrix[i][j] = 0

    def _create_matrx(self, sentence):
        # 初始化共现矩阵
        self.matrix = np.zeros([len(self.id_word), len(self.id_word)])
        sentence_len = len(sentence)
        for index, word in enumerate(sentence):
            left = index - self.window + 1  # 窗口左边界
            right = index + self.window  # 窗口右边界
            if left < 0:
                left = 0
            if right > sentence_len:
                right = sentence_len
            for i in range(left, right):
                if i == index:
                    continue
                self.matrix[sentence[index]][sentence[i]] += 1
                self.matrix[sentence[i]][sentence[index]] += 1

    def _caculate_rank(self):
        # 归一化矩阵
        self._normailize_matrix()
        # 随机初始化权重
        self.PR = np.random.rand(len(self.word_id), 1)
        for i in range(self.iter_num):
            self.PR = (1 - self.damping) + self.damping * np.dot(self.matrix, self.PR)
            self._normailize_matrix()
        return self._get_keywords()

    def _get_keywords(self):
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.id_word[i]] = self.PR[i][0]
        res = sorted(word_pr.items(), key=lambda x: x[1], reverse=True)[:10]
        return res


def read_file(file_name):
    data = ""
    with open(file_name, 'r') as fopen:
        data = fopen.read()
        data = data.strip()
    return data


if __name__ == '__main__':
    data = read_file(sys.argv[1])
    tr = TextRank(8, 0.85, 10)
    keywords = tr.extract_keywords(data)
    print(keywords)
