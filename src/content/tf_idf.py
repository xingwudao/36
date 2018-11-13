#!/usr/bin/python3.5
# -*- coding:utf-8 -*-

import sys

class TfIdf(object):

    def __init__(self, idf_file):
        """
        读入词典的IDF值，并计算出平均IDF值
        Args:
            idf_file -- IDF值文件
        Example:
            >>> tfidf = TfIdf(sys.argv[1])
            >>> document = read_file(sys.argv[2])
            >>> keywords = tfidf.extract_keywords(document)
            >>> print(keywords)
        """
        self._idf = {}
        self._idf_default = 0
        with open(idf_file, 'r') as idf:
            for line in idf:
                word_info = line.strip().split()
                self._idf[word_info[0]] = float(word_info[1])
                self._idf_default += float(word_info[1])
        self._idf_default /= float(len(self._idf))
        self._black_words = ['没有', '不能', '不会', '还有', '不让', '点儿', '知道', '了解', '出来', '不是']

    def extract_keywords(self, document, top = 10):
        """
        抽取关键词接口，关键词排名依据tf*idf

        Args:
            document -- 待抽取关键词的文档
            top -- 输出关键词个数
        returns:
            关键词及其权重
        """
        
        sentences = document.split('\n')
        keywords = {}
        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                if len(word) == 1 or word in self._black_words:
                    continue
                if word not in keywords:
                    keywords[word] = 0
                keywords[word] += 1

        for word in keywords:
            idf = self._idf_default
            if word in self._idf:
                idf = self._idf[word]
            keywords[word] *= idf

        return sorted(keywords.items(), key = lambda x: x[1], reverse = True)[: top]

def read_file(file_name):
    data = ""
    with open(file_name, 'r') as fopen:
        data = fopen.read()
        data = data.strip()
    return data


if __name__ == '__main__':
    tfidf = TfIdf(sys.argv[1])
    document = read_file(sys.argv[2])
    keywords = tfidf.extract_keywords(document)
    print(keywords)
