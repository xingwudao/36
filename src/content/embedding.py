#!/usr/bin/python3.5
# -*- coding:utf-8 -*-

from gensim.models import Word2Vec
import sys
import argparse

class Embedding(object):

    def __init__(self, size = 128, window = 5, 
                 min_count = 5, workers = 2, 
                 epochs = 10,
                 pretrained_model = None):
        """
        以gensim的接口训练词嵌入向量。
        Args:
            size -- 向量维度
            window -- 窗口长度
            min_count -- 最小词频
            workers -- 并行化
            epochs -- 迭代次数
            pretrained_model -- 预训练的模型
        """
        self._model = None
        self._size = size
        self._window = window
        self._min_count = min_count
        self._workers = workers
        self._epochs = epochs
        if pretrained_model:
            self._model = Word2Vec.load(pretrained_model)

    def train(self, sentences = []):
        if self._model:
            self._model.train(sentences,
                              total_examples = len(sentences),
                              epochs = self._epochs)
        else:
            self._model = Word2Vec(sentences,
                                   self._size,
                                   window=self._window,
                                   min_count=self._min_count,
                                   workers=self._workers)
    @property
    def model(self):
        return self._model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train word embedding and query similar words')
    parser.add_argument('-m', '--model', help = """pretrained model""")
    parser.add_argument('-d', '--data', help = """corpus file""")
    parser.add_argument('-s', '--save', help = """save new model""")
    args = vars(parser.parse_args())

    sentences = []
    with open(args['data'], 'r') as corpus_file:
        for line in corpus_file:
            words = line.strip().split()
            if len(words) == 0:
                continue
            sentences.append(words)
    embedding = Embedding(pretrained_model = args['model'])
    embedding.train(sentences)
    embedding.model.save(args['save'])
    while True:
        print("please input a word in vocabulary:")
        line = sys.stdin.readline()
        line = line.strip()
        if line == 'exit' or line == 'quit':
            break
        keywords = embedding.model.wv.similar_by_word(line)
        print(keywords)
