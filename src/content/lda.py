#!/usr/bin/python3.5
# -*- coding:utf-8 -*-

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import sys
import argparse

class LDA(object):

    def __init__(self, topics = 10, 
                 worker = 3, 
                 pretrained_model = None, 
                 dictionary = None):
        """
        lda模型训练初始化。
        Args:
            topics -- 指定主题个数
            worker -- 并行化参数，一般为core数量减一
            pretrained_model -- 预训练的模型，由于支持在线更新，所以可以加载上次训练的模型
            dictionary -- 训练时词需要转换成ID，所以跟模型配套有一个ID映射的词典
        Example:
            >>> lda = LDA(topics = 20, worker = 2, 
                          pretrained_model = model_file, 
                          dictionary = dictionary_file)
            >>> corpus = read_file(corpus_file) # [['word1', 'word2'], ['word3', 'word4']]
            >>> lda.update(corpus)
            >>> lda.save(model_file, dictionary_file)
            >>> topics = lda.inference(['word5', 'word6'])
        """

        self._topics = topics
        self._workers = worker
        self._model = None
        self._common_dictionary = None
        if pretrained_model and common_dictionary:
            self._model = LdaModel.load(pretrained_model)
            self._common_dictionary = Dictionary.load(dictionary)

    def save(self, model_file, dictionary_file):
        """
        保存训练的模型，同时保存对应的词典
        Args:
            model_file -- 模型文件
            dictionary_file -- 词典文件
        Returns:
            无
        """

        if self._model:
            self._model.save(model_file)
        if self._common_dictionary:
            self._common_dictionary.save(dictionary_file)

    def update(self, corpus = [[]]):
        """
        在线更新，在已有模型的基础上在线更新
        Args:
            corpus -- 用于更新的文档列表
        """

        if not self._model and len(corpus) > 0:
            self._common_dictionary = Dictionary(corpus)
            corpus_data =  [self._common_dictionary.doc2bow(sentence) for sentence in corpus]
            self._model = LdaModel(corpus_data, self._topics)
        elif self._model and len(corpus) > 0:
            self._common_dictionary.add_documents(corpus)
            new_corpus_data =  [self._common_dictionary.doc2bow(sentence) for sentence in corpus]
            self._model.update(new_corpus_data)

    def inference(self, document = []):
        """
        对新文档推断其话题分布
        Args:
            document -- 文档，其实是词列表
        Returns:
            话题分布列表        
        """
        if self._model:
            doc =  [self._common_dictionary.doc2bow(document)]
            return self._model.get_document_topics(doc)
        return []

    @property
    def model(self):
        return self._model

    @property
    def dictionary(self):
        return self._common_dictionary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train word embedding and query similar words')
    parser.add_argument('-m', '--model', type = str, default = None, help = """pretrained model""")
    parser.add_argument('-c', '--corpus', help = """corpus file""")
    parser.add_argument('-s', '--save', help = """save new model""")
    parser.add_argument('-d', '--dictionary', type = str, default = None, help = """dictionary""")
    parser.add_argument('-t', '--topics', type = int, default = 10, help = """topics""")
    args = vars(parser.parse_args())
    lda = LDA(topics = args['topics'],
              pretrained_model = args['model'],
              dictionary = args['dictionary'])
    
    sentences = []
    with open(args['corpus'], 'r') as corpus_file:
        for line in corpus_file:
            words = line.strip().split()
            if len(words) == 0:
                continue
            sentences.append(words)
    lda.update(sentences)
    lda.save(args['save'], args['dictionary'])
    print('print topics:')
    topics = lda.model.print_topics(num_topics = -1, num_words = 10)
    for topic in topics:
        words = []
        for word_value in topic[1].split('+'):
            value, word = word_value.split('*')
            word = word.strip()
            word = lda.dictionary[int(word[1:-1])]
            words.append("%s:%s" % (word, value))
        print(" ".join(words))

    while True:
        print("please input a sentence:")
        line = sys.stdin.readline()
        line = line.strip()
        if line == 'exit' or line == 'quit':
            break

        words = line.split()
        topics = lda.inference(words)
        print(topics)
