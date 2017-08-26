#!/user/bin/python
#coding:utf-8
__author__='yanshi'

from com.sy.util import data
import numpy as np
import jieba
import gensim
from gensim import models

class LSACorpus():

    def __init__(self, stopWordsPath, fileTitle, fileIntro):
        initData=data.Init()
        self.stopWords=initData.loadStopWords(stopWordsPath)
        self.filmTitles,self.filmDocs=initData.readData(fileTitle,fileIntro)
        #Dictionary中的参数为被拆成单词集合的文档的集合,dictionary把所有单词取一个set(),并对set中每个单词分配一个Id号的map
        #将所有文本的单词拿出来构成一个字典，将文档转换为LSA可以处理的格式
        self.dictionary=gensim.corpora.Dictionary(self.iter_docs())

    def __len__(self):
        return len(self.filmDocs)

    def __iter__(self):
        for tokens in self.iter_docs():
            #doc2bow根据本词典构构造的向量,是把文档 doc变成一个稀疏向量，[(0, 1), (1, 1)]，表明id为0,1的词汇出现了1次，至于其他词汇，没有出现。
            yield self.dictionary.doc2bow(tokens)

    def iter_docs(self):
        for filmDoc in self.filmDocs:
            yield( word for word in jieba.cut(filmDoc) if word not in self.stopWords)

'''
利用潜在语义分析计算查询与文档的相关度
首先将文档语料映射成三个矩阵U*S*V，这三个矩阵分别是词与主题矩阵，代表词与主题的相关度；主题的对角矩阵；主题与文档矩阵，
表示主题在文档中的分布
然后将查询词也映射到空间中qt=q*U*S中，再qt*V得到查询与每个文档的相关度，返回前top-k个文档

这个方法不同于传统的基于词存在的相关计算，它可以计算出词的相近词，就是説可以计算词不在文档中的相关度
'''
class LSA():
    def __init__(self,stopWordsPath, fileTitle, fileIntro):
        # 将文档转为gensim中的LSA可以读取和处理的格式
        self.corpus = LSACorpus(stopWordsPath, fileTitle, fileIntro)
    def lsaSearch(self,query):
        dict_copus = self.corpus.dictionary
        # 指定10个主题
        topics = 10
        lsi = models.LsiModel(self.corpus, num_topics=topics, id2word=dict_copus)
        # 获取U、V、S矩阵，查询词转换到潜在空间需要这些分解的矩阵
        U = lsi.projection.u
        S = np.eye(topics) * lsi.projection.s
        V = gensim.matutils.corpus2dense(lsi[self.corpus], len(lsi.projection.s)).T / lsi.projection.s
        # 单词的索引字典，将查询词转换为它在dict_copus相应的索引词
        dict_words = {}
        for i in range(len(dict_copus)):
            dict_words[dict_copus[i]] = i

        #将查询query转换为查询词向量
        q=np.zeros(len(dict_words.keys()))
        for word in jieba.cut(query):
            q[dict_words[word]]=1

        #将query的q权重向量（它经分词后的单词在dict_words中的相应索词）
        # 映射到qt中 qt=q*U*S为查询的词矩阵（就是查询中的词与主题矩阵，与主题的相关度）,大小与字典库相同
        qt=np.dot(np.dot(q,U),S)
        #与电影中的每篇简介的相关度
        similarity=np.zeros(len(self.corpus.filmDocs))
        for index in range(len(V)):#这里的V应该行是文档，列是主题（代表该文档在各个主题上的相关度），便与查询词矩阵点乘得到查询与文档的相关度
            similarity[index]=np.dot(qt,V[index])
        index_sim=np.argsort(similarity)[::-1]#排序
        for index in list(index_sim)[:5]:#最相关的前5个文档
            print('sim: %f,title: %s' % (similarity[index], self.corpus.filmTitles[index]))




