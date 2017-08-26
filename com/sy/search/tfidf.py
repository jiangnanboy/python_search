#!/user/bin/python
#coding:utf-8
__author__='yanshi'

from com.sy.util import data
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''
利用单词的tfidf和consine计算query与doc的相关性
'''
class TFIDF():
    def __init__(self, stopWordsPath, fileTitle, fileIntro):
        initData=data.Init()
        self.stopWords=initData.loadStopWords(stopWordsPath)
        self.filmTitles,self.filmDocs=initData.readData(fileTitle,fileIntro)

    def cutText(self):
        cutFileDocs = []
        for filmDoc in self.filmDocs:
            wordsList=self.segment(filmDoc)
            cutFileDocs.append(wordsList)
        return cutFileDocs

    def segment(self,text):
        words=jieba.cut(text)#jieba分词
        wordsList=[word for word in words if word not in self.stopWords]
        return ' '.join(wordsList)

    def tfIDFSearch(self, query):
        cutFileDocs = self.cutText()
        # 文本转为TF-IDF特征矩阵
        tfIDf = TfidfVectorizer(min_df=1)  # min_df=1 过滤低于给出阈值的文档频率的词
        tfIDF_fit = tfIDf.fit(cutFileDocs)  # 开始训练
        tfIDF_vec = tfIDF_fit.transform(cutFileDocs)  # 转换文档为文档-词矩阵

        # 需要将查询语句转转为词向量
        queryWords =[self.segment(query)]
        queryVec=tfIDF_fit.transform(queryWords)
        similarity=cosine_similarity(queryVec,tfIDF_vec)[0]
        similarity_index=similarity.argsort()[::-1]#最相关的排在最前，返回index
        #返回前5个最相关的电影
        for index in list(similarity_index)[:5]:
            print('sim: %f,title: %s' %(similarity[index],self.filmTitles[index]))
