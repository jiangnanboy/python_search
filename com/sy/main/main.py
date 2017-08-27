#!/user/bin/python
#coding:utf-8
__author__='yanshi'

from com.sy.search import tfidf
from com.sy.search import lsa
from com.sy.search import doc2vec

#测试
class MainTest():

    def __init__(self,stopwords,title,text):
        self.stopwordsPath=stopwords
        self.filmTitle=title
        self.filmIntro=text

    def tfidfSearch(self,query):
        TF=tfidf.TFIDF(self.stopwordsPath,self.filmTitle,self.filmIntro)
        TF.tfIDFSearch(query)

    def lsaSearch(self,query):
        LSA=lsa.LSA(self.stopwordsPath,self.filmTitle,self.filmIntro)
        LSA.lsaSearch(query)

    def doc2vec(self,query):
        Doc2vec=doc2vec.Doc2vec(self.stopwordsPath,self.filmTitle,self.filmIntro)
        Doc2vec.doc2VecSearch(query)
if __name__=='__main__':
    test=MainTest('G:\中文语料\stopwords.txt','G:\中文语料\电影语料\电影名称.txt','G:\中文语料\电影语料\电影简介.txt')
    query='恐怖'
    print('------tfidf-------')
    test.tfidfSearch(query)
    print('------lsa-------')
    test.lsaSearch(query)
    print('------doc2vec-------')
    test.doc2vec(query)