#!/user/bin/python
#coding:utf-8
__author__='yanshi'

from com.sy.search import tfidf
from com.sy.search import lsa

class MainTest():
    def tfidfSearch(self):
        TF=tfidf.TFIDF('G:\中文语料\stopwords.txt','G:\中文语料\电影语料\电影名称.txt','G:\中文语料\电影语料\电影简介.txt')
        TF.tfIDFSearch('恐怖')
    def lsaSearch(self):
        LSA=lsa.LSA('G:\中文语料\stopwords.txt','G:\中文语料\电影语料\电影名称.txt','G:\中文语料\电影语料\电影简介.txt')
        LSA.lsaSearch('恐怖')
if __name__=='__main__':
    test=MainTest()
    test.tfidfSearch()
    print('-------------')
    test.lsaSearch()