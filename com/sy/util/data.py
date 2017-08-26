#!/user/bin/python
#coding:utf-8
__author__='yanshi'

import codecs

'''
初始化，加载停词，加载数据
'''
class Init():
    # 加载返回停词
    def loadStopWords(self, stopWordsPath):
        print("加载停词...")
        stopWordsList = [line.strip() for line in codecs.open(stopWordsPath, 'r', 'utf-8').readlines()]
        stopWords = {}.fromkeys(stopWordsList)
        return stopWords

    # 读取电影相关数据，返回电影标题和简介
    def readData(self, fileTitle, fileIntro):
        filmDocs = []  # 电影简介
        filmTitles = []  # 电影标题
        filmTitleDict = {}
        with codecs.open(fileTitle, 'r', 'utf-8') as file:
            id = 0
            for line in file:
                id += 1
                filmTitleDict[id] = line.strip()
        with codecs.open(fileIntro, 'r', 'utf-8') as file:
            id = 0
            for line in file:
                id += 1
                filmTitles.append(filmTitleDict[id])
                filmDocs.append(line.strip())
        return filmTitles,filmDocs


