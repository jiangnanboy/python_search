#!/user/bin/python
#coding:utf-8
__author__='yanshi'

from com.sy.util import data
from collections import namedtuple
import  jieba
import codecs
from gensim.models import Doc2Vec
import multiprocessing
import numpy as np

class Doc2vec():
    def __init__(self, stopWordsPath, fileTitle, fileIntro):
        self.fileIntro=fileIntro
        initData = data.Init()
        self.stopWords = initData.loadStopWords(stopWordsPath)
        self.filmTitles, self.filmDocs = initData.readData(fileTitle, fileIntro)

    #分词
    def segment(self,text):
        words=[word for word in jieba.cut(text) if word not in self.stopWords]
        return words

    #将每行电影简介文本预处理成doc2vec能处理的格式
    def textProcess(self):
        #将每行文本存于namedtuple中，便于处理
        filmText_namedTuple=namedtuple('text','words tags')
        count=0
        self.filmTextList=[]
        with codecs.open(self.fileIntro,'r','utf-8') as file:
            for line in file:
                count += 1
                #这里注意fileText_namedTuple后面一项的id要与self.filmTitles中的id一致，不然在最后的结果显示会出错
                self.filmTextList.append(filmText_namedTuple(self.segment(line),[count]))


    def trainDoc2vec(self):
        #预处理
        self.textProcess()
        #获取cpu核心数
        cpuCores=multiprocessing.cpu_count()
        #隐层维数
        vec_size=500
        '''
        doc2vec参数说明：
        dm 定义了训练的算法。默认是dm=1,使用 ‘distributed memory’ (PV-DM)，否则 distributed bag of words (PV-DBOW)。
        size 是特征向量的纬度。
        window 是要预测的词和文档中用来预测的上下文词之间的最大距离。
        alpha 是初始化的学习速率，会随着训练过程线性下降。
        seed 是随机数生成器。.需要注意的是，对于一个完全明确的重复运行（fully deterministically-reproducible run），你必须同时限制模型单线程工作以消除操作系统线程调度中的有序抖动。（在python3中，解释器启动的再现要求使用PYTHONHASHSEED环境变量来控制散列随机化）
        min_count 忽略总频数小于此的所有的词。
        max_vocab_size 在词汇累积的时候限制内存。如果有很多独特的词多于此，则将频率低的删去。每一千万词类大概需要1G的内存，设为None以不限制（默认）。
        sample 高频词被随机地降低采样的阈值。默认为0（不降低采样），较为常用的事1e-5。
        workers 使用多少现成来训练模型（越快的训练需要越多核的机器）。
        iter 语料库的迭代次数。从Word2Vec中继承得到的默认是5，但在已经发布的‘Paragraph Vector’中，设为10或者20是很正常的。
        hs 如果为1 (默认)，分层采样将被用于模型训练（否则设为0）。
        negative 如果 > 0，将使用负采样，它的值决定干扰词的个数（通常为5-20）。
        dm_mean 如果为0（默认），使用上下文词向量的和；如果为1，使用均值。（仅在dm被用在非拼接模型时使用）
        dm_concat 如果为1，使用上下文词向量的拼接，默认是0。注意，拼接的结果是一个更大的模型，输入的大小不再是一个词向量（采样或算术结合），而是标签和上下文中所有词结合在一起的大小。
        dm_tag_count 每个文件期望的文本标签数，在使用dm_concat模式时默认为1。
        dbow_words 如果设为1，训练word-vectors (in skip-gram fashion) 的同时训练 DBOW doc-vector。默认是0 (仅训练doc-vectors时更快)。
        trim_rule 词汇表修建规则，用来指定某个词是否要被留下来。被删去或者作默认处理 (如果词的频数< min_count则删去)。可以设为None (将使用min_count)，或者是随时可调参 (word, count, min_count) 并返回util.RULE_DISCARD,util.RULE_KEEP ,util.RULE_DEFAULT之一。注意：这个规则只是在build_vocab()中用来修剪词汇表，而且没被保存。
       '''
        doc2vec_model=Doc2Vec(dm=1,dm_concat=0,size=vec_size,window=10,negative=0,hs=0,min_count=1,workers=cpuCores)
        #从一系列句子中建立词汇表
        doc2vec_model.build_vocab(self.filmTextList)
        #训练，训练步数为20，学习速率为0.99
        epoch=20
        for epo in range(epoch):
            try:
                print('epoch: %d' % epo)
                #从一系列句子更新模型的神经权重
                doc2vec_model.train(self.filmTextList,total_examples=len(self.filmTextList),epochs=doc2vec_model.iter)
                #学习速率
                doc2vec_model.alpha*=0.99
                doc2vec_model.min_alpha=doc2vec_model.alpha
            except(KeyboardInterrupt,SystemExit):
                break
        return doc2vec_model

    def doc2VecSearch(self,query):
        doc_model=self.trainDoc2vec()
        #模型的random设置一个固定值，以返回确定结果
        doc_model.random=np.random.RandomState(1)
        #infer_vec将查询query转换为词向量
        query_vec=doc_model.infer_vector(self.segment(query))
        #最相似的文档
        doc_sim=doc_model.docvecs.most_similar([query_vec],topn=5)
        for sim in doc_sim:
            print('sim: %f,title: %s' % (sim[1], self.filmTitles[sim[0]]))


