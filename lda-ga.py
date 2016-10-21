# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:55:49 2016

@author: xinliyang
"""

import xlrd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import lda
import re
import nltk
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors,Crossovers

fname1 = "description.xlsx"
sh1 = xlrd.open_workbook(fname1).sheets()[0]
nrow1=sh1.nrows

st=nltk.stem.snowball.EnglishStemmer()
english_vocab=set(w.lower() for w in nltk.corpus.words.words())
stopwords=nltk.corpus.stopwords.words('english')
corpus=[]
names=[]
for i in range(1,nrow1):
    name=sh1.row_values(i)[0]
    names.append(name) 
    
    string=sh1.row_values(i)[1]     
    string=re.sub(r'\W',' ',string)
    string=re.sub(r'\d','',string)
    #string=re.sub('_',' ',string)
    #string=re.sub('www','',string)
    tokens=nltk.word_tokenize(string)
    words=[st.stem(w) for w in tokens if len(w)>=3 and w.lower() not in stopwords and w.lower() in english_vocab]          
    description=' '.join(words)       
    corpus.append(description)
         
#term frequency   
NumApp=len(corpus)   
#NumFeatures=1000
vectorizer=CountVectorizer(stop_words='english', strip_accents='ascii', dtype='int32')#, max_features=NumFeatures)
tf_array=vectorizer.fit_transform(corpus).toarray()
vocab=vectorizer.get_feature_names()#获取词袋模型中的所有词语 

def eval_func(bstr):
   
    #LDA
    NTopic=bstr[0]*32+bstr[1]*16+bstr[2]*8+bstr[3]*4+bstr[4]*2+bstr[5]
    model = lda.LDA(n_topics=NTopic, n_iter=100)  # random_state=1
    model.fit(tf_array)
    classes=[] 
    doc_topic = model.doc_topic_
    for i in range(NumApp):
        classes.append(doc_topic[i].argmax())
    
    #Centroid
    Centers=[]
    Clusters=[]
    for i in range(NTopic):
        tmp_sum=np.zeros((1,len(vocab)))
        cnt=0
        points=[]
        for j in range(NumApp):
            if classes[j]==i:
                points.append(tf_array[j])
                tmp_sum=tmp_sum+tf_array[j]
                cnt=cnt+1
        c_i=tmp_sum/cnt
        Centers.append(c_i)
        Clusters.append(points)
        
    #Silhouette coefficient
    s_j=[]
    for j in range(NumApp):
        min_b=10000
        max_a=0
        for i in range(len(Centers)):
            b=np.linalg.norm(Centers[i]-tf_array[j])
            if b<min_b and b!=0:
                min_b=b
        clu=classes[j]
        for k in range(len(Clusters[clu])):
            a=np.linalg.norm(Clusters[clu][k]-tf_array[j])
            if a>max_a:
                max_a=a
        s_j.append((min_b-max_a)/max(max_a,min_b))
    s=sum(s_j)/NumApp

    return s+1

#genome = G1DList.G1DList(1)
genome=G1DBinaryString.G1DBinaryString(7)
genome.crossover.set(Crossovers.G1DBinaryStringXUniform)
genome.evaluator.set(eval_func)
#genome.setParams(rangemin=2, rangemax=100)
ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GRankSelector)
ga.setPopulationSize(20)
ga.setGenerations(10)
ga.evolve(freq_stats=1)
print ga.bestIndividual()