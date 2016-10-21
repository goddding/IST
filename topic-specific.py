# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:15:05 2015

@author: xinliyang
"""

import lda
import xlrd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import math
import re
import nltk

def cal_Entropy(ratio):
    if ratio==0 or ratio==1:
        Entropy=0
    else:
        Entropy=-ratio*math.log(ratio,2)-(1-ratio)*math.log(1-ratio,2)
    return Entropy

#description
fname1 = "description.xlsx"    #"description_new.xlsx"
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

#LDA
NumTopic=118
model = lda.LDA(n_topics=NumTopic, n_iter=100, random_state=1)
model.fit(tf_array)

topic_word = model.topic_word_
n_top_words =10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print 'topic'+str(i)+': '
    print ' '.join(topic_words)
    
classes=[] 
doc_topic = model.doc_topic_
for i in range(NumApp):
     classes.append(doc_topic[i].argmax())


#api
fname2 = "api.xlsx"
sh2 = xlrd.open_workbook(fname2).sheets()[0]
nrow2=sh2.nrows

Topic_Api_Entropy=[]
Entropy_total_list=[]
topic_app_num=[]
topic_malicious_num=[]
for topic in range(NumTopic):
    Data_Label=[]
    topic_app=0
    malicious=0
    for i in range(NumApp):
        if classes[i]==topic:
            name=names[i]
            topic_app=topic_app+1
            if i>=1653:
                malicious=malicious+1
        
            j=1
            while sh2.row_values(j)[0]!=name:
                j=j+1
                if j==nrow2:
                    break
            if j==nrow2:
                continue
            while sh2.row_values(j)[0]==name:
                tmp=(sh2.row_values(j)[3],sh2.row_values(j)[4])
                Data_Label.append(tmp)
                j=j+1
                if j==nrow2:
                    break    
    topic_app_num.append(topic_app) 
    topic_malicious_num.append(malicious)
    
    #Information Gain
    dif_api=[]
    for i in range(len(Data_Label)):
        api=Data_Label[i][0]
        if dif_api.count(api)==0:
            dif_api.append(api)
    
    ratio_malicious=malicious*1.0/topic_app   
    Entropy_total=cal_Entropy(ratio_malicious)
    Entropy_total_list.append(Entropy_total)
    
    for i in range(len(dif_api)):
        api=dif_api[i]
        pos_malicious=Data_Label.count((api,1))
        pos_benign=Data_Label.count((api,0))
        pos=pos_malicious+pos_benign      
        if pos==0:
            ratio_pos=0
        else:
            ratio_pos=pos_malicious*1.0/pos    
        Entropy_pos=cal_Entropy(ratio_pos)
        
        neg=topic_app-pos
        neg_malicious=malicious-pos_malicious
        if neg==0:
            ratio_neg=0
        else:
            ratio_neg=neg_malicious*1.0/neg
        Entropy_neg=cal_Entropy(ratio_neg)    
  
        IG= Entropy_total-pos*1.0/topic_app*Entropy_pos-neg*1.0/topic_app*Entropy_neg
        if Entropy_total==0:
            IG_ratio=1   #flag for a special case
        else:
            IG_ratio=IG/Entropy_total
        Topic_Api_Entropy.append((topic,api,IG_ratio))
        
        

maxi_list=[]
num_pattern_list=[]
for idx in range(NumTopic):
    maxi=0
    num_pattern=0
    for i in range(len(Topic_Api_Entropy)):
        if Topic_Api_Entropy[i][0]==idx:
            num_pattern=num_pattern+1
            if Topic_Api_Entropy[i][2]>maxi:
                maxi=Topic_Api_Entropy[i][2]
        if Topic_Api_Entropy[i][0]==idx+1:
            break   
    maxi_list.append(maxi)
    num_pattern_list.append(num_pattern)
    

#api_entropy=[]
#for i in Topic_Api_Entropy:
#    if i[0]==62 and i[2]>0.1:
#        api_entropy.append(i)