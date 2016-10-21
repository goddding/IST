# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 08:36:49 2016

@author: xinliyang
"""

import xlrd
import math

def cal_Entropy(ratio):
    if ratio==0 or ratio==1:
        Entropy=0
    else:
        Entropy=-ratio*math.log(ratio,2)-(1-ratio)*math.log(1-ratio,2)
    return Entropy

#description
fname1 = "description.xlsx"     #"description_new.xlsx"
sh1 = xlrd.open_workbook(fname1).sheets()[0]
nrow1=sh1.nrows

names=[]
for i in range(1,nrow1):
    name=sh1.row_values(i)[0]
    names.append(name) 
NumApp=len(names) 

#api
fname2 = "api.xlsx"
sh2 = xlrd.open_workbook(fname2).sheets()[0]
nrow2=sh2.nrows

Data_Label=[]
for i in range(NumApp):
    name=names[i]

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
    
#Information Gain
dif_api=[]
for i in range(len(Data_Label)):
    api=Data_Label[i][0]
    if dif_api.count(api)==0:
        dif_api.append(api)

malicious=1612  #701
ratio_malicious=malicious*1.0/NumApp    
Entropy_total=cal_Entropy(ratio_malicious)

Api_Entropy=[]
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
    
    neg=NumApp-pos
    neg_malicious=malicious-pos_malicious
    if neg==0:
        ratio_neg=0
    else:
        ratio_neg=neg_malicious*1.0/neg
    Entropy_neg=cal_Entropy(ratio_neg)   
    
    IG= Entropy_total-pos*1.0/NumApp*Entropy_pos-neg*1.0/NumApp*Entropy_neg
    if Entropy_total==0:
        IG_ratio=1   #flag for a special case
    else:
        IG_ratio=IG/Entropy_total
    Api_Entropy.append((api,IG_ratio))

maxi=0
api_entropy=[]
for i in Api_Entropy:
    if i[1]>=0.1:
        api_entropy.append(i)
    if i[1]>maxi:
        maxi=i[1]