# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 08:23:15 2018

@author: Ishani
"""

import pandas as pd
import numpy as np

df2=pd.read_csv("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/query.txt",delimiter="  ",header=None)

df_true=pd.read_csv("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IRAssignment1/output.txt",delimiter=" ",header=None)
df_pred=pd.read_csv("C:/Users/JAYANTA/Downloads/relevant_docs_after_QueryExpansion_debanjana.txt",delimiter=" ",header=None)
q_id=df2[0].values
precision=[]
recall=[]
f1=[]
scores=[]

        
for i in range(len(q_id)):
    pred=np.zeros(50)
    true=[]
    count=0
    
    for j in range(len(df_true[0].values)):
        if df_true[0].values[j]==q_id[i]:
            true.append(df_true[1].values[j])
    
    for k in range(len(df_pred[0].values)):
        if df_pred[0].values[k]==q_id[i] and count<50 and df_pred[1].values[k] in true:
            pred[count]=1
            count=count+1
            #print(count)
       
    if np.count_nonzero(pred)==0:
        prec=0
    else :
        prec=(np.count_nonzero(pred)/50)  
    precision.append(prec)  #tp/tp+fp
    recall.append(prec)     #tp/tp+fn
    if prec==0:
        fscore=0
    else :
        fscore=2*prec*prec/(prec+prec)
    f1.append(fscore)
    scores.append(str(q_id[i])+"  |  Precision = "+str(prec)+"  |  Recall = "+str(prec)+"  |  f1 score = "+str(fscore))
       
with open("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/Performance_after_query_expansion.txt","w") as f:
    for item in scores:
        f.write("%s\n" %item) 
        
    f.write("\n Average Precision = %s" %(np.average(precision)))
    f.write("\n Average Recall = %s" %np.average(recall))
    f.write("\n Average F1 score = %s" %np.average(f1))
    
