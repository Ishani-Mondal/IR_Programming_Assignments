# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:37:41 2018

@author: Debanjana
"""
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




df1=pd.read_csv("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/query.txt",delimiter="  ",header=None)
df2=pd.read_csv("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/sample_file (1).txt",sep=" ", header=None,index_col=0,na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)

#queries=df1[1].values
queries=["describe history oil industry","sources slate stone decorative"]
g_words=df2.index.values
g_vectors=df2.loc[:].values
old_new_queries=[]
all_qv=[]
#formulating query vector
for i in range(len(queries)):
    query_vector=np.zeros(300)
    cos_sim=[]
    expanded_query=""
    for word in queries[i].split():
        if word in g_words:
            query_vector=np.sum([query_vector,df2.loc[word].values],axis=0)
            
    all_qv.append(query_vector)
            
    #finding cosine similarity of the query vector with other words in glove
    for j in range(len(g_vectors)):
        cos_sim.append(cosine_similarity([query_vector],[g_vectors[j]])[0][0])
    
    #finding 5 most similar words
    sort_cos=np.sort(cos_sim)     
    k=len(queries[i].split())+5
    sort_cos=sort_cos[-k:]
    add_words=[]
    for j in range(k):
        if sort_cos[j] in cos_sim:
            add_words.append(g_words[cos_sim.index(sort_cos[j])])
    expanded_query=" ".join(add_words)
    old_new_queries.append([queries[i],expanded_query])
    print("Original query :",queries[i])
    print("Expanded query :",expanded_query)
    
with open("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/query_vector.txt","w") as f:
    for item in all_qv:
        f.write("%s\n" %item)
        
with open("C:/Users/JAYANTA/Documents/Debanjana/IITstuff/IR assignment 3/Expanded query.txt","w") as f:
    for item in old_new_queries:
        f.write("%s\n" %item)

        
    