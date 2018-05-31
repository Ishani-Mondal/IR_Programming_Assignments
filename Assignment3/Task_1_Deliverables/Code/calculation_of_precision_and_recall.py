import os
import sys
import re
import nltk
import pickle as pkl
import time
import pandas as pd


def precision_and_recall(output_file,filename):
    output = open("output.txt",'r')
    query = open("query.txt","r")
    est_output = open(output_file)
    query_ID = []
    prerec = open(filename,"w")
    prerec.write("queryID" + " " + "precision" + " " + "recall" + " " + "F-score" + "\n")
    for line in query:
        query_ID.append(line.split()[0])
    e_out = pd.DataFrame(columns = ["q_ID","Doc"])
    o_out = pd.DataFrame(columns=["q_ID","Doc"])
    query = []
    docs = []
    for line in est_output:
        query.append(line.split()[0])
        docs.append(line.split()[1])
    e_out['q_ID'] = query
    e_out["Doc"] = docs
    query = []
    docs = []
    for line in output:
        query.append(line.split()[0])
        if(line.split()[1] != " "):
            docs.append(line.split()[1])
    o_out['q_ID'] = query
    o_out["Doc"] = docs
        
    for q_ID in query_ID:
        estimated = list(e_out[e_out['q_ID'] == q_ID]["Doc"])
        true = list(o_out[o_out["q_ID"] == q_ID]["Doc"])
        if (len(estimated)>0):
            precision = len(list(set(estimated).intersection(set(true))))/float(len(estimated))
            recall = len(list(set(estimated).intersection(set(true))))/float(len(true))
            f_score = 2 * (precision * recall)/(precision + recall)
            prerec.write(str(q_ID) + " " + str(precision) + " " + str(recall) + " " + str(f_score) + "\n")
    prerec.close()
    output.close()
    est_output.close()

if __name__ == "__main__":
	precision_and_recall("output_lucene.txt","Performance_before_relevance_feedback.txt")
    precision_and_recall("output_lucene_after_relevance_feedback.txt","Performance_before_relevance_feedback.txt")
