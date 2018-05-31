import os
import glob
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import math
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

topic = sys.argv[1]
threshold = float(sys.argv[2])

tokenize = lambda doc: doc.lower().split(" ")

#Topic-wise summary generation
def sentence_node_formation():
	docs = []
	i = 1
	for file in glob.glob('/home/user/Desktop/Assignement2_IR/'+topic+'/*'):
		infile = open(file,"r")
		contents = infile.read()
		soup = BeautifulSoup(contents,'xml')
		paragraphs = soup.find_all('P')
		for paragraph in paragraphs:
			docs.append(paragraph.get_text())
	return docs
	    
sentences = sentence_node_formation()
documents={}
sent = " ".join(str(x) for x in sentences)
sent = sent.replace('\n',"")
sentences = sent_tokenize(sent)
n=len(sentences)

#Compute the cosine similarity of two sentences
all_documents=sentences[:n]
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

# TF-IDF Score computation of the sentence vectors
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)

skl_tfidf_comparisons = []
for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
    for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

#Write the Tf-Idf scores to a file
tf_idf_score = open('tf_idf_score.txt','w')
tf_idf_score.write(str(skl_tfidf_comparisons))
count = 0

# Building the matrix from tf-idf scores
arr=np.empty((n,n))
for i in list(skl_tfidf_comparisons):
	arr[i[1],i[2]]=i[0]
count = 0
matrix = np.asmatrix(arr).tolist()

## Thresholding the matrix and building a boolean sparse matrix out of it
for index, row in enumerate(matrix):
	for elem in row:
		if(elem<threshold):
			matrix[index][row.index(elem)] = 0
		else:
			matrix[index][row.index(elem)] = 1

# Degree-centrality based summary generation of 250 words
summary = []

while (len(summary)<10):
	degree=[]
	for node in matrix:
		degree.append(node.count(1))

	max_degree=max(degree)
	Node_number=degree.index(max_degree)
	print(Node_number,max_degree)
	print('-------------------------Max Node Matrix-----------------------------------')
	print(matrix[Node_number])
	summary.append(Node_number)
	item_indices = []
	incident1_index = 0
	incident1_index_list=[]
	for item in matrix[Node_number]:
		if(item == 1):
			incident1_index_list.append(incident1_index)
		incident1_index = incident1_index + 1
	print('-------------------------Matrix Incident 1 List of Max Node-----------------------------------')
	print(incident1_index_list)
	list_of_zeros = [0]*n
	for index in incident1_index_list:
		for item in matrix[index]:
			matrix[index]=list_of_zeros


file1 = open('Summary_'+topic+'_'+str(threshold),'w')
for elem in summary:
	file1.write(all_documents[elem]+'\n')
	print(all_documents[elem]+'\n')