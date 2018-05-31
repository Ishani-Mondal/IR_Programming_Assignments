import os
import glob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import math
import string
os.chdir('/home/user/Desktop/demo_doc/')

tokens_list = []
terms = []

for file in glob.glob('*'):
	infile = open(file,"r")
	if(file.startswith('GX')):
		contents = infile.read()
		tokens_list.append(word_tokenize(contents))

tokens=[]
for token in tokens_list:
	for each in token:
		tokens.append(each.lower())

print(len(tokens))
terms = set(tokens)
print(len(terms))

stop_words = set(stopwords.words('english'))

filtered_terms = []
for term in terms:
	if term not in stop_words and not term.isdigit() and not term in string.punctuation and not '__' in term:
		filtered_terms.append(term)

vocabulary_size = len(filtered_terms)
print(vocabulary_size)
 
N=2

def word_count(doc):
	terms = []
	with open(doc,'r')as f:
		content = f.read().split()
		for term in content:
			if term not in stop_words and not term.isdigit() and not term in string.punctuation and not '__' in term:
				terms.append(term)
	return len(terms)

def term_frequency(doc, term):
	frequency = []
	with open(doc,'r') as f:
		content = f.read()
		l = content.split()
		frequency = l.count(term)
		return frequency / word_count(doc)

def doc_frequency(term):
	doc_frequency =0
	lowercase_content = []
	os.chdir('/home/user/Desktop/demo_doc/')
	for file in glob.glob('*'):
		infile = open(file)
		if(file.startswith('GX')):
			content = infile.read()
			for term in content.split():
				lowercase_content.append(term.lower())
			if term in lowercase_content:
				doc_frequency = doc_frequency + 1
	return doc_frequency
	#return math.log(1+N/doc_frequency)
	
def tf_idf(doc,term):
	return term_frequency(doc, term) * doc_frequency(term)

def calculate_doc_vector():
	doc_vector={}
	for file in glob.glob('*'):
		if(file.startswith('GX')):
			doc_vector[file]={}
			for term in filtered_terms:
				doc_vector[file][term]=tf_idf(file,term)
	return doc_vector

def query_term_frequency(doc, term):
	frequency = []
	os.chdir('/home/user/Desktop/')
	with open(doc,'r') as f:
		content = f.read()
		l = content.split()
		frequency = l.count(term)
		return frequency / word_count(doc)

def tf_idf_query(doc,term):
	return query_term_frequency(doc,term) * doc_frequency(term)

def calculate_query_vector():
	os.chdir('/home/user/Desktop/')
	query_vector={}
	f = open('query.txt', "r")
	lines = f.readlines()
	for line in lines:
		if (len(line)>1):
			query_id = line.split()[0]
			query_content = line.split()[1:]
			query_vector[query_id]={}
			for term in filtered_terms:
		    		query_vector[query_id][term]=tf_idf_query('query.txt',term)
	return query_vector

doc_vector = calculate_doc_vector()
query_vector = calculate_query_vector()

query_vector_file = open('query_vector.txt','w')
doc_vector_file = open('doc_vector.txt','w')

query_vector_file.write(str(query_vector))
doc_vector_file.write(str(doc_vector))