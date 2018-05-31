import nltk
import itertools
from operator import itemgetter
import networkx as nx
import string
import math
import os
import glob
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cPickle as pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
import glob

topic = sys.argv[1]
tokenize = lambda doc: doc.lower().split(" ")
#apply syntactic filters based on POS tags
def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def lDistance(firstString, secondString):
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def buildGraph(nodes):
    "nodes - list of hashables that represents the nodes of the graph"
    #itertools generate all possible combinations ex {1,2,3} itertools.combinations(array,2)=1,2 1,3 2,3
    gr = nx.Graph() #initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    #add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = lDistance(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr

def extractKeyphrases(text):
    #tokenize the text using nltk
    wordTokens = nltk.word_tokenize(text)

    #assign POS tags to the words in the text
    tagged = nltk.pos_tag(wordTokens)
    textlist = [x[0] for x in tagged]
    
    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)
    #print tagged

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

   #this will be used to determine adjacent words in order to construct keyphrases with two words

    graph = buildGraph(word_set_list)

    #pageRank - initial value of 1.0, error tolerance of 0,0001, 
    #nx.pagerank()-returns the page rank of the nodes in the graph in thr form of a dictionary of nodes with pagerank as value 
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    #print calculated_page_rank

    #most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    print keyphrases
    #the number of keyphrases returned will be relative to the size of the text (a third of the number of vertices)
    aThird = len(word_set_list) / 3
    keyphrases = keyphrases[0:aThird+1]

    #take keyphrases with multiple words into consideration as done in the paper - if two words are adjacent in the text and are selected as keywords, join them
    #together
    modifiedKeyphrases = set([])
    dealtWith = set([]) #keeps track of individual keywords that have been joined to form a keyphrase
    i = 0
    j = 1
    while j < len(textlist):
        firstWord = textlist[i]
        secondWord = textlist[j]
        if firstWord in keyphrases and secondWord in keyphrases:
            keyphrase = firstWord + ' ' + secondWord
            modifiedKeyphrases.add(keyphrase)
            dealtWith.add(firstWord)
            dealtWith.add(secondWord)
        else:
            if firstWord in keyphrases and firstWord not in dealtWith: 
                modifiedKeyphrases.add(firstWord)

            #if this is the last word in the text, and it is a keyword,
            #it definitely has no chance of being a keyphrase at this point    
            if j == len(textlist)-1 and secondWord in keyphrases and secondWord not in dealtWith:
                modifiedKeyphrases.add(secondWord)
        
        i = i + 1
        j = j + 1
        
    return modifiedKeyphrases

def extractSentences(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())
    graph = buildGraph(sentenceTokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    #return a 100 word summary
    summary = ' '.join(sentences)
    summaryWords = summary.split()
    summaryWords = summaryWords[0:101]
    summary = ' '.join(summaryWords)

    return summary


#Topic-wise summary generation
def sentence_node_formation():
    docs = []
    i = 1
    for file in glob.glob('/home/sinchani/IR/Assignement2_IR/'+topic+'/*'):
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
#return sentences
keyphrases = extractKeyphrases(sentences)
summary = extractSentences(sentences)
print summary

'''
#retrieve each of the articles
articles = os.listdir("/home/sinchani/articles")
for article in articles:
    print 'Reading articles/' + article
    articleFile = open('/home/sinchani/articles/' + article, 'r')
    text = articleFile.read()
    text = text.decode('utf-8')
    keyphrases = extractKeyphrases(text)
    #keyphrases=keyphrasel.encode('utf-8')
    summary = extractSentences(text)
	#writeFiles(summary, keyphrases, article)
'''


text="""PINE RIDGE, S.D. (AP) -- President Clinton turned the attention  
of his national poverty tour today to arguably the poorest, most 
forgotten U.S. citizens of them all: American Indians.
Clinton was going to the Pine Ridge Reservation for a visit with  
the Oglala Sioux nation and to participate in a conference on 
Native American homeownership and economic development. He also was 
touring a housing facility and signing a pact with Oglala leaders 
establishing an empowerment zone for Pine Ridge.
But the main purpose of the visit -- the first to a reservation  
by a president since Franklin Roosevelt -- was simply to pay 
attention to American Indians, who are so raked by grinding poverty 
that Clinton's own advisers suggested he come up with special 
proposals geared specifically to the Indians' plight.
At Pine Ridge, a scrolling marquee at Big Bat's Texaco expressed  
both joy over Clinton's visit and wariness of all the official 
attention: ``Welcome President Clinton. Remember Our Treaties,'' 
the sign read.
According to statistics from the Census Bureau and the Bureau of  
Indian Affairs, there are 1.43 million Indians living on or near 
reservations. Roughly 33 percent of them are children younger than 
15, and 38 percent of Indian children aged 6 to 11 live in poverty, 
compared with 18 percent for U.S. children of all other races 
combined.
Aside from that, only 63 percent of Indians are high school  
graduates. Twenty-nine percent are homeless, and 59 percent live in 
substandard housing. Twenty percent of Indian households on 
reservations do not have full access to plumbing, and the majority 
-- 53.4 percent -- do not have telephones.
The per capita income for Indians is $21,619, one-third less  
than the national per capita income of $35,225. An estimated 50 
percent of American Indians are unemployed, and at Pine Ridge the 
problem is even more chronic -- 73 percent of the people do not have 
jobs.
Housing Secretary Andrew Cuomo, who visited the reservation last  
August, said Pine Ridge is a metaphor for the poverty tour, for it 
sits in Shannon County, the poorest census tract in the nation.
This is generations of poverty on the Pine Ridge reservation,  
with very, very little progress,'' Cuomo said. ``We didn't get into 
this situation in a couple of weeks and we're not going to get out 
of it in a couple of weeks. It's going to take years.
To begin addressing the housing problem, Clinton was announcing  
a partnership between the Treasury Department, the Department of 
Housing and Urban Development, tribal governments and mortgage 
companies to help 1,000 Indians become homeowners over the next 
three years -- a small number that nonetheless would double the 
number of government-insured home mortgages issued on tribal lands. 
Under the effort, ``one-stop mortgage centers'' would be opened at 
Pine Ridge and on the Navajo Reservation in Arizona to help 
streamline the mortgage lending process.
Cuomo said special steps were needed to help Indians create and  
own houses because the nature of the land on which they live 
effectively shuts them out of conventional home loan processes.
``The land is held in trust. The bank doesn't want to take it as  
collateral because it's in trust,'' Cuomo said. ``So the main asset 
on the reservation -- the land -- can't even be used.''
Also, two of the country's largest municipal securities  
underwriters, Banc One Capital Markets and George K. Baum &AMP; Co., 
were committing to underwriting $300 million in bonds annually for 
five years to create a market for reservation mortgages. Those 
bonds would help raise $1.5 billion that could then be lent to 
tribes, tribal housing authorities and individuals for buying 
homes.
The announcement was part of Clinton's four-day, cross-country  
tour to highlight the ``untapped markets'' in America's inner 
cities and rural areas
"""

