import math
import sys
import cPickle as pickle
import PorterStemmer

BETA = 0.65
STEM_IN_ROCCHIO = False
class RocchioOptimizeQuery:



    def __init__(self, firstQueryTerm):

        self.query = {}
        self.query[firstQueryTerm] = 1
     
        
    def Rocchio(self, invertedFile, documentsList, relevantDocs):

        p = PorterStemmer.PorterStemmer()

        weights = {}
        for term in invertedFile.iterkeys():
            sterm = term
            if STEM_IN_ROCCHIO:
                sterm = p.stem(term.lower(), 0,len(term)-1)            
            weights[sterm] = 0.0    #initialize weight vector for each key in inverted file
        print ''

        relevantDocsTFWeights = {}
   
        # ------------------------------------- #
        # Compute relevantDocsTFWeights and nonrelevantDocsTFWeights vectors
        for docId in relevantDocs:
            doc = documentsList[docId]
            for term in doc["tfVector"]:
                sterm = term
                if STEM_IN_ROCCHIO:
                    sterm = p.stem(term.lower(), 0,len(term)-1)

                if sterm in relevantDocsTFWeights:
                    relevantDocsTFWeights[sterm] = relevantDocsTFWeights[sterm] + doc["tfVector"][term]
                else:
                    relevantDocsTFWeights[sterm] = doc["tfVector"][term]


        # ------------------------------------- #
        # Compute Rocchio vector
        for term in invertedFile.iterkeys():
            idf = math.log(float(len(documentsList)) / float(len(invertedFile[term].keys())), 10)


            sterm = term
            if STEM_IN_ROCCHIO:
                sterm = p.stem(term.lower(), 0,len(term)-1)


            # Terms 2 and 3 of Rocchio algorithm
            for docId in invertedFile[term].iterkeys():
                if documentsList[docId]['IsRelevant'] == 1:
                    # Term 2: Relevant documents weights normalized and given BETA weight
                    weights[sterm] = weights[sterm] + constants.BETA * idf * (relevantDocsTFWeights[sterm] / len(relevantDocs))

            # Term 1 of Rocchio, query terms
            if term in self.query:
                self.query[term] = constants.BETA * self.query[term] + weights[sterm]   #build new query vector of weights
            elif weights[sterm] > 0:
                self.query[term] = weights[sterm]

        with open('output_lucene_after_relevance_feedback.txt', 'w') as file:
            file.write(pickle.dumps(self.query)) 