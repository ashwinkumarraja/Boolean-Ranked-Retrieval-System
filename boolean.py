import timeit
import nltk
import collections
import re
import os
import collections
import time
import string
import math
import time as tm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from natsort import natsorted

no_of_doc = 0
posting_list = {}
freq = {}
fileno = 0
file_map = {}
pos_index = {}
tfidf_matrix = []
doclist = []
magnitude = []
queryterms = []
wt_query_dict = {}
weight = []
#finalWeight = []

# function to calculate weighted term frequency respective to each document


def weighted_term_frequency(term, tokenized_document):
    try:
        count = len(pos_index[term][1][tokenized_document])
    except KeyError:
        return 0
    return 1 + math.log(count)


def weighted_query_term_frequency(term, tokenized_document):
    try:
        count = tokenized_document.count(term)
    except KeyError:
        return 0
    return 1 + math.log(count)

# function to calculate the inverse document frequency values for each document term


def inverse_document_frequencies():
    idf_values = {}

    for tkn in pos_index:
        idf_values[tkn] = 1 + math.log(no_of_doc / len(pos_index[tkn][1]))
    return idf_values


# function to calculate the tf-idf for each document term


def tfidf():
    for i in range(no_of_doc):
        magnitude.append(0)

    idf = inverse_document_frequencies()
    for term in pos_index:
        templist = []
        # print(pos_index[term][1])
        for document in doclist:
            tf = weighted_term_frequency(term, document)
            doc_tfidf = tf * idf[term]
            templist.append(doc_tfidf)
            magnitude[document] += doc_tfidf*doc_tfidf
        tfidf_matrix.append(templist)
    # Divide by magnitude of vectors
    for termidf in tfidf_matrix:
        for i in range(no_of_doc):
            termidf[i] = termidf[i]/math.sqrt(magnitude[i])
    # print(tfidf_matrix)


# function to calculate the tf-idf value for each query term
def tfidf_query(query):
    
    queryy = preprocessing(query)
    

    #wt_query_dict = collections.defaultdict(dict)
    idf_query = inverse_document_frequencies()

    for term in queryy:
        tf = weighted_query_term_frequency(term, queryy)
        query_tfidf = tf * idf_query[term]
        wt_query_dict[term] = query_tfidf


def preprocessing(final_string):
    # Tokenize.
    
    
    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(final_string)

    # Remove punctuations.
    table = str.maketrans('', '', '\t')
    token_list = [word.translate(table) for word in token_list]
    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in token_list]
    token_list = [str for str in stripped_words if str]

    # Change to lowercase.
    token_list = [word.lower() for word in token_list]
    stemmer = PorterStemmer()
    for pos, term in enumerate(token_list):
        token_list[pos] = stemmer.stem(term)
    return token_list


def final_weight(queryterms):
    tfidf_query(queryterms)
    queryterms = preprocessing(queryterms)
    proximity_weight(queryterms)
    '''
    [weight] holds proximity weights of queryterms per document
    {wt_query_dict} maps a queryterm with its tfidf
    [[tfidf_matrix]] holds list of (list of tfidf per document) for each word in corpus
    '''
    finalWeight=[]
    # print(tfidf_matrix[list(pos_index).index('appl')])
    for doc in range(no_of_doc):
        docweight = 0
        for term in queryterms:
            docweight += (0.3 *
                          tfidf_matrix[list(pos_index).index(term)][doc]*wt_query_dict[term])+(0.7*weight[doc])
        finalWeight.append((docweight, doc))
    
    temp = sorted(finalWeight,key= lambda x:float(x[0]),reverse=True)
    finalWeight = temp
    
    for i in finalWeight:
        if i[0] > 0:
            print("Document ID: ",doclist[i[1]]+1,".txt","-",i[0])
    #print(finalWeight)


def proximity_weight(input1):

    print("Preprocessed Query - ", input1)
    wt = 0
    #weight = []
    for docid in range(no_of_doc):
        pointers = []
        for qterm in input1:
            if qterm in pos_index:
                try:
                    pointers.append([0, pos_index[qterm][1][docid][0]])
                except:
                    pointers.append([0, -1])
        # pointers = list(pointers)
        while True:
            # print(type(pointers))
            try:
                minptr, minval = min(pointers, key=lambda t: t[1])
                maxval = max(pointers)[1]
                # print(maxval)
                # print(minval)
                if maxval-minval <= 10:
                    wt += 1
                    for i in range(len(input1)):
                        pointers[i][0] += 1
                        pointers[i][1] = pos_index[input1[i]
                                                   ][1][docid][pointers[i][0]]
                else:
                    for i in range(len(input1)):
                        if pointers[i][0] == minptr:
                            pointers[i][0] += 1
                            pointers[i][1] = pos_index[input1[i]
                                                       ][1][docid][pointers[i][0]]
                            break
            except KeyError:
                weight.append(0)
                break
            except IndexError:
                weight.append(wt)
                break
    # print(weight)
    return weight


if __name__ == "__main__":
    start_time = tm.time()
    path = os.getcwd()
    path = path + "\Data\\"
    start_time = time.time()
    doc_dictionary = dict()
    # master_dict = collections.defaultdict(dict)
    stemmer = PorterStemmer()
    for file in os.listdir(path):
        #print("Document ID: ", file)
        qp_file = open(path+file, 'r')

        if file.endswith(".txt"):
            doc_id = re.sub('[^\d+]', "", file)
            doc = "doc" + str(doc_id)

            no_of_doc += 1
            # Assigning unique id to each document
            doc_dictionary.__setitem__(doc, file)

            # to read specific file
            currentFile = open(os.path.join(path, file), encoding = "utf8")
            content =re.sub('[^A-Za-z+]', "", file)
            
            content = currentFile.read().lower()
            currentFile.close()

            filtered_sentence = []
            filtered_sentence = preprocessing(content)

            for pos, term in enumerate(filtered_sentence):

                # First stem the term.
                term = stemmer.stem(term)

                # If term already exists in the positional index dictionary.
                if term in pos_index:

                    # Increment total freq by 1.
                    pos_index[term][0] = pos_index[term][0] + 1

                    # Check if the term has existed in that DocID before.
                    if fileno in pos_index[term][1]:
                        pos_index[term][1][fileno].append(pos)

                    else:
                        pos_index[term][1][fileno] = [pos]

                # If term does not exist in the positional index dictionary
                # (first encounter).
                else:

                    # Initialize the list.
                    pos_index[term] = []
                    # The total frequency is 1.
                    pos_index[term].append(1)
                    # The postings list is initally empty.
                    pos_index[term].append({})
                    # Add doc ID to postings list.
                    pos_index[term][1][fileno] = [pos]

            # Map the file no. to the file name.
            file_map[fileno] = file
            doclist.append(fileno)
            # Increment the file no. counter for document ID mapping
            fileno += 1
            # print(pos_index['appl'][1][0][5])
    tfidf()
    print(" \n Preprocessing of given data set took --- %s seconds --- \n" % (tm.time() - start_time)) 
    '''
    while(query exists)
    {
        take query
        final_weight(query)
    }
    '''
    
    query = input("Enter the query : ")

    start_time = tm.time()

    
    final_weight(query)


    print("\n Total Time taken for retrieval --- %s seconds --- \n" % (tm.time() - start_time))
    
   

    # Also implement parts of speech tagging and dictionary compression schemes along with heap minimizations
