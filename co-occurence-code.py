from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText
import re
import numpy as np
import itertools

def co_occurrence(docs,k,num_words,select_words):
    #select_words will be added to the word_list if not in the word_list
    # we concatenate the docs into one single doc
    text=""
    for doc in docs:
        text+= " "+ doc
    token= tokenize.sent_tokenize(text)
    token={sentence.lower() for sentence in token}
    token = {re.sub("[^a-zA-Z .!?]", " ", sentence) for sentence in token}

    new_token=[]
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in token:
        new_token.append(tokenizer.tokenize(sentence))

    word_list=np.hstack(new_token)
    counter = collections.Counter(word_list)
    
    word_list= list(counter.keys())[0:num_words]
    for word in select_words:
        if word not in word_list:
            word_list.append(word)
    word_list=sorted(set(word_list))
    
    
    newnewtoken=[]
    for sentence in new_token:
        newnewsentence=[]
        for word in sentence:
            if word in word_list:
                newnewsentence.append(word)
        if newnewsentence:
            newnewtoken.append(newnewsentence)
    
    word2Ind= dict(zip(word_list,range(0,len(word_list))))

    M= [[0]*len(word_list) for i in range(len(word_list))]
    
    # count co-occurence
    for sentence in newnewtoken:
        if len(sentence)<=k:
            for i in range(len(sentence)-1):
                tmp_prod= list(itertools.product([sentence[i]],sentence[i+1:]))
                for tup in tmp_prod:
                    M[word2Ind[tup[0]]][word2Ind[tup[1]]] += 1
        else:
            for i in range(k):
                tmp_prod= list(itertools.product([sentence[i]],sentence[i+1:k+1]))
                for tup in tmp_prod:
                    M[word2Ind[tup[0]]][word2Ind[tup[1]]] += 1

            for i in range(1,len(sentence)-k):
                #tmp_prod= list(itertools.product(sentence[i:i+k],[sentence[i+k]]))
                #for tup in tmp_prod:
                #   dicts[tup] += 1
                for j in range(i,i+k):
                    M[word2Ind[sentence[j]]][word2Ind[sentence[i+k]]] += 1

    M=np.array(M)
    M= M+M.transpose()

    return M, word2Ind
