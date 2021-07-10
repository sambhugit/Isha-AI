import json 
import numpy as np
import string
import random
import pickle
import nltk




############################################ Secondary Sentence Cleaner Functions #################################################

def full_remove(x, removal_list):

    for w in removal_list:
        x = x.replace(w, ' ')
    return x

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

def stem_with_porter(words):
    porter = nltk.PorterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words

# Main sentence cleaner function

def remove(sentences):
    ## Remove digits ##
    digits = [str(x) for x in range(10)]
    remove_digits = [full_remove(x, digits) for x in sentences]
    ## Remove punctuation ##
    remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]
    ## Make everything lower-case and remove any white space ##
    sents_lower = [x.lower() for x in remove_punc]
    sents_lower = [x.strip() for x in sents_lower]
    ## Remove stop words ##
    stops = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of']
    sents_processed = [removeStopWords(stops,x) for x in sents_lower]
    porter = [stem_with_porter(x.split()) for x in sents_processed]
    sents_processed = [" ".join(i) for i in porter]

    return sents_processed
