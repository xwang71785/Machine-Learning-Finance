# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:06:33 2017
Natural Language Process - NLTK
@author: wangx3
"""
import nltk
import nltk.book as bk
import nltk.corpus as cp
import random
from nltk.corpus import sinica_treebank

sinica_treebank.words()
sinica_treebank.parsed_sents()[15].draw()

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.wword_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]



'''
def gender_features(word):
    return {'last_letter': word[-1]}

Names = ([(name, 'male') for name in cp.names.words('male.txt')] + 
          [(name, 'female') for name in cp.names.words('female.txt')])
random.shuffle(Names)
featuresets = [(gender_features(n), g) for (n, g) in Names]
train_set, test_set = featuresets[500 :], featuresets[: 500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))
print(nltk.classify.accuracy(classifier, test_set))
'''


'''
def lexical_diversity(text):
    return len(text) / len(set(text))

# searching text in books, concordance provide context
bk.text1.concordance('monstrous')

bk.text2.common_contexts(['monstrous', 'very'])
bk.text4.dispersion_plot(['citizens', 'democracy', 'freedom','duties','America'])
print(len(set(bk.text3)))
print(len(bk.text3))
print(bk.text3.count('smote'))

sent1 = ['Call', 'me', 'Ishmael', '.']
print(lexical_diversity(sent1))
'''

