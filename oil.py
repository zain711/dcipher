# summarize the number of unique values for each column using numpy
import json
import pandas as pd
import io
import requests
import spacy
import urllib
from gensim.models import Word2Vec


with open('WoS_data.json', 'r') as read_file:
   data = json.load(read_file)
#print(data)

nlp = spacy.load('en_core_web_sm')
lemmas = []
for d in data:
	text = nlp(str(d))
	lemmas.append([token.lemma_ for token in text if not token.is_stop])
print(lemmas)

model = Word2Vec(lemmas)

model = Word2Vec(min_count=1)
model.build_vocab(lemmas)  # prepare the model vocabulary
model.train(lemmas, total_examples=model.corpus_count, epochs=model.epochs)
#print('a')

import json
import pandas as pd
import numpy as np
import spacy
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from datascience.tables import Table

with open('WoS_data.json', 'r') as read_file:
    data = json.load(read_file)

nlp = spacy.load('en_core_web_sm')
lemmas = []
#for d in data:
text = nlp(str(data[0]))
lemmas.append([token.lemma_ for token in text if not token.is_stop and not token.is_punct])
#print(lemmas)

model = Word2Vec(lemmas, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = 4)
sustain = nlp('sustainability').vector

words = []
euc_dist = []
for list in lemmas:
    for word in list:
        vector = model[word]
        words.append(word)
        euc_dist.append(np.abs(np.average(sustain) - np.average(vector)))

tab = Table().with_columns(
    "word", words,
    "similarity", euc_dist
)
tab.sort("similarity")
#model.wv.similar_by_vector("bio",50)
#print(euc_dist)
#model.wv.similar_by_vector("bio",50)
#print(euc_dist)
