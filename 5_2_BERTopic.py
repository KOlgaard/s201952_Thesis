#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import kaleido
import pickle
import numpy as np
import pandas as pd
import os.path
import os
import nltk, re, pprint
#nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams
import unidecode
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import networkx as nx
from nltk.corpus import PlaintextCorpusReader
from math import log, log10
from wordcloud import WordCloud
import gensim
from nltk.test.gensim_fixt import setup_module
setup_module()
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from bertopic import BERTopic
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from umap import UMAP
from bertopic.representation import BaseRepresentation

def get_text(data):
  textstrings = data['post'].to_list()
  textstrings = [post + ' <message_end>' for post in textstrings]
  return textstrings

def chunkify(text_data, chunk_size=100):
  #textchunks = np.empty((int(len(textstrings)/chunk_size),chunk_size),np.dtype('U100'))

  textstrings = get_text(text_data)
  textchunks = []
  chunk_idx = 0
  word_idx = 0
  for string in textstrings:
    for word in string.split():
      if word_idx == 0:
        textchunks.append('')
      if word_idx >= chunk_size and word=='<message_end>':
        chunk_idx += 1
        word_idx = 0
      else:
        if word != '<message_end>':
          textchunks[chunk_idx] += word + ' '
          word_idx += 1
        else:
          textchunks[chunk_idx] = textchunks[chunk_idx][:-1]+'. '
  return textchunks

def get_coherence(model, docs, metric='c_npmi'):
  # Preprocess Documents
  documents = pd.DataFrame({"Document": docs,
                            "ID": range(len(docs)),
                            "Topic": model.topics_})
  documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
  cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

  # Extract vectorizer and analyzer from BERTopic
  vectorizer = model.vectorizer_model
  analyzer = vectorizer.build_analyzer()


  # Extract features for Topic Coherence evaluation
  tokens = [analyzer(doc) for doc in cleaned_docs]
  dictionary = corpora.Dictionary(tokens)
  corpus = [dictionary.doc2bow(token) for token in tokens]

  # Extract words in each topic if they are non-empty and exist in the dictionary
  topic_words = []
  for topic in range(len(set(model.topics_))-model._outliers):
      words = list(zip(*model.get_topic(topic)))[0]
      words = [word for word in words if word in dictionary.token2id]
      topic_words.append(words)
  topic_words = [words for words in topic_words if len(words) > 0]

  # Evaluate Coherence
  coherence_model = CoherenceModel(topics=topic_words,
                                  texts=tokens,
                                  corpus=corpus,
                                  dictionary=dictionary,
                                  coherence=metric)
  coherence = coherence_model.get_coherence_per_topic(with_std=True)
  return coherence


#%%
with open('./Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
  data = pickle.load(input_file)

#Global variables
VERSION = 1
filename = 'BertModel_{}'.format(VERSION)
#transformer = 'multi-qa-MiniLM-L6-cos-v1'
transformer = 'all-MiniLM-L6-v2'
top_n_words = 10
n_gram_range = (1, 2)
min_topic_size = 10
nr_topics = None
low_memory = True

#Preprocess data
docs = chunkify(data,100)

#Text Embedding
embedding_model = SentenceTransformer(transformer)
if os.path.exists('/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_{}.pkl'.format(transformer)):
  with open('/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_{}.pkl'.format(transformer), "rb") as input_file:
    embeddings = pickle.load(input_file)
else:
  sentence_model = SentenceTransformer(transformer)
  embeddings = sentence_model.encode(docs, show_progress_bar=True)
  with open('/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_{}.pkl'.format(transformer), "wb") as output_file:
    pickle.dump(embeddings,output_file)

# UMAP or another algorithm that has .fit and .transform functions
umap_var = {'n_neighbors' : 15,
            'n_components': 5,
            'min_dist': 0.0,
            'metric':'cosine',
            'low_memory': low_memory
            }

umap_model =  UMAP(n_neighbors = umap_var['n_neighbors'],
                   n_components = umap_var['n_components'],
                   min_dist = umap_var['min_dist'],
                   metric = umap_var['metric'],
                   low_memory = low_memory)

# HDBSCAN or another clustering algorithm that has .fit and .predict functions and
# the .labels_ variable to extract the labels
hdbscan_var = {'min_cluster_size' : min_topic_size,
               'metric' : 'euclidean',
               'cluster_selection_method': 'eom',
               'prediction_data': True
               }

hdbscan_model = HDBSCAN(min_cluster_size = hdbscan_var['min_cluster_size'],
                        metric = hdbscan_var['metric'],
                        cluster_selection_method = hdbscan_var['cluster_selection_method'],
                        prediction_data = hdbscan_var['prediction_data'])

# Vectorizer
vectorizer_var = {'ngram_range' : n_gram_range,
                  'stop_words' : 'english',
                  'tokenizer' : None,
                  'analyzer' : None,
                  'min_df' : 1,
                  'max_df' : 1
                  }

vectorizer_model = CountVectorizer(ngram_range = vectorizer_var['ngram_range'],
                                   stop_words = vectorizer_var['stop_words'],
                                   tokenizer = vectorizer_var['tokenizer'],
                                   #analyzer = vectorizer_var['analyzer'],
                                   #min_df = vectorizer_var['min_df'],
                                   #max_df = vectorizer_var['max_df']
                                   )

# Topic modeling
ctfidf_var = {'bm25_weighting' : False,
              'reduce_frequent_words' : False}

ctfidf_model = ClassTfidfTransformer(bm25_weighting = ctfidf_var['bm25_weighting'],
                                     reduce_frequent_words = ctfidf_var['reduce_frequent_words']
                                     )

# Representation model
representation_var = {}

def representation_model(name=None):
  global representation_var
  if name == None:
    return BaseRepresentation()
  elif name == 'MMR':
    representation_var = {'diversity':0.1,
                          'top_n_words': top_n_words
                          }
    return MaximalMarginalRelevance(diversity=representation_var['diversity'],
                                    top_n_words=representation_var['top_n_words']
                                    )

# BERTopic model
topic_model = BERTopic(language = "english",
                        top_n_words = top_n_words,
                        n_gram_range = n_gram_range,
                        min_topic_size = min_topic_size,
                        nr_topics = nr_topics,
                        low_memory = low_memory,
                        calculate_probabilities = False,
                        seed_topic_list = None,
                        embedding_model = embedding_model,
                        umap_model = umap_model,
                        hdbscan_model = hdbscan_model,
                        vectorizer_model = vectorizer_model,
                        ctfidf_model = ctfidf_model,
                        representation_model = representation_model('MMR'),
                        verbose = True)

# Fit the model on a corpus
topics, probs = topic_model.fit_transform(documents=docs,
                                          embeddings=embeddings
                                          )
# Calculate coherence
coherence = get_coherence(topic_model,docs)

#Save model and settings
variables = {'filename' : filename,
             'coherence' : coherence,
             'transformer': transformer,
             'top_n_words': top_n_words,
             'n_gram_range': n_gram_range,
             'min_topic_size': min_topic_size,
             'nr_topics': nr_topics,
             'low_memory': low_memory,
             'umap_model' : umap_var,
             'hdbscan_model' : hdbscan_var,
             'vectorizer_model' : vectorizer_var,
             'ctfidf_model' : ctfidf_var,
             'representation_model' : representation_var
             }

topic_model.save('./Data/{}'.format(filename))
with open('./Data/{}_vars.plk'.format(filename), "wb") as output_file:
  pickle.dump(variables, output_file)
  
#%% Create table

with open('./Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
  data = pickle.load(input_file)
docs = chunkify(data,100)
topic_model = BERTopic.load('./Data/BertModel_10')

topic_df = pd.DataFrame(topic_model.get_topic_info())
coherence = get_coherence(topic_model, docs)
topic_results = {'Topic Label':[],
                 'Key words':[],
                 'Document excerpt':[],
                 'Topic Probability':[],
                 'Coherence':[],
                 'Std error':[],
                 'Av. valence':[]}
idx = 0
# Get main topic in each document
for index, row in topic_df.iterrows():
    if row['Topic'] > -1 and row['Topic'] != 18:
      topic_results['Topic Label'].append(row['Topic'])
      topic_results['Key words'].append(', '.join(row['Representation']))
      topic_results['Document excerpt'].append('. '.join(row['Representative_Docs']))
      #topic_results['Topic Frequency'].append(row['Count'])
      topic_results['Topic Probability'].append(row['Count']/len(docs))
      topic_results['Coherence'].append(coherence[idx][0])
      topic_results['Std error'].append(coherence[idx][1])
      idx += 1
      val=[]
      for word in row['Representation']:
          val.append(np.nanmean(data.loc[data['post'].str.contains(word), 'valence']))
      topic_results['Av. valence'].append(np.nanmean(val))

