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
import seaborn as sns

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

def train_bert(docs, filename, nr_topics, embedding_model, embeddings=None):
    # BERTopic model
    topic_model = BERTopic(nr_topics=nr_topics,
                           low_memory=True,
                           embedding_model=embedding_model
                           )

    # Fit the model on a corpus
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    topic_model.save('/content/drive/MyDrive/Colab Notebooks/' + filename)
    return topic_model

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
  coherence = coherence_model.get_coherence()
  return coherence

#%% Test sentence transformers

with open('./Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
    data = pickle.load(input_file)

topic_results = {'all-MiniLM-L12-v2':{},
                 'multi-qa-MiniLM-L6-cos-v1':{},
                 'multi-qa-distilbert-cos-v1':{},
                 'all-MiniLM-L6-v2':{},
                 'all-mpnet-base-v2':{}}

embeddings_files = {'all-MiniLM-L12-v2':"/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_all-MiniLM-L12-v2.pkl",
                    'multi-qa-MiniLM-L6-cos-v1':"/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_multi-qa-MiniLM-L6-cos-v1.pkl",
                    'multi-qa-distilbert-cos-v1':"/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_multi-qa-distilbert-cos-v1.pkl",
                    'all-MiniLM-L6-v2':"/content/drive/MyDrive/Colab Notebooks/BertModel_embeddings_all-MiniLM-L6-v2.pkl",
                    'all-mpnet-base-v2': ""}

topics = [60,40,20,15,12,None]

text_data = chunkify(data,100)
for topic in topics:
  for senttrans, embeddings in embeddings_files.items():
    print("Sentence Transformer: {}, Topic: {}".format(senttrans, topic))
    if topic_results[senttrans].get(topic) == None:
      print("Running...")
      embedding_model = SentenceTransformer(senttrans)
      try:
        with open(embeddings, "rb") as input_file:
          embeddings = pickle.load(input_file)
      except:
          embeddings = embedding_model.encode(text_data, show_progress_bar=True)
          with open('./Data/BertModel_embeddings_{}.pkl'.format(senttrans), "wb") as output_file:
            pickle.dump(embeddings, output_file)
          embeddings_files[senttrans] = './Data/BertModel_embeddings_{}.pkl'.format(senttrans)
      if not os.path.exists('./Data/bert_model_t'+str(topic)+'_'+senttrans+'_v2'):
        model = train_bert(text_data, 'bert_model_t'+str(topic)+'_'+senttrans+'_v2' ,topic, embedding_model=embedding_model, embeddings=embeddings)
      else:
        model = BERTopic.load('./Data/bert_model_t'+str(topic)+'_'+senttrans+'_v2')
      topic_results[senttrans][str(topic)] = get_coherence(model, text_data)
      try:
        fig = model.visualize_topics()
        fig.write_image('./Plots/Tests_{}_{}.pdf'.format(str(topic),senttrans))
      except:
        print("ERROR: Sentence Transformer: {}, Topic: {} could not be exported as graph".format(senttrans, topic))
      print("Done!")
    else:
      print("Already exists!")

#%% Plot heatmap

topic_results2 = pd.DataFrame(topic_results).round(3)
topic_results2 = topic_results2[['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'multi-qa-MiniLM-L6-cos-v1', 'multi-qa-distilbert-cos-v1']]
topic_results2 = topic_results2.reindex(['None','60','40','20','15','12'])

plt.figure()
sns.heatmap(data=topic_results2,annot=True,linewidth=.5,fmt=".3f",cmap="coolwarm", cbar=False)
plt.xticks(rotation=45)
plt.ylabel("Reduced number of topics")
plt.xlabel("Sentence Transformer")
plt.savefig('./Plots/fig_TopicCoherence_heatmap_bert.pdf',bbox_inches='tight')
plt.show()
