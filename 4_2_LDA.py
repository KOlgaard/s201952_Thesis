#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from math import sqrt, log10
import unidecode
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import re
import os
import pickle
import gensim
from gensim.corpora.dictionary import Dictionary
import spacy
nlp = spacy.load('en_core_web_sm')
#import pyLDAvis.gensim

# create Tokenizer, Lemmatizer and SentimentIntensityAnalyzer objects:
tk = WordPunctTokenizer()
lm = nltk.WordNetLemmatizer()

project_dir = './'

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

def preprocess(data):
    textstrings = []
    manual_remove = ['midjourney', 'mj', 'dall', 'dalle', 'dalle2', 'de2', 'openai', 'stable', 'diffusion','sd']

    for post in data:
        # Remove links
        post = re.sub(r'http\S+', '', post, flags=re.MULTILINE)
        post = re.sub(r'([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)','', post)
        # Tokenize text
        token = nlp(post)
        # Remove punctuation
        token = (w for w in token if w.text.isalnum() and not w.text.isdigit())
        # Remove stop words
        token = (w for w in token if not w.is_stop)
        # Lemmatize
        lemma = [w.lemma_ for w in token]
        # Remove single characters
        lemma = [w.lower() for w in lemma if len(w)>1]
        # Remove custom words
        lemma = [w for w in lemma if w not in manual_remove]
        textstrings.append(lemma)
    return textstrings

#%% Run LDA model

with open(project_dir + 'Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
    data = pickle.load(input_file)

textchunks = chunkify(data, chunk_size=100)
textchunks = preprocess(textchunks)
dictionary=Dictionary(textchunks)
dictionary.filter_extremes(no_below=5, no_above=0.5)
training_set = [dictionary.doc2bow(text) for text in textchunks]

topics=12
alpha=0.9
beta='symmetric'

model_lda = gensim.models.ldamulticore.LdaMulticore(
                        corpus=training_set,
                        num_topics=topics,
                        id2word=dictionary,
                        eval_every = 1,
                        #random_state=100,
                        chunksize=100,
                        passes=10,
                        alpha=alpha,
                        eta=beta) 

gensim.models.LdaMulticore.save(model_lda, 'Data/LDA_Model_K12')


#%% Create table

topic_results = {'Topic':[],'Key words':[], 'Topic Frequency':[], 'Topic Probability':[], 'Coherence':[], 'Av. valence':[]}
doc_topics = []
# Get main topic in each document
for i, row in enumerate(model_lda[training_set]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if prop_topic > (1/len(row)):  # => dominant topic
                wp = model_lda.show_topic(topic_num)[:15]
                topic_keywords = ", ".join([word for word, prop in wp])
                doc_topics.append([i, int(topic_num), round(prop_topic,3), topic_keywords])
            else:
                break

doc_topics_df = pd.DataFrame(doc_topics, columns = ['Document ID','Topic', 'Perc_Contribution', 'Topic_Keywords'])
dom_topics_df = doc_topics_df.groupby(['Topic','Topic_Keywords'])['Perc_Contribution'].size().reset_index()
dom_topics_df['Perc_Contribution2'] = dom_topics_df['Perc_Contribution']/len(model_lda[training_set])
dom_topics_df = dom_topics_df.rename(columns={'Perc_Contribution':'Frequency','Perc_Contribution2':'Perc_Contribution'})

for rank, topic in model_lda.show_topics(num_topics=-1, formatted=False, num_words= 15):
    for words, c_npmi in model_lda.top_topics(corpus=training_set, texts=textchunks,
                                        coherence='c_npmi', topn=15):
        if [w[0] for w in topic] == [w[1] for w in words]:
            topic_results['Topic'].append(rank)
            topic_results['Key words'].append([w[0] for w in topic])
            topic_results['Topic Frequency'].append(dom_topics_df.loc[dom_topics_df['Topic']==rank]['Frequency'].item())
            topic_results['Topic Probability'].append(dom_topics_df.loc[dom_topics_df['Topic']==rank]['Perc_Contribution'].item())
            topic_results['Coherence'].append(c_npmi)
            val=[]
            for word,_ in topic:
                val.append(np.mean(data.loc[data['post'].str.contains(word), 'valence']))
            topic_results['Av. valence'].append(np.mean(val))
            
topic_df = pd.DataFrame(topic_results)
topic_df['Topic'] += 1
topic_df['Key words'] = topic_df['Key words'].apply(lambda x: ', '.join(map(str, x)))
topic_df = topic_df.sort_values(by=['Topic Frequency'], ascending=False).round(3)
