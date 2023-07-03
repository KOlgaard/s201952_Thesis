#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import os.path
import os
import nltk, re, pprint
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams
from matplotlib import pyplot as plt
from nltk.corpus import PlaintextCorpusReader
from wordcloud import WordCloud


tk = WordPunctTokenizer()
lm = nltk.WordNetLemmatizer()

project_dir = './'

#%% Convert dataframes to text
for filename in os.listdir(project_dir):
    if filename.endswith('all_data_preprocessed.pkl'):
        print(filename)
        with open(project_dir + filename, "rb") as input_file:
            df = pickle.load(input_file)
            input_file.close()
        
        df_channels = df.groupby('channel')

        for name, group in df_channels:
            print('|\n---'+name)
            server = group['server'].unique()[0].replace('/','')
            #convert to string
            allposts = '. '.join(group.post.tolist())
            if filename.startswith('FAQs'):
                allposts = '. '.join(group.comments.tolist())
            # Tokenize text
            token = tk.tokenize(allposts)
            # Remove stop-words, and make lower-case
            token = [w.lower() for w in token if not w.lower() in set(stopwords.words('english'))]
            # Remove punctuation
            #token = [w for w in token if not w in ['.',',','-',':',';',".'",'."',"'",'"','",','!','?','/']]
            token = [w for w in token if w.isalpha()]
            #Lemmatize
            lemma = [lm.lemmatize(w) for w in token]
            #save page contents for each character as txt file
            text = str(' '.join(lemma).encode('utf8'))
            if filename.startswith('Qs'):
                text_file = project_dir + 'Text/Questions/'+server+'_'+name+'.txt'
            else:
                text_file = project_dir + 'Text/All/'+server+'_'+name+'.txt'
            text_file = open(text_file,"w+")
            text_file.write(text)
            text_file.close()
    
#%% Frequency distribution
#Create corpus
text_corpus = PlaintextCorpusReader(project_dir + 'Text/Questions/', os.listdir('./Text/Questions'),encoding='latin-1')
text = nltk.Text(text_corpus.words())
#text = nltk.bigrams(text_corpus.words())

#Plot Frequency distribution
fig = plt.figure(figsize=(15,5))
plt.suptitle('Frequency distribution of bigrams', fontweight='bold')
plt.title('Questions')
fdist1 = nltk.FreqDist(text)
fdist1.plot(75, cumulative=True)
plt.show()


#%% TF-IDF


freq_matrix = {}
tc_matrix = {}

# ----- TF ----- #
def TF(data):
    print("\nTF")
    with open(project_dir + 'MASTER_all_data_preprocessed', "rb") as input_file:
        data = pickle.load(input_file)
    freq_table = {}
    tf_table = {}
    word_per_text_table = {}
    #Calculate the Term Frequency of each term in each race document
    for text in data['post'].to_list():
        #Previous scripts have not generated text for all characters and races
        #Check that text exists for that race
        if len(text) > 0:
            #Text is formatted as str. Turn into list
            token = tk.tokenize(text)
            # Remove stop-words, and make lower-case
            token = [w.lower() for w in token if not w.lower() in set(stopwords.words('english'))]
            # Remove punctuation
            #token = [w for w in token if not w in ['.',',','-',':',';',".'",'."',"'",'"','",','!','?','/']]
            token = [w for w in token if w.isalpha()]
            #Lemmatize
            words = [lm.lemmatize(w) for w in token]
            #Count the term frequency for that race, and add to dictionary
            for word in words:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
                    
            for word, count in freq_table.items():
                if word in word_per_text_table:
                    word_per_text_table[word] += 1
                else:
                    word_per_text_table[word] = 1
    
    #Calculate term frequency, where
    #TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
    for word, count in freq_table.items():
        tf_table[word] = count / len(freq_table)
    
    #Sort dictionary to print top 5
    TF_ordered = sorted(tf_table.items(), key=lambda x: x[1], reverse=True)
    print("   Top 5 terms of: {}".format(TF_ordered[1:5]))
    return tf_table, word_per_text_table

def IDF(data):
    #For each term, calculate the Inverse Document Frequency, where
    #IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    tf_table, word_per_text_table = TF(data)
    idf_table = {}
    for word in word_per_text_table.keys():
        idf_table[word] = log10(len(data) / float(word_per_text_table[word]))
    #print("IDF of terms: {}".format(idf_table))
    return tf_table, idf_table

def TF_IDF(data):
    print("\nTF-IDF")
    tf_table, idf_table = IDF(data)
    # --- TF-IDF --- #
    tf_idf_table = {}
    #Iterate through each term in the IDF table
    for word, idf in idf_table.items():
        #If this word exists in the race text, calculate TF-IDF of that term
        if word in tf_table:
            tf_idf_table[word] = tf_table[word]*idf_table[word]
    
    #Order by importance, and print top 5
    tf_idf_ordered = sorted(tf_idf_table.items(), key=lambda x: x[1], reverse=True)
    print("   Top 5 terms: {}".format(tf_idf_ordered[0:4]))
    return 



#%% Build wordclouds

tc_table_all = {}

# -------------- WORDCLOUD BUILDER -------------- #
#Iterate through each race and their term count (not normalized)
for month, tc_table in tc_matrix.items():
    #Iterate through each race and their term count (not normalized)
    for word, count in tc_table.items():
        try:
            tc_table_all[word] += int(count)
        except:
            tc_table_all[word] = int(count)
    
    
term_str = ''
#Iterate through each term in the IDF table
for word, freq in idf_table.items():
    #If the word in the idf table is in the race text
    if word in freq_table.keys():
        #Set the number of occurences of the term to the rounded IDF times the term count
        term_repeat = int(round(freq))*freq_table[word]
        #Add term to race word string according to the adjusted term count
        for i in range(term_repeat):
            term_str = term_str+' '+word


#Generate word cloud
wordcloud = WordCloud(
                background_color='white',
                width=1800,
                height=1400,
                collocations=False).generate(term_str)

#Show word cloud for each race
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.suptitle("Midjourney Wordcloud",fontweight='bold')
plt.axis('off')
plt.show()

