#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import os.path
import os
import nltk, re, pprint
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams
import unidecode
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import networkx as nx
from nltk.corpus import PlaintextCorpusReader
from math import log, log10
import gensim
from nltk.test.gensim_fixt import setup_module
setup_module()
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from multiprocessing import Process, freeze_support
import spacy
nlp = spacy.load('en_core_web_sm')
import seaborn as sns
#from bertopic import BERTopic

project_dir = './'

#%% definne helper functions

def assymetric_prior(num_topics):
    prior = np.fromiter(
                    (1.0 / (i + np.sqrt(num_topics)) for i in range(num_topics)),
                    dtype=float, count=num_topics,
                )
    return prior.mean()

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

def eval_model(model, corpus, dictionary, metrics):
    coherence = []
    for metric in metrics:
        coherence_model_lda = CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence=metric)
        coherence.append(coherence_model_lda.get_coherence())
    return coherence

def trainandtest_LDA(training_corpus, corpus, dictionary, topics=25, alpha=0.01, beta=0.1, metrics=['c_npmi']):
    print('--- training model...')
    #model = gensim.models.ldamulticore.LdaMulticore(
    model = gensim.models.ldamodel.LdaModel(
                            corpus=training_corpus,
                            num_topics=topics,
                            id2word=dictionary,
                            eval_every = 1,
                            #random_state=100,
                            chunksize=100,
                            passes=10,
                            alpha=alpha,
                            eta=beta)
    # Compute Coherence Score
    print('--- model trained successfully!')
    print('--- evaluating coherence...')
    evaluation = eval_model(model, corpus, dictionary, metrics)
    return evaluation
    
#%% Hyperparameter gridsearch

with open(project_dir + 'Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
    data = pickle.load(input_file)


textchunks = chunkify(data, chunk_size=100)
textchunks = preprocess(textchunks)
dictionary=Dictionary(textchunks)
dictionary.filter_extremes(no_below=5, no_above=0.5)
training_set = [dictionary.doc2bow(text) for text in textchunks]


# Can take a long time to run
START_AT = 0

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Number of topics
topics=list(np.arange(4, 104, 4))

# Metrics
metric=['c_v', 'c_npmi', 'u_mass']

if START_AT != 0:
    idx = START_AT
    new_start_topic = topics[START_AT // (len(alpha)*len(beta))] 
    # Number of topics
    topics=list(np.arange(new_start_topic, 104, 4))
    with open(project_dir + 'coherence_df_extra.pkl', "rb") as input_file:
        coherence_dict = pickle.load(input_file)
    coherence_df = pd.DataFrame(coherence_dict, columns=['Topics', 'Alpha', 'Beta', 'Coherence c_npmi'])
    coherence_df_cropped = coherence_df.iloc[:START_AT]
    model_results = coherence_df_cropped.to_dict('list')
else:
    idx = 0
    # initialise model_results
    model_results = {'Topics': [],
                    'Alpha': [],
                    'Beta': [],
                    'c_v': [],
                    'c_NPMI': [],
                    'c_UMass': []
                    }

total_calc = len(list(np.arange(4, 104, 4)))*len(alpha)*len(beta)

# Can take a long time to run
for k in topics:
    for a in alpha:
        for b in beta:
                print('\n{}/{}  Topics: {}, Alpha = {} and Beta = {}'.format(idx, total_calc, k, a, b))
                cv = trainandtest_LDA(training_set, textchunks, dictionary, alpha=a, beta=b, topics=k, metrics=metric)
                # Save the model results
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['c_v'].append(cv[0])
                model_results['c_NPMI'].append(cv[1])
                model_results['c_UMass'].append(cv[2])
                with open(project_dir + 'Data/LDA_hyperparameters.pkl', "wb") as input_file:
                    pickle.dump(model_results, input_file)
                print('Coherence scores: c_v={}, c_npmi={}, c_umass={}'.format(cv[0],cv[1],cv[2]))
                idx += 1
c_dataframe = pd.DataFrame(model_results)
if __name__ == '__main__':
    freeze_support()

#%% Plot in 3D (scatter)

with open(project_dir + 'Data/LDA_hyperparameters.pkl', "rb") as input_file:
    coherence_df = pickle.load(input_file)

measures = ['c_v','c_NPMI','c_UMass']

fig,axs = plt.subplots(1, 3, layout='constrained',figsize=plt.figaspect(0.5))
for idx, coherence in enumerate(measures):
    coherence_df = coherence_df[pd.to_numeric(coherence_df['Beta'], errors='coerce').notnull()]
    coherence_df = coherence_df[pd.to_numeric(coherence_df['Alpha'], errors='coerce').notnull()]
    # Load and format data
    z = coherence_df['Topics']
    x = coherence_df['Beta']
    y = coherence_df['Alpha']
    size = ((coherence_df[coherence] - np.min(coherence_df[coherence])) * (100 - 1) / (np.max(coherence_df[coherence]) - np.min(coherence_df[coherence]))) + 1

    ax = plt.subplot(1, 3, idx+1, projection='3d')
    p = ax.scatter3D(x, y, z, s=size, c=size, cmap = cm.coolwarm)
    ax.set_xlabel('Beta', fontsize='small')
    ax.set_ylabel('Alpha', fontsize='small')
    ax.set_zlabel('Topics', fontsize='small')
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    if coherence == 'c_v':
        ax.set_title('$c_{v}$', fontweight='bold')
    elif coherence == 'c_UMass':
        ax.set_title('$c_{UMass}$', fontweight='bold')
    else:
        ax.set_title('$c_{NPMI}$', fontweight='bold')
    ax.set_box_aspect(aspect = (1,1,1.5))
    ax.tick_params(axis='both', which='major', labelsize=8, pad=0)

cbar = plt.colorbar(p, shrink=0.5, aspect=15, pad=0.15)
cbar.set_label('Normalised Coherence')
plt.savefig('Plots/fig_TopicCoherence_all.pdf')
plt.show()

#%% Plot in 3D (spline grid)

for coherence in measures:
    beta = coherence_df.groupby(['Topics','Beta'])[coherence].mean().reset_index()
    #beta['Beta'] = beta['Beta'].mask(beta['Beta'] == 'symmetric', 1/ beta['Topics'])
    beta_sym = beta.loc[beta['Beta'] == 'symmetric'].copy()
    beta_sym['Beta'] = 1/ beta_sym['Topics']
    beta = beta[pd.to_numeric(beta['Beta'], errors='coerce').notnull()]
    beta = beta.sort_values(['Topics','Beta'],ascending=True)

    alpha = coherence_df.groupby(['Topics','Alpha'])[coherence].mean().reset_index()
    #alpha['Alpha'] = alpha['Alpha'].mask(alpha['Alpha'] == 'symmetric', 1/ alpha['Topics'])
    alpha_sym = alpha.loc[alpha['Alpha'] == 'symmetric'].copy()
    alpha_sym['Alpha'] = 1/ alpha_sym['Topics']
    alpha['Assymetric'] = alpha['Topics'].apply(assymetric_prior) 
    #alpha['Alpha'] = alpha['Alpha'].mask(alpha['Alpha'] == 'asymmetric', alpha['Assymetric'])
    alpha_asym = alpha.loc[alpha['Alpha'] == 'asymmetric'].copy()
    alpha_asym['Alpha'] = alpha_asym['Assymetric']
    alpha = alpha[pd.to_numeric(alpha['Alpha'], errors='coerce').notnull()]
    alpha = alpha.sort_values(['Topics','Alpha', coherence],ascending=True)

    # Load and format data
    z = np.array(beta.groupby(['Topics'])[coherence].apply(list).reset_index()[coherence].to_list())
    nrows, ncols = z.shape
    x = np.linspace(beta['Beta'].min(), beta['Beta'].max(), ncols)
    y = np.linspace(beta['Topics'].min(), beta['Topics'].max(), nrows)
    x, y = np.meshgrid(x, y)

    #region = np.s_[5:50, 5:50]
    #x, y, z = x[region], y[region], z[region]

    # Set up plot
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    ax.set_title('Beta prior', fontsize='medium')
    ax.set_xlabel('Beta $β$', fontsize='small')
    ax.set_ylabel('Num. topics $K$', fontsize='small')
    ax.set_zlabel('Coherence $C$', fontsize='small')
    surf1 = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, shade=False, alpha=0.75)
    ax.plot3D(np.repeat(beta['Beta'].max()+0.05, nrows), beta_sym['Topics'].to_list(), beta_sym[coherence].to_list(), color='black', linestyle='--', dashes=(2, 1), label='symmetric')
    ax.view_init(azim=135)
    plt.legend()
    z = np.array(alpha.groupby(['Topics'])[coherence].apply(list).reset_index()[coherence].to_list())
    x = np.linspace(alpha['Alpha'].min(), alpha['Alpha'].max(), ncols)
    y = np.linspace(alpha['Topics'].min(), alpha['Topics'].max(), nrows)
    x, y = np.meshgrid(x, y)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, shade=False, alpha=0.75)

    ax.plot3D(np.repeat(alpha['Alpha'].max()+0.05, nrows), alpha_asym['Topics'].to_list(), alpha_asym[coherence].to_list(), color='grey', linestyle='--', dashes=(2, 1),label='asymmetric')
    ax.plot3D(np.repeat(alpha['Alpha'].max()+0.05, nrows), alpha_sym['Topics'].to_list(), alpha_sym[coherence].to_list(),color='black', linestyle='--', dashes=(2, 1),label='symmetric')
    #ax.invert_xaxis()
    #ax.contour(x, y, z, zdir='x', offset=alpha['Alpha'].max()+0.05, cmap=cm.coolwarm, levels=4)
    #ax.contour(x, y, z, zdir='y', offset=0, cmap=cm.coolwarm_r,levels=4)

    plt.legend()
    ax.set_title('Alpha prior', fontsize='medium')
    ax.set_xlabel('Alpha $α$', fontsize='small')
    ax.set_ylabel('Num. topics $K$', fontsize='small')
    ax.set_zlabel('Coherence $C$', fontsize='small')
    if coherence == 'c_v':
        plt.suptitle('$C_{v}$', fontweight='bold')
    elif coherence == 'c_UMass':
        plt.suptitle('$C_{UMass}$', fontweight='bold')
    else:
        plt.suptitle('$C_{NPMI}$', fontweight='bold')
    ax.view_init(azim=135)
    plt.savefig('./Plots/fig_TopicCoherence_priors_{}.pdf'.format(coherence))
    plt.show()
    
#%% Plot heatmap

plt.rcParams.update({'font.size': 10})

for coherence in measures:
    beta = coherence_df.groupby(['Topics','Beta'])[coherence].mean().reset_index()
    #beta['Beta'] = beta['Beta'].mask(beta['Beta'] == 'symmetric', 1/ beta['Topics'])
    beta_sym = beta.loc[beta['Beta'] == 'symmetric'].copy()
    beta_sym['Beta'] = 1/ beta_sym['Topics']
    beta = beta[pd.to_numeric(beta['Beta'], errors='coerce').notnull()]
    beta = beta.sort_values(['Topics','Beta'],ascending=True)

    alpha = coherence_df.groupby(['Topics','Alpha'])[coherence].mean().reset_index()
    #alpha['Alpha'] = alpha['Alpha'].mask(alpha['Alpha'] == 'symmetric', 1/ alpha['Topics'])
    alpha_sym = alpha.loc[alpha['Alpha'] == 'symmetric'].copy()
    alpha_sym['Alpha'] = 1/ alpha_sym['Topics']
    alpha['Assymetric'] = alpha['Topics'].apply(assymetric_prior) 
    #alpha['Alpha'] = alpha['Alpha'].mask(alpha['Alpha'] == 'asymmetric', alpha['Assymetric'])
    alpha_asym = alpha.loc[alpha['Alpha'] == 'asymmetric'].copy()
    alpha_asym['Alpha'] = alpha_asym['Assymetric']
    alpha = alpha[pd.to_numeric(alpha['Alpha'], errors='coerce').notnull()]
    alpha = alpha.sort_values(['Topics','Alpha', coherence],ascending=True)

    beta = beta.astype(float).round(2)
    beta_ar = beta.pivot("Topics", "Beta", coherence)
    fig = plt.figure(figsize=plt.figaspect(0.40))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1 = sns.heatmap(data=beta_ar,annot=True,linewidth=.5,fmt=".2f",cmap="coolwarm", cbar=False)
    ax1.invert_xaxis()
    
    alpha = alpha.astype(float).round(2)
    alpha_ar = alpha.pivot("Topics", "Alpha", coherence)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(data=alpha_ar,annot=True,linewidth=.5,fmt=".2f",cmap="coolwarm", cbar=False)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    ax2.invert_xaxis()

    plt.savefig('./Plots/fig_TopicCoherence_heatmap_{}.pdf'.format(coherence))
    plt.show()