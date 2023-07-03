#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import nltk
from nltk.tokenize import WordPunctTokenizer
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

# create Tokenizer, Lemmatizer and SentimentIntensityAnalyzer objects:
tk = WordPunctTokenizer()
lm = nltk.WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

with open('./Data/MASTER_all_data_preprocessed.pkl', "rb") as input_file:
    data = pickle.load(input_file)

colors = ["#990000",
          "#030F4F",
          "#79238E",
          "#E83F48",
          "#FC7634",
          "#F7BBB1",
          "#F6D04D",
          "#1FD082",
          "#008835",
          "#DADADA"]

#%% Histogram of data over time

sns.set_style(style=None, rc=None)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "10"

#plt.figure(figsize=(15,8))
plt.figure()
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
#hist = sns.displot(data=data, x="date posted", col="server",col_wrap=2, multiple="stack")
hist = sns.displot(data=data, x="date posted", col="server",col_wrap=2, hue="channel", multiple="stack", hue_order=['de2-tips-n-tricks',
                               'de2-prompt-help',
                               'dall-e-discussions',
                               'prompt-chat',
                               'sd-chat',
                               'prompting-help',
                               'general-chat',
                               'general',
                               'textual-inversion',
                               'prompting'])
#hist.fig.suptitle('Data collected over time', fontweight='bold')
hist.set_axis_labels('Date', 'Count of records')
hist.set_titles(col_template='{col_name}', row_template='{row_name}', size=12)
ax1, ax2, ax3, ax4 = hist.axes

#MJ
ax2.axvline(np.datetime64("2022-07-25 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Midjourney V3', color='lightgrey') #MJ V3
ax2.axvline(np.datetime64("2022-11-05 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Midjourney V4', color='grey') #MJ V4
ax2.axvline(np.datetime64("2023-03-15 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Midjourney V5', color='black') #MJ V5
ax2.legend()

#OpenAI
ax1.axvline(np.datetime64("2022-07-20 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='OpenAI Beta1', color='lightgrey') 
ax1.axvline(np.datetime64("2022-09-28 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='OpenAI Beta2', color='grey') 
ax1.axvline(np.datetime64("2022-11-03 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='OpenAI Release', color='black') 
ax1.legend()

#SD
ax3.axvline(np.datetime64("2022-08-22 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 1.4', color='lightgrey') #MJ V3
ax3.axvline(np.datetime64("2022-10-20 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 1.5', color='grey') #MJ V4
ax3.axvline(np.datetime64("2022-11-24 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 2.0', color='darkgrey') #MJ V5
ax3.axvline(np.datetime64("2022-12-07 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 2.1', color='black') #MJ V5
ax3.legend()

#rSD
ax4.axvline(np.datetime64("2022-08-22 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 1.4', color='lightgrey') #MJ V3
ax4.axvline(np.datetime64("2022-10-20 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 1.5', color='grey') #MJ V4
ax4.axvline(np.datetime64("2022-11-24 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 2.0', color='darkgrey') #MJ V5
ax4.axvline(np.datetime64("2022-12-07 00:00:00"), ymax=25000, ymin=0, linestyle=':', label='Stable Diffusion 2.1', color='black') #MJ V5
ax4.legend()

plt.savefig("./Plots/datacollected_overtime2.pdf", bbox_inches="tight")

#%% Distribution of Valence if question plot
import ptitprince as pt

# Draw a nested boxplot
plt.figure(figsize=(10,7))
colors = ["#990000",
          #"#030F4F",
          "#E83F48",
          "#FC7634",
          "#F7BBB1",
          "#F6D04D",
          "#1FD082",
          "#008835",
          "#DADADA"]
sns.set_palette(sns.color_palette(colors))
# Draw a nested violinplot and split the violins for easier comparison
#plt.suptitle('Distribution of Valence', fontweight='bold')
"""
sns.violinplot(data=data, x="valence", y="server", hue="is question",
               split=True, inner="box", linewidth=1, orient='h',
               palette={True: colors[3], False: colors[7]})
sns.despine(left=True)
"""
pt.half_violinplot( x = "valence", y = "server", data = data, palette = colors,
                      scale = "area", inner = None, orient = 'h')
sns.stripplot( x = "valence", y = "server", data = data, palette = colors, edgecolor = "white",
                 size = 1, jitter = 1, zorder = 0, orient = 'h')
ax = sns.boxplot( x = "valence", y = "server", data = data, color = "black", width = .1, zorder = 10,
               showcaps = True, boxprops = {'facecolor':'none', "zorder":10},
               showfliers=False, whiskerprops = {'linewidth':1, "zorder":10},
               saturation = 1, orient = 'h', dodge=False)
ax.set(xlabel='Valence score', ylabel='Server')
plt.savefig("./Plots/datacollected_valence.pdf", bbox_inches="tight")

#%% Distribution of Valence if question plot

# Draw a nested boxplot
plt.figure()
colors = ["#990000",
          #"#030F4F",
          "#E83F48",
          "#FC7634",
          "#F7BBB1",
          "#F6D04D",
          "#1FD082",
          "#008835",
          "#DADADA"]
sns.set_palette(sns.color_palette(colors))
g = sns.FacetGrid(data, col="is question", sharey=True, height=5, aspect=1.25)
g.map(pt.half_violinplot, "valence", "server", palette = colors,
                      scale = "area", inner = None, orient = 'h')
g.map(sns.stripplot, "valence", "server", palette = colors, edgecolor = "white",
                 size = 1, jitter = 1, zorder = 0, orient = 'h')
g.map(sns.boxplot, "valence", "server", color = "black", width = .1, zorder = 10,
               showcaps = True, boxprops = {'facecolor':'none', "zorder":10},
               showfliers=False, whiskerprops = {'linewidth':1, "zorder":10},
               saturation = 1, orient = 'h', dodge=False)
g.set_axis_labels('Valence score', 'Server')
g.set_titles(col_template='Text contains a question: {col_name}', size=12)
plt.savefig("./Plots/datacollected_valencequestion.pdf", bbox_inches="tight")


#%% Posts per user plot

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5),sharey=False)
#plt.suptitle('Count of records per user', fontweight='bold')
#palette = sns.color_palette("Blues", n_colors=1)
#palette.reverse()
#sns.set_theme(style="whitegrid")
hist = sns.countplot(data=data, dodge=False, x='username', order=pd.value_counts(data['username']).iloc[:250].index, color="#030F4F", ax=ax1)
#ax = hist.axes
#ax.axvline(np.mean(data['username']), ymax=3500, ymin=0, linestyle=':', label='Mean', color='grey') #MJ V3
#plt.yscale('log')
hist.set(xticklabels=[])  
hist.set(xlabel='User (top 250, descending)', ylabel='Count of records')
hist.tick_params(bottom=False)  # remove the ticks
#plt.savefig("/Users/kristineolgaard/Documents/DTU/Thesis/Thesis/Plots/datacollected_userposts.pdf", bbox_inches="tight")

perc = 0.90
ax2.plot(range(1,data['username'].nunique()+1), data['username'].value_counts(),'-', label='Message count per user')
ax2.plot(range(1,data['username'].nunique()+1),np.cumsum(data['username'].value_counts()),'-', label='Cummulative message count')
ax2.axhline(data['username'].value_counts()[0],linestyle=':',linewidth=1,color='grey')
ax2.axhline(perc*np.cumsum(data['username'].value_counts())[-1],linestyle='--',linewidth=1,color='black')
idx = np.argwhere(np.diff(np.sign(perc*np.cumsum(data['username'].value_counts())[-1] - np.cumsum(data['username'].value_counts())))).flatten()
ax2.axvline(idx,linestyle='--',linewidth=1,color='black')
ax2.text(data['username'].nunique()+1000, perc*np.cumsum(data['username'].value_counts())[-1]-30000, '90%')
ax2.text(idx-700, 500000, str(idx[0]))
plt.yscale('log')
# Put a legend below current axis
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
ax2.set_xlabel('User')
ax2.set_ylabel('Count of records (log)')
plt.savefig("./Plots/datacollected_userposts2.pdf", bbox_inches="tight")
plt.show()

#%% Reaction vs Valence plot

plt.figure(figsize=(15,4))
hist1 = sns.scatterplot(x="num. reactions", y="valence", size=1,
                color="#1FD082",
                data=data)
hist2 = sns.regplot(data=data, x="num. reactions", y="valence", scatter=False, color='black')
hist2.set(ylim=(-1.1, 1.1))
hist2.set(xlabel='Number of message reactions (log)', ylabel='Valence score')
plt.xscale('log')
plt.savefig("/Users/kristineolgaard/Documents/DTU/Thesis/Thesis/Plots/datacollected_reactions.pdf", bbox_inches="tight")
