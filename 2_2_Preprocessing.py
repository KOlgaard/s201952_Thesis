#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np

project_dir = './'

with open(project_dir + 'Data/MASTER_all_data_raw.pkl', "rb") as output_file:
  data = pickle.load(output_file)

data = data.replace({'server': {'r/Stable Diffusion':'r/StableDiffusion'}})
data = data.replace({'server':{'Stable Foundation (stability.ai)':'Stable Foundation'}})
data = data.replace({'reply to id':{np.nan:''}})
data = data.dropna()
data = data.drop_duplicates()

data.groupby('channel')['date posted'].max()
print("\nNumber of posts")
print(data.groupby('channel')['post id'].count())

print("\nNumber of unique users")
print(data.groupby('channel')['username'].nunique())

print("\nNumber of total unique users")
print(data['username'].nunique())

print("\nNumber of total posts")
print(data['post id'].count())


# Create a dictionary to map each unique name to a unique ID
name_to_id = {}
id_counter = 0
for name in data['username'].unique():
    id_str = str(id_counter + 1).zfill(6)  # pad with leading zeros
    name_to_id[name] = f'{id_str}'
    id_counter += 1

# Replace the usernames with their corresponding IDs
data['username'] = data['username'].apply(lambda x: name_to_id[x])

data = data.drop(columns=['link to post'])

with open(project_dir + 'MASTER_all_data_preprocessed.pkl', "wb") as output_file:
  pickle.dump(data, output_file)