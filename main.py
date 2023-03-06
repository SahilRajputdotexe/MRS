import re
import sys
import itertools
import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

spotify_df = pd.read_csv('D:/cpp/MRS/dataset.csv')


sorted_data = spotify_df.drop(['Unnamed: 0'], axis=1)
sorted_data.drop_duplicates(subset=['track_id'], inplace=True)


spotify_df['artists'] = spotify_df['artists'].apply(str)


spotify_df['artists_upd'] = spotify_df['artists'].apply(
    lambda x: re.findall(r"([^']*)", x))

for i in range(spotify_df.shape[0]):
    spotify_df['artists_upd'][i] = spotify_df['artists_upd'][i][0].split(
        ';')

spotify_df = spotify_df.explode('artists_upd')
spotify_df['artists_song'] = spotify_df.apply(
    lambda row: row['artists_upd']+str(row['track_name']), axis=1)
spotify_df.drop_duplicates('artists_song', inplace=True)
print(spotify_df)
