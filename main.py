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

print(spotify_df.head())

sorted_data = spotify_df.drop(['Unnamed: 0', ])
sorted_data.drop_duplicates(subset=['track_id'], inplace=True)
