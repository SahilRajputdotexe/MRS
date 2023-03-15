import re
import sys
import itertools
import pandas as pd
import numpy as np
import json

from skimage import io
from sklearn.feature_extraction.text import TfidfVectorizer
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

spotify_df['popularity_5'] = spotify_df['popularity'].apply(lambda x: int(x/5))
float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values


def ohe_prep(df, column, new_name):
    """
    Create One Hot Encoded features of a specific column

    Parameters:
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used

    Returns:
        tf_df: One hot encoded features
    """

    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df


def create_feature_Set(df, float_cols):

    # tfiidf genre list
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['track_genre'])
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre'+'|' + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)

    # OHE PREP
    pop_ohe = ohe_prep(df, 'popularity_5', 'pop')*0.15

    floats = df[float_cols].reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(
        floats), columns=floats.columns) * 0.2

    final = pd.concat([genre_df, floats_scaled, pop_ohe], axis=1)
    final['track_id'] = df['track_id'].values

    return final


feature_set = create_feature_Set(spotify_df, float_cols=float_cols)

client_id = "7e439f5bb1894d09a1bb330d90f5f589"
client_secret = "f5389add4e8b4d08be7e6aad586685e3"


scope = "user-library-read"
# if len(sys.argv) > 1:
#     username = sys.argv[1]
# else:
#print("Usage: %s username" % (sys.argv[0],))
# sys.exit()

auth_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

token = util.prompt_for_user_token(
    scope, client_id=client_id, client_secret=client_secret, redirect_uri="https://localhost:8881/")
sp = spotipy.Spotify(auth=token)

# user playlsts
id_name = {}
list_photo = {}
for i in sp.current_user_playlists()['items']:
    id_name[i['name']] = i['uri'].split(':')[2]
    list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']


def create_neccesary_playlist(playlist_name, id_dic, df):
    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        # print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artists'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id']
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])
    playlist = playlist[playlist['id'].isin(
        df['track_id'].values)].sort_values('date_added', ascending=True)
    return playlist


def visualize_songs(df):
    temp = df['url'].values
    plt.figure(figsize=(15, int(0.625*len(temp))))
    columns = 5

    for i, url in enumerate(temp):
        plt.subplot(int(len(temp)/columns+1), columns, i+1)
        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color='w', fontsize=0.1)
        plt.yticks(color='w', fontsize=0.1)
        plt.xlabel(df['name'].values[i], fontsize=12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()


user_playlist = create_neccesary_playlist('Updowntown', id_name, spotify_df)

# visualize_songs(user_playlist)

print(feature_set)
print(user_playlist)


def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(
        playlist_df['id'].values)]
