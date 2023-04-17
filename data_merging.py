import pandas as pd
import numpy as np
import requests

# Load in our CSV files from the Kaggle dataset
artists_data = pd.read_csv("data/artists-data.csv")
lyrics_data = pd.read_csv("data/lyrics-data.csv")
lyrics_data = lyrics_data.rename(columns={"ALink": "Link"})

# Filter and retain songs with English lyrics
english_lyrics_data = lyrics_data[lyrics_data["language"] == 'en']

# Merge the two datasets to get lyrics and artists together and drop the original genres column as it was very generic
songs = pd.merge(artists_data[["Artist", "Genres", "Link"]], english_lyrics_data[["SName", "Link", "Lyric"]], on="Link")
songs = songs.drop(["Link", "Genres"], axis=1)


def get_track_info(artist, track, api_key="564f3d633d008c5c7592375a9abda311"):
    """
    Calls Last.fm API to get genre of songs
    :param artist: Artist of the song
    :param track: Name fo the song
    :param api_key: API key for making the call
    :return: Genre of the song
    """
    url = 'https://ws.audioscrobbler.com/2.0/'
    params = {
        'method': 'track.getInfo',
        'artist': artist,
        'track': track,
        'api_key': api_key,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()

    try:
        genre = data['track']['toptags']['tag'][0]['name']
    except (KeyError, IndexError):
        genre = None

    return genre

# Gets the genre for each song
songs['Genre'] = songs.apply(lambda x: get_track_info(x["Artist"], x["SName"]), axis=1)


# Fill in missing genres with NaN values
songs['Genre'] = songs['Genre'].fillna(np.nan)

# Replaced NaN values with most popular genre for the artist
most_common_genres = songs.groupby('Artist')['Genre'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)

# Remove artists that had no genres in the API
artist_with_genres = {k: v for k, v in most_common_genres.items() if v == v}
artists_to_keep = list(artist_with_genres.keys())
filtered_df = songs[songs['Artist'].isin(artists_to_keep)]

# Save the file for preprocessing before training
filtered_df.to_csv("data/training_data.csv")

