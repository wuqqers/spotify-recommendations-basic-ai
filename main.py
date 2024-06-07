import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import json
import random
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load environment variables from .env file
load_dotenv()

# Spotify API credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

# Required scope
scope = 'user-library-read playlist-read-private user-top-read playlist-modify-private playlist-modify-public'

# File to store recommended songs
recommended_songs_file = "recommended_songs.json"

# CSV file to store the dataset
dataset_file = "playlist_dataset.csv"

def fetch_audio_features(sp, track_ids):
    features = []
    track_names = []
    total_tracks = len(track_ids)

    for start in range(0, total_tracks, 100):
        end = start + 100
        batch_ids = track_ids[start:end]
        try:
            batch_features = sp.audio_features(batch_ids)
            if batch_features:
                features.extend(batch_features)
                for track_id in batch_ids:
                    track = sp.track(track_id)
                    track_names.append(track['name'])
        except Exception as e:
            print(f"Error: {e}. Track IDs: {batch_ids}")
            continue

    return features, track_names

def save_dataset(features, track_names):
    df = pd.DataFrame(features)
    df['track_name'] = track_names
    if os.path.exists(dataset_file):
        df_existing = pd.read_csv(dataset_file)
        df = pd.concat([df_existing, df]).drop_duplicates(subset=['id'], keep='last')
    df.to_csv(dataset_file, index=False)

def get_liked_songs(sp):
    liked_songs = []
    try:
        tracks = sp.current_user_saved_tracks()
        for item in tracks['items']:
            track = item['track']
            liked_songs.append(track['id'])
    except spotipy.SpotifyException as e:
        print(f"Error: {e}")

    return liked_songs

def get_playlist_tracks(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    track_ids = [track['track']['id'] for track in tracks]
    return track_ids

def create_training_data(sp, track_ids):
    training_data = []
    features, track_names = fetch_audio_features(sp, track_ids)
    save_dataset(features, track_names)
    valid_features = [f for f in features if f is not None]
    for i, feature in enumerate(valid_features):
        feature['track_name'] = track_names[i]
        feature['language'] = detect(track_names[i])
        training_data.append(feature)
    return training_data

def train_model(training_data, model=None):
    df = pd.DataFrame(training_data)
    X = df.drop(['track_name', 'language', 'id', 'track_href', 'analysis_url', 'uri', 'type'], axis=1)
    y = df['language'].apply(lambda x: 1 if x == 'tr' else 0)  # Binary classification for Turkish and other languages
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if model is None:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # For binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=2)

    model.save("language_model.keras")
    return model

def append_new_recommendations(recommendations_info):
    try:
        with open(recommended_songs_file, "r", encoding="utf-8") as f:
            previous_recommendations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_recommendations = {"new_recommendations": []}

    new_recommendations = []
    for rec_info in recommendations_info:
        track_info = f"{rec_info[0]} - {rec_info[1]}"
        if track_info not in previous_recommendations['new_recommendations']:
            new_recommendations.append(track_info)
            previous_recommendations['new_recommendations'].append(track_info)

    with open(recommended_songs_file, "w", encoding="utf-8") as f:
        json.dump(previous_recommendations, f, ensure_ascii=False, indent=4)

def create_or_update_playlist(sp, playlist_name, track_ids):
    user_id = sp.current_user()['id']
    playlists = sp.current_user_playlists()
    playlist = None
    for item in playlists['items']:
        if item['name'] == playlist_name:
            playlist = item
            break

    if playlist:
        # If playlist exists, add tracks to the existing playlist
        sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids)
        print(f"Added recommended songs to the existing playlist '{playlist_name}'.")
    else:
        # If playlist doesn't exist, create a new playlist and add tracks
        sp.user_playlist_create(user_id, playlist_name, public=False)
        playlists = sp.current_user_playlists()
        for item in playlists['items']:
            if item['name'] == playlist_name:
                playlist = item
                break
        sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids)
        print(f"Created playlist '{playlist_name}' and added recommended songs.")

def main():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                                   client_secret=SPOTIPY_CLIENT_SECRET,
                                                   redirect_uri=SPOTIPY_REDIRECT_URI,
                                                   scope=scope))

    update_dataset = input("Do you want to update the dataset? (yes/no): ").lower()

    if update_dataset == 'yes':
        playlist_id = '5o7RjYinGfWcoF0miijZyI'
        track_ids = get_playlist_tracks(sp, playlist_id)
        features, track_names = fetch_audio_features(sp, track_ids)
        save_dataset(features, track_names)
        training_data = create_training_data(sp, track_ids)
        model = train_model(training_data)
    else:
        try:
            model = load_model("language_model.keras")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except:
            print("Error loading model or model not found. Please update the dataset first.")
            return

        # Repeated training with the existing model
        playlist_id = '5o7RjYinGfWcoF0miijZyI'
        track_ids = get_playlist_tracks(sp, playlist_id)
        training_data = create_training_data(sp, track_ids)
        model = train_model(training_data, model)

    liked_songs = get_liked_songs(sp)

    filter_genre = input("Do you want to filter recommendations by genre? (yes/no): ").lower()
    genre_filter = None
    if filter_genre == 'yes':
        genre_filter = input("Enter a genre (e.g., Rock, Pop, HipHop): ").lower()
        valid_genres = sp.recommendation_genre_seeds()['genres']
        if genre_filter not in valid_genres:
            print(f"Invalid genre '{genre_filter}'. Please choose from the following: {valid_genres}")
            return

    filter_language = input("Do you want to filter recommendations by language? (yes/no): ").lower()

    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError:
        print(f"{dataset_file} not found. Please update the dataset first.")
        return

    track_ids = df['id'].tolist()
    track_ids = [track_id for track_id in track_ids if track_id not in liked_songs]
    training_data = create_training_data(sp, track_ids)

    if filter_language == 'yes':
        language_filter = input("Enter 'tr' for Turkish or 'en' for English: ").lower()
        market_filter = 'TR' if language_filter == 'tr' else 'US'
    else:
        language_filter = None
        market_filter = None

    # Ensure seed tracks are selected from the dataset
    seed_tracks = df['id'].tolist()

    # Filter tracks by selected genre and language if necessary
    if genre_filter:
        filtered_tracks = []
        for track_id in seed_tracks:
            track_info = sp.track(track_id)
            artist_id = track_info['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            artist_genres = artist_info['genres']
            if genre_filter in artist_genres:
                filtered_tracks.append(track_id)
        seed_tracks = filtered_tracks

    if language_filter:
        language_filtered_tracks = []
        for track_id in seed_tracks:
            track_name = sp.track(track_id)['name']
            track_language = detect(track_name)
            if (language_filter == 'tr' and track_language == 'tr') or (language_filter == 'en' and track_language == 'en'):
                language_filtered_tracks.append(track_id)
        seed_tracks = language_filtered_tracks

    if len(seed_tracks) < 5:
        print("Not enough tracks to generate recommendations.")
        return

    seed_tracks = random.sample(seed_tracks, 5)

    recommendations = sp.recommendations(seed_tracks=seed_tracks, limit=20, market=market_filter)
    recommended_tracks = [track['id'] for track in recommendations['tracks']]
    recommended_tracks_info = [(track['name'], track['artists'][0]['name']) for track in recommendations['tracks']]
    recommended_songs = [track for track in recommended_tracks if track not in liked_songs]

    if recommended_songs:
        create_or_update_playlist(sp, "recommendations-basic-ai-playlist", recommended_songs)
        append_new_recommendations(recommended_tracks_info)
    else:
        print("No new recommendations found. Try updating the dataset.")

if __name__ == "__main__":
    main()
