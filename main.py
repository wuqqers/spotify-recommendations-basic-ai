import os
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Spotify API credentials
SPOTIPY_CLIENT_ID = 'SPOTIPY_CLIENT_ID'
SPOTIPY_CLIENT_SECRET = 'SPOTIPY_CLIENT_SECRET'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback/'

# Required scope
scope = 'user-library-read playlist-read-private user-top-read'

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

def train_model(training_data):
    df = pd.DataFrame(training_data)
    X = df.drop(['track_name', 'language', 'id', 'track_href', 'analysis_url', 'uri', 'type'], axis=1)
    y = df['language'].apply(lambda x: 1 if x == 'tr' else 0)  # Binary classification for Turkish and other languages
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    model.save("language_model.h5")
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
            model = load_model("language_model.h5")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except:
            print("Error loading model or model not found. Please update the dataset first.")
            return

    liked_songs = get_liked_songs(sp)

    filter_genre = input("Do you want to filter recommendations by genre? (yes/no): ").lower()
    if filter_genre == 'yes':
        genre_filter = input("Enter a genre (e.g., Rock, Pop, HipHop): ").lower()
    else:
        genre_filter = None

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
    else:
        language_filter = None

    try:
        model = load_model("language_model.h5")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"Error: {e}")
        model = train_model(training_data)

    # Limit seed tracks to 5
    seed_tracks = track_ids[:5]

    try:
        if genre_filter:
            recommendations = sp.recommendations(seed_genres=[genre_filter], limit=50)
        else:
            recommendations = sp.recommendations(seed_tracks=seed_tracks, limit=50)
        
        recommended_tracks = [track['id'] for track in recommendations['tracks']]
    except spotipy.SpotifyException as e:
        print(f"Error fetching recommendations: {e}")
        return
    
    scaler = MinMaxScaler()
    features, _ = fetch_audio_features(sp, recommended_tracks)
    X = pd.DataFrame(features).drop(['id', 'track_href', 'analysis_url', 'uri', 'type'], axis=1)
    X = scaler.fit_transform(X)
    predictions = model.predict(X)
    filtered_recommendations = [track_id for track_id, prediction in zip(recommended_tracks, predictions) if prediction.any() > 0.5]

    filtered_recommendations_info = []
    for rec_id, prediction in zip(recommended_tracks, predictions):
        try:
            if prediction > 0.5:
                track_info = sp.track(rec_id)
                track_name = track_info['name']
                artist_name = track_info['artists'][0]['name']
                language = detect(track_name)
                if (not language_filter or language == language_filter) and (not genre_filter or genre_filter in track_info.get('genres', [])):
                    filtered_recommendations_info.append((track_name, artist_name))
        except spotipy.SpotifyException as e:
            print(f"Error: {e}")

    print("\nRecommended Songs:")

    if not os.path.exists(recommended_songs_file):
        with open(recommended_songs_file, "w", encoding="utf-8") as f:
            json.dump({"new_recommendations": []}, f)

    append_new_recommendations(filtered_recommendations_info)

    for rec_info in filtered_recommendations_info:
        track_info = f"{rec_info[0]} - {rec_info[1]}"
        print(track_info)

if __name__ == '__main__':
    main()
