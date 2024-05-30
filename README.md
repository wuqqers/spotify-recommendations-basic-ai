<h1 align="center" id="title">Spotify Recommendations with Basic AI</h1>

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg" alt="project-image"></p>

<p id="description">The Spotify Recommendations with Basic AI project is a Python-based application that leverages the Spotify API to provide personalized music recommendations to users. By analyzing user preferences playlist data and track features the system generates tailored recommendations aimed at enhancing the user's music discovery experience.</p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Data Collection: Retrieve user playlist and liked song data from Spotify. Audio Feature Analysis: Analyze audio features such as tempo energy and danceability to understand user preferences. Language Detection: Utilize natural language processing techniques to detect the language of track names for language-specific recommendations. Genre Filtering: Allow users to filter recommendations based on preferred music genres. Model Training: Train a machine learning model for language classification to enhance recommendation accuracy. Recommendation Generation: Generate personalized recommendations considering user preferences genre and language. User Interaction: Interact with the system via a command-line interface for seamless user experience.

<h2>🛠️ Installation Steps:</h2>

<p>1. Install Dependencies:</p>

```
pip install -r requirements.txt
```

<p>2. Configure SPOTIPY_CLIENT_ID &amp; SPOTIPY_CLIENT_SECRET Spotify API credentials from code</p>

```
SPOTIPY_CLIENT_ID = 'SPOTIPY_CLIENT_ID' SPOTIPY_CLIENT_SECRET = 'SPOTIPY_CLIENT_SECRET'
```

<p>3. Run the Script:</p>

```
python main.py
```
