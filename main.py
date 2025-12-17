from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib
import numpy as np  # Added for memory efficiency

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MEMORY OPTIMIZATION: Load only 15,000 songs ---
# The full 57k dataset is too large for 512MB RAM.
songs_data = pd.read_csv("spotify_millsongdata.csv").head(15000)

# Important Parts
songs_data = songs_data[['song', 'artist', 'text']]
songs_data = songs_data.fillna("")

# FEATURE COMBINATION
songs_data['combined_features'] = (
    songs_data['song'] + " " +
    songs_data['artist'] + " " +
    songs_data['text']
)

# Tf-IDf
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(songs_data['combined_features'])

# --- 2. REMOVED: Global similarity matrix calculation ---
# We no longer calculate 'similarity = cosine_similarity(feature_vectors)' here.

# Songs List
list_of_all_songs = songs_data['song'].tolist()

class SongRequest(BaseModel):
    song: str

# API Routes
@app.get("/")
def root():
    return {"message": "Song Recommendation API (Optimized for Render Free Tier)"}

@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    # Find closest song name
    close_matches = difflib.get_close_matches(song_name, list_of_all_songs)

    if not close_matches:
        return {"Error": "Song Not Found"}
    
    close_match = close_matches[0]
    song_index = songs_data[songs_data.song == close_match].index[0]

    # --- 3. CALCULATION ON-THE-FLY ---
    # Instead of a 57k x 57k matrix, we calculate a 1 x 15k vector for the input song only.
    # This uses < 1MB of RAM instead of 26GB!
    current_song_vector = feature_vectors[song_index]
    scores = cosine_similarity(current_song_vector, feature_vectors).flatten()
    
    # Enumerate and sort
    similarity_scores = list(enumerate(scores))
    sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, song in enumerate(sorted_songs[1:11]):
        index = song[0]
        recommended_song = songs_data.iloc[index].song
        recommendations.append(recommended_song)

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }