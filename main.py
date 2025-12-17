from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load Data
songs_data = pd.read_csv("spotify_millsongdata.csv")

#Important Parts
songs_data = songs_data[['song', 'artist', 'text']]

#Fillijng Missing Values
songs_data = songs_data.fillna("")

#FEATURE COMBINATION
songs_data['combined_features'] = (
    songs_data['song'] + " " +
    songs_data['artist'] + " " +
    songs_data['text']
)

#Tf-IDf
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(songs_data['combined_features'])

#Cosine SIMILARITY 
similarity = cosine_similarity(feature_vectors)

#Songs List
list_of_all_songs = songs_data['song'].tolist()


class SongRequest(BaseModel):
    song: str


#API Routes
@app.get("/")
def root():
    return {"message": "Song Recommendation API (Spotify Dataset)"}

@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    # Find closest song name
    close_matches = difflib.get_close_matches(song_name, list_of_all_songs)

    if not close_matches:
        return {"Error": "Song Not Found"}
    
    close_match = close_matches[0]

    song_index = songs_data[songs_data.song == close_match].index[0]

    similarity_scores = list(enumerate(similarity[song_index]))

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



