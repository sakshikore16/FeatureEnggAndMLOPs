"""
StreamPulse Music Analytics - FastAPI Microservice
Serves real-time Spotify song stream predictions and hit potential tiering.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure features module can be resolved regardless of working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features import SpotifyFeatureCreator

app = FastAPI(
    title="StreamPulse Music Forecasting API",
    description="Microservice providing real-time Spotify stream forecasts and hit potential classification.",
    version="1.0.0"
)

# Load pipeline at startup
MODEL_PATH = os.path.join(project_root, 'models', 'spotify_pipeline.joblib')
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model not found at {MODEL_PATH}. Run train.py first.")
        pipeline = joblib.load(MODEL_PATH)
    return pipeline


class TrackPredictionRequest(BaseModel):
    track_name: str = Field(default="My New Track", description="Title of the song")
    artist_name: str = Field(default="Independent Artist", description="Artist name")
    released_year: int = Field(default=2023, ge=1950, le=2030, description="Release year")
    released_month: int = Field(default=6, ge=1, le=12, description="Release month (1-12)")
    bpm: float = Field(default=120.0, ge=40, le=250, description="Beats per minute")
    danceability_pct: float = Field(default=70.0, ge=0, le=100, alias="danceability_%", description="Danceability percentage")
    valence_pct: float = Field(default=60.0, ge=0, le=100, alias="valence_%", description="Valence / cheerfulness percentage")
    energy_pct: float = Field(default=75.0, ge=0, le=100, alias="energy_%", description="Energy percentage")
    acousticness_pct: float = Field(default=20.0, ge=0, le=100, alias="acousticness_%", description="Acousticness percentage")
    instrumentalness_pct: float = Field(default=0.0, ge=0, le=100, alias="instrumentalness_%", description="Instrumentalness percentage")
    liveness_pct: float = Field(default=15.0, ge=0, le=100, alias="liveness_%", description="Liveness percentage")
    speechiness_pct: float = Field(default=6.0, ge=0, le=100, alias="speechiness_%", description="Speechiness percentage")
    key: str = Field(default="C", description="Musical key (e.g. C, C#, D, G, A#)")
    mode: str = Field(default="Major", description="Musical mode ('Major' or 'Minor')")
    in_spotify_playlists: int = Field(default=500, ge=0, description="Number of Spotify playlists")
    in_apple_playlists: int = Field(default=50, ge=0, description="Number of Apple Music playlists")
    in_deezer_playlists: int = Field(default=30, ge=0, description="Number of Deezer playlists")
    in_spotify_charts: int = Field(default=20, ge=0, description="Spotify charts rank/count")
    in_apple_charts: int = Field(default=15, ge=0, description="Apple charts count")
    in_deezer_charts: int = Field(default=5, ge=0, description="Deezer charts count")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "track_name": "Summer Glow",
                "artist_name": "Nova Sound",
                "released_year": 2023,
                "released_month": 7,
                "bpm": 122.0,
                "danceability_%": 78.0,
                "valence_%": 65.0,
                "energy_%": 80.0,
                "acousticness_%": 15.0,
                "instrumentalness_%": 0.0,
                "liveness_%": 12.0,
                "speechiness_%": 5.0,
                "key": "G",
                "mode": "Major",
                "in_spotify_playlists": 850,
                "in_apple_playlists": 65,
                "in_deezer_playlists": 40,
                "in_spotify_charts": 35,
                "in_apple_charts": 20,
                "in_deezer_charts": 10
            }
        }
    }


class TrackPredictionResponse(BaseModel):
    track_name: str
    artist_name: str
    predicted_streams: int
    predicted_streams_formatted: str
    hit_potential_tier: str
    log_stream_index: float
    status: str = "success"


def format_streams(streams: int) -> str:
    if streams >= 1_000_000_000:
        return f"{streams / 1_000_000_000:.2f}B streams"
    elif streams >= 1_000_000:
        return f"{streams / 1_000_000:.1f}M streams"
    elif streams >= 1_000:
        return f"{streams / 1_000:.0f}K streams"
    return f"{streams:,} streams"


def get_hit_tier(streams: int) -> str:
    if streams >= 500_000_000:
        return "🔥 Global Viral Phenomenon (500M+)"
    elif streams >= 150_000_000:
        return "🌟 Major Streaming Hit (150M - 500M)"
    elif streams >= 50_000_000:
        return "📈 Mainstream Radio & Playlist Success (50M - 150M)"
    elif streams >= 10_000_000:
        return "🎵 Solid Indie Streamer (10M - 50M)"
    return "🌱 Emerging Discovery (<10M)"


@app.get("/health")
def health_check():
    """Service health verification endpoint."""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy",
        "service": "StreamPulse Music API",
        "version": "1.0.0",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=TrackPredictionResponse)
def predict_streams(payload: TrackPredictionRequest):
    """Serve real-time song stream prediction from the ML pipeline."""
    try:
        model = get_pipeline()
        
        # Prepare input dataframe with exact feature names
        input_data = pd.DataFrame([{
            'released_year': payload.released_year,
            'released_month': payload.released_month,
            'bpm': payload.bpm,
            'danceability_%': payload.danceability_pct,
            'valence_%': payload.valence_pct,
            'energy_%': payload.energy_pct,
            'acousticness_%': payload.acousticness_pct,
            'instrumentalness_%': payload.instrumentalness_pct,
            'liveness_%': payload.liveness_pct,
            'speechiness_%': payload.speechiness_pct,
            'key': payload.key,
            'mode': payload.mode,
            'in_spotify_playlists': payload.in_spotify_playlists,
            'in_apple_playlists': payload.in_apple_playlists,
            'in_deezer_playlists': payload.in_deezer_playlists,
            'in_spotify_charts': payload.in_spotify_charts,
            'in_apple_charts': payload.in_apple_charts,
            'in_deezer_charts': payload.in_deezer_charts
        }])

        # Predict log-streams and invert with expm1
        log_pred = float(model.predict(input_data)[0])
        raw_streams = int(np.maximum(0, np.expm1(log_pred)))

        return TrackPredictionResponse(
            track_name=payload.track_name,
            artist_name=payload.artist_name,
            predicted_streams=raw_streams,
            predicted_streams_formatted=format_streams(raw_streams),
            hit_potential_tier=get_hit_tier(raw_streams),
            log_stream_index=round(log_pred, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
