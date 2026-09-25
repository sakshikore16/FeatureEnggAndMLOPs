"""
Unit tests for StreamPulse FastAPI Endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health endpoint responds with 200 and expected status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_valid_request():
    """Verify prediction endpoint handles a realistic payload."""
    payload = {
        "track_name": "Midnight Drive",
        "artist_name": "Sakshi Kore",
        "released_year": 2023,
        "released_month": 5,
        "bpm": 124.0,
        "danceability_%": 75.0,
        "valence_%": 60.0,
        "energy_%": 82.0,
        "acousticness_%": 12.0,
        "instrumentalness_%": 0.0,
        "liveness_%": 10.0,
        "speechiness_%": 5.0,
        "key": "A",
        "mode": "Major",
        "in_spotify_playlists": 800,
        "in_apple_playlists": 60,
        "in_deezer_playlists": 30,
        "in_spotify_charts": 25,
        "in_apple_charts": 15,
        "in_deezer_charts": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_streams" in data
    assert data["predicted_streams"] > 0
    assert "predicted_streams_formatted" in data
    assert "hit_potential_tier" in data


def test_predict_endpoint_validation_error():
    """Verify invalid payloads (e.g. invalid percentage or year) trigger 422."""
    invalid_payload = {
        "track_name": "Invalid Track",
        "released_year": 1800,  # Invalid year < 1950
        "bpm": -50  # Invalid BPM < 40
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
