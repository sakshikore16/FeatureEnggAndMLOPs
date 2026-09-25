"""
Unit tests for StreamPulse ML Feature Engineering & Pipeline.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features import SpotifyFeatureCreator

MODEL_PATH = os.path.join(project_root, 'models', 'spotify_pipeline.joblib')


@pytest.fixture
def pipeline():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}. Run train.py first."
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_track():
    return pd.DataFrame([{
        'track_name': 'Test Anthem',
        'artist(s)_name': 'Test Artist',
        'released_year': 2023,
        'released_month': 6,
        'bpm': 120,
        'danceability_%': 75,
        'valence_%': 65,
        'energy_%': 80,
        'acousticness_%': 15,
        'instrumentalness_%': 0,
        'liveness_%': 10,
        'speechiness_%': 6,
        'key': 'C',
        'mode': 'Major',
        'in_spotify_playlists': 500,
        'in_apple_playlists': 50,
        'in_deezer_playlists': 20,
        'in_spotify_charts': 10,
        'in_apple_charts': 5,
        'in_deezer_charts': 2
    }])


def test_pipeline_loads_correctly(pipeline):
    """Confirm the trained pipeline can be deserialized and contains all stages."""
    assert pipeline is not None
    assert 'creator' in pipeline.named_steps
    assert 'prep' in pipeline.named_steps
    assert 'regressor' in pipeline.named_steps or 'model' in pipeline.named_steps


def test_pipeline_prediction_range(pipeline, sample_track):
    """Predictions should yield plausible positive stream counts."""
    log_pred = pipeline.predict(sample_track)[0]
    streams = np.expm1(log_pred)
    assert streams > 0
    assert streams < 100_000_000_000  # under 100 Billion


def test_directional_sanity_playlists(pipeline, sample_track):
    """Increasing playlist reach should increase expected streams."""
    base_pred = pipeline.predict(sample_track)[0]

    higher_reach_track = sample_track.copy()
    higher_reach_track['in_spotify_playlists'] = 15000
    higher_pred = pipeline.predict(higher_reach_track)[0]

    assert higher_pred > base_pred, "Model failed directional sanity: higher playlist reach must yield more streams!"


def test_pipeline_handles_unseen_category(pipeline, sample_track):
    """Model must gracefully handle unknown musical keys without throwing an error."""
    unseen_track = sample_track.copy()
    unseen_track['key'] = 'Unknown_Weird_Key_X'
    unseen_track['mode'] = 'Lydian'

    log_pred = pipeline.predict(unseen_track)[0]
    assert not np.isnan(log_pred)


def test_pipeline_is_deterministic(pipeline, sample_track):
    """Identical inputs must yield identical predictions."""
    pred1 = pipeline.predict(sample_track)[0]
    pred2 = pipeline.predict(sample_track)[0]
    assert np.isclose(pred1, pred2)
