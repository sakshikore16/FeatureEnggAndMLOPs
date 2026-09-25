"""
StreamPulse Music Analytics - Streamlit Web Dashboard
Interactive Spotify stream forecasting and hit potential analysis tool.
"""

import os
import sys
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page metadata and styling
st.set_page_config(
    page_title="StreamPulse - AI Music Forecasting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Spotify-themed CSS
st.markdown("""
<style>
    /* Dark theme with Spotify Green accents */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1DB954;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #B3B3B3;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #181818 0%, #282828 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .stream-number {
        font-size: 3.2rem;
        font-weight: 900;
        color: #1DB954;
        margin: 10px 0;
    }
    .hit-badge {
        display: inline-block;
        background-color: rgba(29, 185, 84, 0.15);
        border: 1px solid #1DB954;
        color: #1DB954;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 8px;
    }
    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #1DB954;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 15px;
        font-size: 0.95rem;
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to sys.path to enable loading features.py locally
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features import SpotifyFeatureCreator

# API endpoint URL
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
MODEL_PATH = os.path.join(project_root, "models", "spotify_pipeline.joblib")


@st.cache_resource
def load_local_model():
    """Load local model artifact as an instant fallback if API is not running."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def format_streams(streams: int) -> str:
    if streams >= 1_000_000_000:
        return f"{streams / 1_000_000_000:.2f} Billion"
    elif streams >= 1_000_000:
        return f"{streams / 1_000_000:.1f} Million"
    elif streams >= 1_000:
        return f"{streams / 1_000:.0f} Thousand"
    return f"{streams:,}"


def get_hit_tier(streams: int) -> str:
    if streams >= 500_000_000:
        return "🔥 Global Viral Phenomenon (500M+)"
    elif streams >= 150_000_000:
        return "🌟 Major Streaming Hit (150M - 500M)"
    elif streams >= 50_000_000:
        return "📈 Mainstream Playlist Hit (50M - 150M)"
    elif streams >= 10_000_000:
        return "🎵 Solid Streaming Track (10M - 50M)"
    return "🌱 Emerging Discovery (<10M)"


# --- Sidebar Navigation & Presets ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=50)
st.sidebar.title("StreamPulse AI")
st.sidebar.caption("Music Hit Forecaster & Analytics")

st.sidebar.markdown("### 🎛️ Quick Track Presets")
preset = st.sidebar.selectbox(
    "Load Song Profile",
    ["Custom Track", "High-Energy Pop Banger", "Acoustic Indie Ballad", "Club Electronic / Dance", "Reggaeton / Latin Flow"]
)

# Preset values
defaults = {
    "Custom Track": {"dance": 72, "energy": 75, "valence": 60, "bpm": 122, "acoustic": 18, "speech": 6, "sp_pl": 600, "ap_pl": 45},
    "High-Energy Pop Banger": {"dance": 80, "energy": 88, "valence": 82, "bpm": 128, "acoustic": 8, "speech": 5, "sp_pl": 2500, "ap_pl": 180},
    "Acoustic Indie Ballad": {"dance": 45, "energy": 32, "valence": 25, "bpm": 85, "acoustic": 85, "speech": 4, "sp_pl": 350, "ap_pl": 20},
    "Club Electronic / Dance": {"dance": 86, "energy": 94, "valence": 70, "bpm": 126, "acoustic": 2, "speech": 9, "sp_pl": 1800, "ap_pl": 120},
    "Reggaeton / Latin Flow": {"dance": 88, "energy": 79, "valence": 84, "bpm": 96, "acoustic": 14, "speech": 12, "sp_pl": 1400, "ap_pl": 90}
}
cur = defaults[preset]

# --- Main Page Layout ---
st.markdown("<div class='main-header'>🎵 StreamPulse: Song Stream Forecaster</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict total Spotify streaming performance and hit potential using machine learning and audio feature engineering.</div>", unsafe_allow_html=True)

col_meta1, col_meta2, col_meta3 = st.columns([2, 2, 1])
with col_meta1:
    track_name = st.text_input("Song Title", value="Cruel Energy")
with col_meta2:
    artist_name = st.text_input("Artist Name", value="Sakshi Kore & The Sound")
with col_meta3:
    release_year = st.number_input("Release Year", min_value=1950, max_value=2026, value=2023)

st.markdown("---")

tab_audio, tab_reach = st.tabs(["🎚️ 1. Audio Characteristics & Music Theory", "📈 2. Playlist Reach & Platform Exposure"])

with tab_audio:
    c1, c2, c3 = st.columns(3)
    with c1:
        danceability = st.slider("💃 Danceability (%)", 0, 100, cur["dance"], help="Suitability of a track for dancing")
        energy = st.slider("⚡ Energy (%)", 0, 100, cur["energy"], help="Perceptual measure of intensity and activity")
        valence = st.slider("😊 Valence / Happiness (%)", 0, 100, cur["valence"], help="Musical positiveness conveyed by a track")
    with c2:
        bpm = st.slider("🥁 Tempo (BPM)", 50, 210, cur["bpm"], help="Speed / Beats per minute")
        acousticness = st.slider("🎸 Acousticness (%)", 0, 100, cur["acoustic"], help="Confidence measure of whether track is acoustic")
        speechiness = st.slider("🎤 Speechiness (%)", 0, 100, cur["speech"], help="Presence of spoken words in a track")
    with c3:
        key = st.selectbox("🎹 Musical Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Unknown"], index=7)
        mode = st.radio("🎼 Mode", ["Major", "Minor"], index=0, horizontal=True)
        liveness = st.slider("🎙️ Liveness (%)", 0, 100, 12, help="Probability that the track was performed live")
        instrumentalness = st.slider("🎻 Instrumentalness (%)", 0, 100, 0, help="Likelihood of no vocal content")

with tab_reach:
    r1, r2, r3 = st.columns(3)
    with r1:
        sp_playlists = st.number_input("Spotify Playlist Inclusions", min_value=0, max_value=100000, value=cur["sp_pl"])
        sp_charts = st.number_input("Spotify Chart Rankings / Count", min_value=0, max_value=500, value=25)
    with r2:
        ap_playlists = st.number_input("Apple Music Playlist Inclusions", min_value=0, max_value=5000, value=cur["ap_pl"])
        ap_charts = st.number_input("Apple Music Chart Count", min_value=0, max_value=500, value=15)
    with r3:
        dz_playlists = st.number_input("Deezer Playlist Inclusions", min_value=0, max_value=5000, value=30)
        dz_charts = st.number_input("Deezer Chart Count", min_value=0, max_value=200, value=5)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Forecast Streaming Volume", type="primary", use_container_width=True):
    with st.spinner("Analyzing audio features, computing interaction synergy, and generating stream forecast..."):
        payload = {
            "track_name": track_name,
            "artist_name": artist_name,
            "released_year": int(release_year),
            "released_month": 7,
            "bpm": float(bpm),
            "danceability_%": float(danceability),
            "valence_%": float(valence),
            "energy_%": float(energy),
            "acousticness_%": float(acousticness),
            "instrumentalness_%": float(instrumentalness),
            "liveness_%": float(liveness),
            "speechiness_%": float(speechiness),
            "key": key,
            "mode": mode,
            "in_spotify_playlists": int(sp_playlists),
            "in_apple_playlists": int(ap_playlists),
            "in_deezer_playlists": int(dz_playlists),
            "in_spotify_charts": int(sp_charts),
            "in_apple_charts": int(ap_charts),
            "in_deezer_charts": int(dz_charts)
        }

        predicted_streams = None
        hit_tier = None
        source_engine = "FastAPI Microservice"

        # 1. Try FastAPI Backend
        try:
            resp = requests.post(API_URL, json=payload, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                predicted_streams = data["predicted_streams"]
                hit_tier = data["hit_potential_tier"]
        except Exception:
            # Fallback to local model artifact if API is not active
            source_engine = "Local Pipeline (Fallback Engine)"
            local_model = load_local_model()
            if local_model is not None:
                inp_df = pd.DataFrame([payload])
                log_pred = float(local_model.predict(inp_df)[0])
                predicted_streams = int(np.maximum(0, np.expm1(log_pred)))
                hit_tier = get_hit_tier(predicted_streams)

        if predicted_streams is not None:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color: #888888; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;'>Forecasted Lifetime Spotify Streams</div>
                <div class='stream-number'>{format_streams(predicted_streams)}</div>
                <div class='hit-badge'>{hit_tier}</div>
                <div style='color: #666666; font-size: 0.8rem; margin-top: 15px;'>Prediction served via: {source_engine}</div>
            </div>
            """, unsafe_allow_html=True)

            res_c1, res_c2, res_c3 = st.columns(3)
            party_score = (danceability * energy) / 100.0
            total_reach = sp_playlists + ap_playlists + dz_playlists

            with res_c1:
                st.metric("🎉 Party Energy Score", f"{party_score:.1f}/100", delta="Dance x Energy Synergy")
            with res_c2:
                st.metric("📡 Total Playlist Placement", f"{total_reach:,}", delta="Cross-platform reach")
            with res_c3:
                st.metric("📊 Chart Exposure Points", f"{sp_charts + ap_charts + dz_charts}", delta="Active chart traction")

            st.markdown("""
            <div class='info-box'>
                💡 <strong>Feature Engineering Impact:</strong> Notice how increasing <em>Spotify Playlist Inclusions</em> or the <em>Party Energy Synergy</em> significantly lifts the forecast. Our feature pipeline uses logarithmic scaling to ensure extreme reach numbers do not distort linear model weights.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not generate forecast. Please ensure the model is trained (`python train.py`).")
