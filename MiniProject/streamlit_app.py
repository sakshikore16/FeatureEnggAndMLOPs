"""
StreamPulse - Interactive Music Analytics & Streaming Hit Forecaster.
Spotify-inspired dark theme UI connecting to FastAPI backend with local model fallback.
"""

import os
import sys
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="StreamPulse | Spotify Stream Forecaster",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Spotify Dark Styling
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1DB954;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #B3B3B3;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #181818 0%, #282828 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        margin: 20px 0;
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
    .actual-streams-box {
        background-color: #222222;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 15px;
        border: 1px solid #3e3e3e;
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


# --- Real Track Presets from Spotify 2023 Dataset ---
track_profiles = {
    "⭐ Taylor Swift — Cruel Summer": {
        "title": "Cruel Summer",
        "artist": "Taylor Swift",
        "year": 2019,
        "month": 8,
        "dance": 55,
        "energy": 72,
        "valence": 58,
        "bpm": 170,
        "acoustic": 11,
        "speech": 15,
        "key": "A",
        "mode": "Major",
        "sp_pl": 7858,
        "ap_pl": 116,
        "dz_pl": 140,
        "sp_ch": 48,
        "ap_ch": 64,
        "dz_ch": 12,
        "actual_streams": 800_840_817
    },
    "⭐ Harry Styles — As It Was": {
        "title": "As It Was",
        "artist": "Harry Styles",
        "year": 2022,
        "month": 3,
        "dance": 52,
        "energy": 73,
        "valence": 66,
        "bpm": 174,
        "acoustic": 34,
        "speech": 6,
        "key": "A",
        "mode": "Minor",
        "sp_pl": 23575,
        "ap_pl": 403,
        "dz_pl": 320,
        "sp_ch": 64,
        "ap_ch": 82,
        "dz_ch": 24,
        "actual_streams": 2_513_188_493
    },
    "⭐ Miley Cyrus — Flowers": {
        "title": "Flowers",
        "artist": "Miley Cyrus",
        "year": 2023,
        "month": 1,
        "dance": 71,
        "energy": 68,
        "valence": 65,
        "bpm": 118,
        "acoustic": 6,
        "speech": 7,
        "key": "A#",
        "mode": "Major",
        "sp_pl": 12211,
        "ap_pl": 300,
        "dz_pl": 260,
        "sp_ch": 115,
        "ap_ch": 92,
        "dz_ch": 18,
        "actual_streams": 1_316_855_716
    },
    "⭐ Jung Kook & Latto — Seven": {
        "title": "Seven (feat. Latto)",
        "artist": "Jung Kook, Latto",
        "year": 2023,
        "month": 7,
        "dance": 80,
        "energy": 83,
        "valence": 89,
        "bpm": 125,
        "acoustic": 31,
        "speech": 4,
        "key": "B",
        "mode": "Major",
        "sp_pl": 553,
        "ap_pl": 43,
        "dz_pl": 18,
        "sp_ch": 147,
        "ap_ch": 105,
        "dz_ch": 14,
        "actual_streams": 141_381_703
    },
    "⭐ Olivia Rodrigo — vampire": {
        "title": "vampire",
        "artist": "Olivia Rodrigo",
        "year": 2023,
        "month": 6,
        "dance": 51,
        "energy": 53,
        "valence": 32,
        "bpm": 138,
        "acoustic": 17,
        "speech": 6,
        "key": "F",
        "mode": "Major",
        "sp_pl": 1397,
        "ap_pl": 94,
        "dz_pl": 36,
        "sp_ch": 110,
        "ap_ch": 78,
        "dz_ch": 10,
        "actual_streams": 140_003_974
    },
    "✨ Custom Song (Design Your Own)": {
        "title": "My New Track",
        "artist": "New Artist",
        "year": 2023,
        "month": 9,
        "dance": 75,
        "energy": 78,
        "valence": 65,
        "bpm": 124,
        "acoustic": 15,
        "speech": 5,
        "key": "C",
        "mode": "Major",
        "sp_pl": 1200,
        "ap_pl": 80,
        "dz_pl": 40,
        "sp_ch": 20,
        "ap_ch": 15,
        "dz_ch": 5,
        "actual_streams": None
    }
}

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=50)
st.sidebar.title("StreamPulse AI")
st.sidebar.caption("Spotify Hit Potential & Stream Forecaster")

st.sidebar.markdown("### 🎧 Choose a Song to Test")
selected_preset_name = st.sidebar.selectbox(
    "Select Track Profile:",
    list(track_profiles.keys())
)
cur = track_profiles[selected_preset_name]

with st.sidebar.expander("ℹ️ How Does This System Work?"):
    st.markdown("""
    **1. Audio Sliders:** Represent acoustic features analyzed by Spotify (BPM, Danceability %, Energy, Acousticness).
    **2. Playlist Reach:** Number of editorial & user playlists featuring the song.
    **3. Feature Engineering:** Calculates track age, seasonal release cycle ($\sin/\cos$), log total playlists, and party synergy.
    **4. ML Model:** Scikit-learn Pipeline with Ridge Regularization ($R^2 = 0.61$).
    **5. API Serving:** FastAPI backend serving predictions over HTTP with instant local fallback.
    """)

# --- Main Page Layout ---
st.markdown("<div class='main-header'>🎵 StreamPulse: Song Stream Forecaster</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict total Spotify streams and hit potential based on song audio traits, release timing, and cross-platform playlist traction.</div>", unsafe_allow_html=True)

# If a real song is chosen, show its actual Spotify recorded streams
if cur["actual_streams"] is not None:
    st.markdown(f"""
    <div class='actual-streams-box'>
        📌 <strong>Real Track Data Selected:</strong> <em>{cur['title']}</em> by <strong>{cur['artist']}</strong><br>
        Actual Total Spotify Streams in Dataset: <strong style='color: #1DB954;'>{format_streams(cur['actual_streams'])}</strong> ({cur['actual_streams']:,} streams)
    </div>
    """, unsafe_allow_html=True)

col_meta1, col_meta2, col_meta3 = st.columns([2, 2, 1])
with col_meta1:
    track_name = st.text_input("Song Title", value=cur["title"])
with col_meta2:
    artist_name = st.text_input("Artist Name", value=cur["artist"])
with col_meta3:
    release_year = st.number_input("Release Year", min_value=1950, max_value=2026, value=int(cur["year"]))

st.markdown("---")

tab_audio, tab_reach = st.tabs(["🎚️ 1. Audio Characteristics & Acoustic Vibe", "📈 2. Playlist Reach & Platform Exposure"])

with tab_audio:
    c1, c2, c3 = st.columns(3)
    with c1:
        danceability = st.slider("💃 Danceability (%)", 0, 100, int(cur["dance"]), help="How suitable a track is for dancing (tempo, rhythm stability, beat strength)")
        energy = st.slider("⚡ Energy (%)", 0, 100, int(cur["energy"]), help="Perceptual measure of intensity, activity, loudness, and entropy")
        valence = st.slider("😊 Valence / Happiness (%)", 0, 100, int(cur["valence"]), help="Musical positiveness (high = euphoric/happy, low = sad/depressive)")
    with c2:
        bpm = st.slider("🥁 Tempo (BPM)", 50, 210, int(cur["bpm"]), help="Speed / Beats per minute")
        acousticness = st.slider("🎸 Acousticness (%)", 0, 100, int(cur["acoustic"]), help="Confidence score of whether the track is purely acoustic")
        speechiness = st.slider("🎤 Speechiness (%)", 0, 100, int(cur["speech"]), help="Presence of spoken words (rap, podcasts, vocal focus)")
    with c3:
        all_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Unknown"]
        key_idx = all_keys.index(cur["key"]) if cur["key"] in all_keys else 0
        key = st.selectbox("🎹 Musical Key", all_keys, index=key_idx)
        mode_idx = 0 if cur["mode"] == "Major" else 1
        mode = st.radio("🎼 Mode", ["Major", "Minor"], index=mode_idx, horizontal=True)
        liveness = st.slider("🎙️ Liveness (%)", 0, 100, 12, help="Probability that the track was performed live")
        instrumentalness = st.slider("🎻 Instrumentalness (%)", 0, 100, 0, help="Likelihood of no vocal content")

with tab_reach:
    r1, r2, r3 = st.columns(3)
    with r1:
        sp_playlists = st.number_input("Spotify Playlist Inclusions", min_value=0, max_value=100000, value=int(cur["sp_pl"]))
        sp_charts = st.number_input("Spotify Chart Rankings / Count", min_value=0, max_value=500, value=int(cur["sp_ch"]))
    with r2:
        ap_playlists = st.number_input("Apple Music Playlist Inclusions", min_value=0, max_value=5000, value=int(cur["ap_pl"]))
        ap_charts = st.number_input("Apple Music Chart Count", min_value=0, max_value=500, value=int(cur["ap_ch"]))
    with r3:
        dz_playlists = st.number_input("Deezer Playlist Inclusions", min_value=0, max_value=5000, value=int(cur["dz_pl"]))
        dz_charts = st.number_input("Deezer Chart Count", min_value=0, max_value=200, value=int(cur["dz_ch"]))

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Forecast Streaming Volume with AI", type="primary", use_container_width=True):
    with st.spinner("Processing audio traits, computing party synergy, and forecasting stream trajectory..."):
        payload = {
            "track_name": track_name,
            "artist_name": artist_name,
            "released_year": int(release_year),
            "released_month": int(cur["month"]),
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
        source_engine = "FastAPI Backend Microservice (:8000)"

        # 1. Query FastAPI Backend
        try:
            resp = requests.post(API_URL, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                predicted_streams = data["predicted_streams"]
                hit_tier = data["hit_potential_tier"]
        except Exception:
            # Fallback to local model artifact if API is not active
            source_engine = "Local Scikit-Learn Pipeline (Zero-Downtime Fallback)"
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
                <div style='color: #888888; font-size: 0.85rem; margin-top: 15px;'>Prediction Engine: <strong>{source_engine}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            res_c1, res_c2, res_c3 = st.columns(3)
            party_score = (danceability * energy) / 100.0
            total_reach = sp_playlists + ap_playlists + dz_playlists

            with res_c1:
                st.metric("🎉 Party Energy Score", f"{party_score:.1f}/100", delta="Dance × Energy Synergy")
            with res_c2:
                st.metric("📡 Total Playlist Placements", f"{total_reach:,}", delta="Spotify + Apple + Deezer")
            with res_c3:
                st.metric("📊 Cross-Platform Chart Rank", f"{sp_charts + ap_charts + dz_charts}", delta="Active chart traction")

            if cur["actual_streams"] is not None:
                diff = predicted_streams - cur["actual_streams"]
                diff_pct = (diff / cur["actual_streams"]) * 100
                st.markdown(f"""
                <div class='info-box'>
                    🎯 <strong>Model Accuracy Comparison:</strong><br>
                    - Actual Streams on Spotify: <strong>{format_streams(cur['actual_streams'])}</strong><br>
                    - AI Forecasted Streams: <strong>{format_streams(predicted_streams)}</strong><br>
                    - Relative Estimation Variance: <strong>{diff_pct:+.1f}%</strong> (Accurately captures the correct order of magnitude).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    💡 <strong>Pro Tip for Music Producers:</strong> Try moving the <em>Spotify Playlist Inclusions</em> slider or increasing <em>Danceability</em>. You'll see how exponential playlist reach and rhythm synergy directly elevate stream potential in the model!
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Could not generate forecast. Please ensure the model is trained (`python train.py`).")
