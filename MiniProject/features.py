"""
StreamPulse Music Analytics - Feature Engineering Module
Custom Scikit-Learn Transformer for Music Feature Extraction.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SpotifyFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that engineers domain-specific features
    from raw Spotify audio attributes and streaming reach metadata.

    Features Engineered:
    --------------------
    1. track_age_years: Evaluates how long the song has accumulated streams (2023 - release_year).
    2. release_month_sin & cos: Cyclical encoding preserving annual calendar seasonality.
    3. log_total_playlists: Aggregates Spotify, Apple, and Deezer playlist inclusions,
       then applies log1p to compress extreme right-tail reach variance.
    4. total_charts: Combined active chart presence across streaming platforms.
    5. dance_energy_ratio: Interaction term capturing high-tempo party/dance intensity.
    6. mode_binary: 1 for Major scale (happier), 0 for Minor scale (moodier).
    7. key: Fills null musical keys with 'Unknown' for downstream categorical encoding.
    """

    def __init__(self, reference_year=2023):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_out = pd.DataFrame(X).copy()

        # 1. Track Age
        if 'released_year' in df_out.columns:
            rel_year = pd.to_numeric(df_out['released_year'], errors='coerce').fillna(self.reference_year)
            df_out['track_age_years'] = np.maximum(0, self.reference_year - rel_year)
        else:
            df_out['track_age_years'] = 0

        # 2. Release Month Cyclical Encodings (1 to 12)
        if 'released_month' in df_out.columns:
            rel_month = pd.to_numeric(df_out['released_month'], errors='coerce').fillna(6)
        else:
            rel_month = 6
        df_out['release_month_sin'] = np.sin(2 * np.pi * rel_month / 12)
        df_out['release_month_cos'] = np.cos(2 * np.pi * rel_month / 12)

        # 3. Aggregated Cross-Platform Playlist Reach
        sp_pl = pd.to_numeric(df_out.get('in_spotify_playlists', 0), errors='coerce').fillna(0)
        ap_pl = pd.to_numeric(df_out.get('in_apple_playlists', 0), errors='coerce').fillna(0)
        
        # Deezer playlist values occasionally have commas
        dz_raw = df_out.get('in_deezer_playlists', 0)
        if isinstance(dz_raw, pd.Series):
            dz_pl = pd.to_numeric(dz_raw.astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        else:
            dz_pl = pd.to_numeric(dz_raw, errors='coerce') or 0

        total_playlists = np.maximum(0, sp_pl + ap_pl + dz_pl)
        df_out['log_total_playlists'] = np.log1p(total_playlists)

        # 4. Total Chart Exposure
        sp_ch = pd.to_numeric(df_out.get('in_spotify_charts', 0), errors='coerce').fillna(0)
        ap_ch = pd.to_numeric(df_out.get('in_apple_charts', 0), errors='coerce').fillna(0)
        dz_ch = pd.to_numeric(df_out.get('in_deezer_charts', 0), errors='coerce').fillna(0)
        df_out['total_charts'] = np.maximum(0, sp_ch + ap_ch + dz_ch)

        # 5. Audio Interaction: Danceability x Energy (Party Synergy)
        dance = pd.to_numeric(df_out.get('danceability_%', 50), errors='coerce').fillna(50)
        energy = pd.to_numeric(df_out.get('energy_%', 50), errors='coerce').fillna(50)
        df_out['dance_energy_ratio'] = (dance * energy) / 100.0

        # 6. Mode Binary (Major = 1, Minor = 0)
        if 'mode' in df_out.columns:
            df_out['mode_binary'] = (df_out['mode'].astype(str).str.strip() == 'Major').astype(int)
        else:
            df_out['mode_binary'] = 1

        # 7. Musical Key Handling
        if 'key' in df_out.columns:
            df_out['key'] = df_out['key'].fillna('Unknown').astype(str).str.strip()
            df_out.loc[df_out['key'] == '', 'key'] = 'Unknown'
            df_out.loc[df_out['key'] == 'nan', 'key'] = 'Unknown'
        else:
            df_out['key'] = 'Unknown'

        # Ensure basic audio percentages are numeric
        for audio_col in ['bpm', 'danceability_%', 'valence_%', 'energy_%',
                          'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']:
            if audio_col in df_out.columns:
                df_out[audio_col] = pd.to_numeric(df_out[audio_col], errors='coerce').fillna(50)
            else:
                df_out[audio_col] = 50

        return df_out
