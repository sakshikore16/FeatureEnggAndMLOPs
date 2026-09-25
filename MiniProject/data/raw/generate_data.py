"""
Data Verification and Preprocessing Utility for StreamPulse.
"""

import os
import pandas as pd


def verify_and_clean():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(base_dir, 'spotify_2023.csv')

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source dataset not found at: {source_file}")

    df = pd.read_csv(source_file, encoding='latin-1')
    initial_count = len(df)

    # Filter invalid stream entries
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    clean_df = df.dropna(subset=['streams']).reset_index(drop=True)

    # Clean comma-formatted numeric strings
    for col in ['in_deezer_playlists', 'in_shazam_charts']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.replace(',', '')
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)

    clean_df.to_csv(source_file, index=False)
    print(f"Data verified: {initial_count} raw rows -> {len(clean_df)} clean rows saved to {source_file}")


if __name__ == '__main__':
    verify_and_clean()
