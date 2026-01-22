import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from data_loader import load_hit_data, load_track_data, load_relation_data, merge_data
from geometry import calculate_coordinates
from analysis import prepare_doppler_spectrogram, analyze_tracks_ai
from visualizer import plot_3d_tracks

def test_pipeline():
    print("Testing Pipeline...")

    # 1. Load Dummy Data
    print("Loading Data...")
    try:
        with open("data/hit_data.csv", "rb") as f:
            hits, doppler_cols = load_hit_data(f)
        with open("data/track_data.csv", "rb") as f:
            tracks = load_track_data(f)
        with open("data/relation_data.csv", "rb") as f:
            relations = load_relation_data(f)
    except FileNotFoundError:
        print("Data files not found. Please generate dummy data first.")
        return

    print(f"Loaded {len(hits)} hits, {len(tracks)} tracks, {len(relations)} relations.")

    # 2. Geometry
    print("Testing Geometry...")
    hits.columns = [c.strip() for c in hits.columns]

    x_w, y_w, z_w = calculate_coordinates(
        hits['range'].values,
        hits['azimuth'].values,
        hits['elevation'].values,
        radar_height=10,
        radar_direction=45
    )

    hits['X'] = x_w
    hits['Y'] = y_w
    hits['Z'] = z_w
    print("Geometry Transform OK.")

    # 3. Merge
    print("Merging Data...")
    merged = merge_data(tracks, hits, relations, doppler_cols)
    print(f"Merged Data Rows: {len(merged)}")

    # 4. Analysis
    print("Testing Analysis...")
    if not merged.empty:
        # Spectrogram
        track_id = merged['TrackID'].iloc[0]
        t_data = merged[merged['TrackID'] == track_id]
        spec, time = prepare_doppler_spectrogram(t_data, doppler_cols)
        print(f"Spectrogram Shape: {spec.shape}")

    # AI
    print("Testing AI Clustering...")
    res, feats = analyze_tracks_ai(tracks)
    print(f"Clustering Result: {res['Cluster'].unique()}")

    # 5. Visualizer
    print("Testing Visualizer...")
    fig = plot_3d_tracks(tracks, hits)
    print("Figure Created.")

    print("Pipeline Test Passed!")

if __name__ == "__main__":
    test_pipeline()
