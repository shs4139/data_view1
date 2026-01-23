import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def prepare_doppler_spectrogram(track_hits, doppler_cols):
    """
    Prepares the Doppler data for visualization.
    Input: DataFrame of hits associated with a track.
    Output: 2D Matrix (Time x FrequencyBins), Time labels.
    """
    # Sort by time
    # Check if 'LogTime' is parsable, otherwise use 'ScanNum' or index.
    # Assuming ScanNum is reliable for ordering.
    track_hits = track_hits.sort_values('ScanNum')

    # Extract Doppler Columns
    spectrogram = track_hits[doppler_cols].values.T # Transpose to get Frequency x Time

    time_labels = track_hits['ScanNum'].values

    return spectrogram, time_labels

def identify_static_tracks(merged_df, doppler_cols, center_bin=30, width=2, threshold=0.5):
    """
    Identifies tracks that are likely static clutter based on Doppler distribution.

    Args:
        merged_df: DataFrame containing merged Track and Hit data.
        doppler_cols: List of column names for Doppler bins.
        center_bin: The index of the Doppler bin representing zero/low velocity (default 30).
        width: The range +/- around the center to consider as static (default 2).
        threshold: The fraction of hits (0.0 to 1.0) that must be static to filter the track.

    Returns:
        valid_track_ids: List of TrackIDs that are NOT static.
    """
    if merged_df.empty or not doppler_cols:
        return []

    # 1. Calculate Peak Bin for each Hit
    # Convert object to float if necessary, though argmax works on numeric types
    # We use numpy argmax on the subset of columns
    doppler_data = merged_df[doppler_cols].values

    # Ensure numeric
    # If mixed types, force cast. usually better to do this once during load, but safe here.
    try:
        doppler_data = doppler_data.astype(float)
    except ValueError:
        pass # Handle if needed

    peak_bins = np.argmax(doppler_data, axis=1)

    # 2. Check if Peak is Static
    # Static if center - width <= peak <= center + width
    is_static_hit = (peak_bins >= (center_bin - width)) & (peak_bins <= (center_bin + width))

    # 3. Aggregate per Track
    # Create a small temp DF
    temp_df = pd.DataFrame({
        'TrackID': merged_df['TrackID'],
        'IsStatic': is_static_hit
    })

    # Group by TrackID
    track_stats = temp_df.groupby('TrackID')['IsStatic'].agg(['mean', 'count'])
    # 'mean' of boolean is the ratio of True (Static)

    # 4. Filter
    # If mean (ratio) > threshold, it is static.
    # We want valid tracks (<= threshold)
    valid_tracks = track_stats[track_stats['mean'] <= threshold].index.tolist()

    return valid_tracks

def analyze_tracks_ai(tracks_df):
    """
    Groups tracks by characteristics using K-Means.
    Returns the dataframe with a new 'Cluster' column and the cluster centers.
    """
    # Feature Extraction per TrackID
    # We need to aggregate the track points per TrackID

    features_list = []
    track_ids = tracks_df['TrackID'].unique()

    for tid in track_ids:
        t_data = tracks_df[tracks_df['TrackID'] == tid]

        # Features
        avg_z = t_data['Z'].mean()
        max_z = t_data['Z'].max()

        # Velocity
        v = np.sqrt(t_data['VX']**2 + t_data['VY']**2 + t_data['VZ']**2)
        avg_v = v.mean()
        max_v = v.max()
        std_v = v.std() # Maneuverability

        avg_rcs = t_data['RCS'].mean()

        features_list.append([tid, avg_z, max_z, avg_v, max_v, std_v, avg_rcs])

    feature_df = pd.DataFrame(features_list, columns=['TrackID', 'AvgZ', 'MaxZ', 'AvgV', 'MaxV', 'StdV', 'AvgRCS'])

    # Drop NaN
    feature_df = feature_df.fillna(0)

    # Clustering
    # Select features for clustering (excluding ID)
    X = feature_df.drop(columns=['TrackID'])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    # Determine K? Let's assume 3 types (e.g., Drone, Bird, Aircraft)
    kmeans = KMeans(n_clusters=3, random_state=42)
    feature_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map back to original tracks
    tracks_with_cluster = tracks_df.merge(feature_df[['TrackID', 'Cluster']], on='TrackID', how='left')

    return tracks_with_cluster, feature_df
