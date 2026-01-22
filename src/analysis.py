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
