import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_3d_tracks(tracks_df, hits_df=None, show_hits=True):
    """
    Creates a 3D scatter plot of tracks and optionally hits.
    """
    fig = go.Figure()

    # Plot Tracks
    # Group by TrackID
    for tid in tracks_df['TrackID'].unique():
        t_data = tracks_df[tracks_df['TrackID'] == tid]

        # Track Line
        fig.add_trace(go.Scatter3d(
            x=t_data['X'], y=t_data['Y'], z=t_data['Z'],
            mode='lines+markers',
            name=f'Track {tid}',
            marker=dict(size=4),
            line=dict(width=2)
        ))

    # Plot Hits
    if show_hits and hits_df is not None and not hits_df.empty:
        # We assume hits_df has transformed X, Y, Z columns
        if 'X' in hits_df.columns:
            # Color by Doppler Intensity or ID?
            # Let's use a distinct color or symbol
            fig.add_trace(go.Scatter3d(
                x=hits_df['X'], y=hits_df['Y'], z=hits_df['Z'],
                mode='markers',
                name='Hits',
                marker=dict(size=2, color='gray', opacity=0.5),
                hoverinfo='text',
                text=[f"HitID: {row['hitId']}<br>R: {row['range']:.1f}<br>Az: {row['azimuth']:.1f}"
                      for _, row in hits_df.iterrows()]
            ))

            # Draw lines between Track Point and Hits?
            # This might be too cluttered if there are many hits.
            # Only do this if specific track is selected (handled in main app logic usually).

    fig.update_layout(
        title="3D Track & Hit Visualization",
        scene=dict(
            xaxis_title='X (East)',
            yaxis_title='Y (North)',
            zaxis_title='Z (Up)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_doppler_spectrogram(spectrogram, time_labels):
    """
    Plots the Doppler Spectrogram (Heatmap).
    spectrogram: (FrequencyBins x TimeSteps)
    """
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        x=time_labels,
        # y=Frequency Bins (Indices)
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Doppler Spectrogram (STFT)",
        xaxis_title="Scan / Time",
        yaxis_title="Doppler Bin Index"
    )
    return fig

def plot_doppler_spectrum(doppler_values, title="Doppler Spectrum"):
    """
    Line chart for a single hit's doppler.
    """
    fig = go.Figure(data=go.Scatter(
        y=doppler_values,
        mode='lines+markers'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Bin Index",
        yaxis_title="Amplitude"
    )
    return fig
