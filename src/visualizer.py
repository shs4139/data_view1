import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_3d_tracks(tracks_df, hits_df=None, show_hits=True, x_range=None, y_range=None, z_range=None):
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
            fig.add_trace(go.Scatter3d(
                x=hits_df['X'], y=hits_df['Y'], z=hits_df['Z'],
                mode='markers',
                name='Hits',
                marker=dict(size=2, color='gray', opacity=0.5),
                hoverinfo='text',
                text=[f"HitID: {row['hitId']}<br>R: {row['range']:.1f}<br>Az: {row['azimuth']:.1f}"
                      for _, row in hits_df.iterrows()]
            ))

    # Update Layout with Axis Ranges
    scene_dict = dict(
        xaxis_title='X (East)',
        yaxis_title='Y (North)',
        zaxis_title='Z (Up)'
    )
    if x_range: scene_dict['xaxis'] = dict(range=x_range)
    if y_range: scene_dict['yaxis'] = dict(range=y_range)
    if z_range: scene_dict['zaxis'] = dict(range=z_range)

    fig.update_layout(
        title="3D Track & Hit Visualization",
        scene=scene_dict,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_2d_view(tracks_df, hits_df=None, view_type='top', x_range=None, y_range=None, z_range=None):
    """
    Creates a 2D scatter plot (Top or Side View).
    view_type: 'top' (X-Y) or 'side' (X-Z)
    """
    fig = go.Figure()

    if view_type == 'top':
        x_col, y_col = 'X', 'Y'
        title = "Top View (X-Y)"
        yaxis_title = "Y (North)"
        y_axis_range = y_range
    else: # side
        x_col, y_col = 'X', 'Z'
        title = "Side View (X-Z)"
        yaxis_title = "Z (Up)"
        y_axis_range = z_range

    # Plot Tracks
    for tid in tracks_df['TrackID'].unique():
        t_data = tracks_df[tracks_df['TrackID'] == tid]
        fig.add_trace(go.Scatter(
            x=t_data[x_col], y=t_data[y_col],
            mode='lines+markers',
            name=f'Track {tid}',
            marker=dict(size=5)
        ))

    # Plot Hits
    if hits_df is not None and not hits_df.empty:
        fig.add_trace(go.Scatter(
            x=hits_df[x_col], y=hits_df[y_col],
            mode='markers',
            name='Hits',
            marker=dict(size=3, color='gray', opacity=0.5)
        ))

    # Layout
    layout_args = dict(
        title=title,
        xaxis_title="X (East)",
        yaxis_title=yaxis_title,
        xaxis=dict(range=x_range) if x_range else None,
        yaxis=dict(range=y_axis_range) if y_axis_range else None,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    # Remove None values
    layout_args = {k: v for k, v in layout_args.items() if v is not None}

    fig.update_layout(**layout_args)
    return fig

def plot_doppler_spectrogram(spectrogram, time_labels):
    """
    Plots the Doppler Spectrogram (Heatmap) in dB.
    spectrogram: (FrequencyBins x TimeSteps)
    """
    # Ensure numeric
    spec_vals = np.array(spectrogram, dtype=float)

    # Convert to dB
    epsilon = 1e-9
    spec_db = 20 * np.log10(np.abs(spec_vals) + epsilon)

    fig = go.Figure(data=go.Heatmap(
        z=spec_db,
        x=time_labels,
        # y=Frequency Bins (Indices)
        colorscale='Viridis',
        colorbar=dict(title="Amplitude (dB)")
    ))

    fig.update_layout(
        title="Doppler Spectrogram (STFT) - dB",
        xaxis_title="Scan / Time",
        yaxis_title="Doppler Bin Index"
    )
    return fig

def plot_doppler_spectrum(doppler_values, title="Doppler Spectrum"):
    """
    Line chart for a single hit's doppler in dB.
    """
    # Ensure values are numeric float (handles object arrays)
    d_vals = np.array(doppler_values, dtype=float)

    # Convert to dB: 20 * log10(abs(amplitude))
    # Handle zeros by adding epsilon
    epsilon = 1e-9
    doppler_db = 20 * np.log10(np.abs(d_vals) + epsilon)

    fig = go.Figure(data=go.Scatter(
        y=doppler_db,
        mode='lines+markers'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Bin Index",
        yaxis_title="Amplitude (dB)"
    )
    return fig
