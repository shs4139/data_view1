import streamlit as st
import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from data_loader import load_hit_data, load_track_data, load_relation_data, merge_data
from geometry import calculate_coordinates
from visualizer import plot_3d_tracks, plot_2d_view, plot_doppler_spectrogram, plot_doppler_spectrum
from analysis import prepare_doppler_spectrogram, analyze_tracks_ai
from utils import generate_dummy_data

st.set_page_config(page_title="Radar Data Analyst", layout="wide")

st.title("Radar Data Analysis & Visualization")

# --- Sidebar: Configuration & Inputs ---
st.sidebar.header("Configuration")

# 1. File Inputs
st.sidebar.subheader("Data Source")
source_mode = st.sidebar.radio("Select Mode", ["File Upload", "Local Directory"], index=0)

use_dummy = False
load_from_dir = False
hit_file = None
track_file = None
relation_file = None
data_dir = ""

if source_mode == "Local Directory":
    # Helper to open folder dialog
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = ""

    col_dir1, col_dir2 = st.sidebar.columns([3, 1])
    with col_dir1:
        data_dir_input = st.text_input("Data Directory Path", value=st.session_state["data_dir"], key="dir_input")
    with col_dir2:
        if st.button("Browse"):
            try:
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                folder_selected = filedialog.askdirectory(master=root)
                root.destroy()
                if folder_selected:
                    st.session_state["data_dir"] = folder_selected
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Cannot open folder dialog: {e}")

    data_dir = st.session_state["data_dir"] if st.session_state["data_dir"] else data_dir_input

    if data_dir and os.path.isdir(data_dir):
        h_path = os.path.join(data_dir, "out_hitinfo.csv")
        t_path = os.path.join(data_dir, "out_trackinfo.csv")
        r_path = os.path.join(data_dir, "track_hitlist_global.csv")

        if os.path.exists(h_path) and os.path.exists(t_path) and os.path.exists(r_path):
            load_from_dir = True
            st.sidebar.success(f"Directory Valid: {data_dir}")
        else:
            st.sidebar.error("Required files (out_hitinfo.csv, out_trackinfo.csv, track_hitlist_global.csv) not found.")
            use_dummy = True
    else:
        st.sidebar.info("Please select a valid directory.")
        use_dummy = True

else:
    # File Upload Mode
    hit_file = st.sidebar.file_uploader("Hit Data (CSV)", type=["csv", "txt"])
    track_file = st.sidebar.file_uploader("Track Data (CSV)", type=["csv", "txt"])
    relation_file = st.sidebar.file_uploader("Relation Data (CSV)", type=["csv", "txt"])

    if not (hit_file and track_file and relation_file):
        st.sidebar.info("Using Dummy Data (Upload files to override)")
        use_dummy = True

# 2. Radar Geometry
st.sidebar.subheader("Radar Geometry")
radar_height = st.sidebar.number_input("Radar Height (m)", value=0.0)
# Tilt removed as per new logic requirement
radar_dir = st.sidebar.number_input("Radar Direction (deg, Azimuth offset)", value=0.0)

# 3. Filters
st.sidebar.subheader("Filters")
min_power = st.sidebar.number_input("Min Hit Power", value=0)
min_velocity = st.sidebar.number_input("Min Track Velocity", value=0.0)

# 4. Visualization Settings
st.sidebar.subheader("Axis Ranges (Optional)")
st.sidebar.caption("Leave 0 to auto-scale")
x_min = st.sidebar.number_input("X Min", value=0.0)
x_max = st.sidebar.number_input("X Max", value=0.0)
y_min = st.sidebar.number_input("Y Min", value=0.0)
y_max = st.sidebar.number_input("Y Max", value=0.0)
z_min = st.sidebar.number_input("Z Min", value=0.0)
z_max = st.sidebar.number_input("Z Max", value=0.0)

# Helper to construct ranges
x_range = [x_min, x_max] if x_min != x_max else None
y_range = [y_min, y_max] if y_min != y_max else None
z_range = [z_min, z_max] if z_min != z_max else None

# --- Data Loading ---
@st.cache_data
def load_data(h_file, t_file, r_file, _use_dummy=False, _load_from_dir=False, _dir_path=""):
    if _use_dummy:
        base_dir = "data"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Check if files exist, if not generate
        required_files = ["hit_data.csv", "track_data.csv", "relation_data.csv"]
        if not all(os.path.exists(os.path.join(base_dir, f)) for f in required_files):
            with st.spinner("Generating Dummy Data..."):
                generate_dummy_data(base_dir)

        with open(os.path.join(base_dir, "hit_data.csv"), "rb") as f:
            hits, doppler_cols = load_hit_data(f)
        with open(os.path.join(base_dir, "track_data.csv"), "rb") as f:
            tracks = load_track_data(f)
        with open(os.path.join(base_dir, "relation_data.csv"), "rb") as f:
            relations = load_relation_data(f)

    elif _load_from_dir:
        # Load from specified directory with fixed filenames
        with open(os.path.join(_dir_path, "out_hitinfo.csv"), "rb") as f:
            hits, doppler_cols = load_hit_data(f)
        with open(os.path.join(_dir_path, "out_trackinfo.csv"), "rb") as f:
            tracks = load_track_data(f)
        with open(os.path.join(_dir_path, "track_hitlist_global.csv"), "rb") as f:
            relations = load_relation_data(f)

    else:
        # Use Uploaded Files
        hits, doppler_cols = load_hit_data(h_file)
        tracks = load_track_data(t_file)
        relations = load_relation_data(r_file)

    return hits, tracks, relations, doppler_cols

try:
    hits_df, tracks_df, relations_df, doppler_cols = load_data(hit_file, track_file, relation_file, use_dummy, load_from_dir, data_dir)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if hits_df is None:
    st.warning("No data available. Please generate dummy data or upload files.")
    st.stop()

# --- Preprocessing: Coordinate Transform ---
st.sidebar.markdown("---")
st.sidebar.text("Processing Coordinates...")

# Calculate Cartesian
# Ensure column names match. The dummy data has 'range', 'azimuth', 'elevation'.
# Make sure casing matches.
hits_df.columns = [c.strip() for c in hits_df.columns]
req_cols = ['range', 'azimuth', 'elevation']
if not all(col in hits_df.columns for col in req_cols):
    st.error(f"Hit data missing required columns: {req_cols}. Found: {hits_df.columns}")
    st.stop()

x_w, y_w, z_w = calculate_coordinates(
    hits_df['range'].values,
    hits_df['azimuth'].values,
    hits_df['elevation'].values,
    radar_height,
    radar_dir
)

hits_df['X'] = x_w
hits_df['Y'] = y_w
hits_df['Z'] = z_w

# Merge Data
# We merge to associate Hits with TrackIDs
merged_df = merge_data(tracks_df, hits_df, relations_df, doppler_cols)

# --- Apply Filters ---
# 1. Hit Power
if min_power > 0:
    hits_df = hits_df[hits_df['power'] >= min_power]
    merged_df = merged_df[merged_df['power'] >= min_power]

# 2. Track Velocity
# Calculate velocity magnitude for tracks
tracks_df['Velocity'] = np.sqrt(tracks_df['VX']**2 + tracks_df['VY']**2 + tracks_df['VZ']**2)
if min_velocity > 0:
    # Filter tracks by average velocity > min
    track_avg_v = tracks_df.groupby('TrackID')['Velocity'].mean()
    valid_track_ids = track_avg_v[track_avg_v >= min_velocity].index
    tracks_df = tracks_df[tracks_df['TrackID'].isin(valid_track_ids)]
    # Also filter merged
    merged_df = merged_df[merged_df['TrackID'].isin(valid_track_ids)]

# --- Global Statistics ---
st.write(f"**Loaded:** {len(tracks_df['TrackID'].unique())} Tracks, {len(hits_df)} Hits (Filtered)")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["3D Visualization", "Doppler Analysis", "AI Classification", "Data Export"])

# === Tab 1: 3D Visualization ===
with tab1:
    st.subheader("3D Trajectory & Hits")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])

    with col_ctrl1:
        # Filter by Track ID
        track_ids = sorted(tracks_df['TrackID'].unique())
        selected_track_id = st.selectbox("Select Track to View (Highlight)", [None] + list(track_ids))

    with col_ctrl2:
        # Time Slider (Playback)
        # Find min/max scan
        min_scan = int(tracks_df['ScanNum'].min()) if not tracks_df.empty else 0
        max_scan = int(tracks_df['ScanNum'].max()) if not tracks_df.empty else 100

        enable_playback = st.checkbox("Enable Playback Mode")
        if enable_playback:
            current_scan = st.slider("Time (Scan Number)", min_scan, max_scan, min_scan)
        else:
            current_scan = max_scan

    # Prepare data for plotting
    plot_tracks = tracks_df
    plot_hits = hits_df

    if enable_playback:
        # Filter Hits to current scan
        plot_hits = plot_hits[plot_hits['ScanNum'] == current_scan]

        # Filter Tracks: Show full history up to current scan? Or just current point?
        # Usually full history is nice.
        plot_tracks = plot_tracks[plot_tracks['ScanNum'] <= current_scan]

        # If we want to show the "head" of the track differently, we can do that in visualizer,
        # but for now standard plot is okay.

    if selected_track_id:
        # Filter Hits to only those in the track (and scan if playback)
        track_hits_subset = merged_df[merged_df['TrackID'] == selected_track_id]

        if enable_playback:
            track_hits_subset = track_hits_subset[track_hits_subset['ScanNum'] == current_scan]

        fig = plot_3d_tracks(plot_tracks, hits_df=track_hits_subset, show_hits=True, x_range=x_range, y_range=y_range, z_range=z_range)

        # Add visual lines connecting Track Points to Hits
        t_data = plot_tracks[plot_tracks['TrackID'] == selected_track_id]

        import plotly.graph_objects as go

        # Add connection lines
        connector_x = []
        connector_y = []
        connector_z = []

        # If playback is enabled, we only want connections for the current scan (if track exists in this scan)
        if enable_playback:
            t_scan = t_data[t_data['ScanNum'] == current_scan]
            # If track exists in this scan
            if not t_scan.empty:
                # Use just this row
                iter_data = t_scan
            else:
                iter_data = pd.DataFrame()
        else:
            iter_data = t_data

        for _, t_row in iter_data.iterrows():
            scan = t_row['ScanNum']
            h_data = track_hits_subset[track_hits_subset['ScanNum'] == scan]

            tx, ty, tz = t_row['X'], t_row['Y'], t_row['Z']

            for _, h_row in h_data.iterrows():
                hx, hy, hz = h_row['X'], h_row['Y'], h_row['Z']
                connector_x.extend([tx, hx, None])
                connector_y.extend([ty, hy, None])
                connector_z.extend([tz, hz, None])

        if connector_x:
            fig.add_trace(go.Scatter3d(
                x=connector_x, y=connector_y, z=connector_z,
                mode='lines',
                line=dict(color='rgba(200,200,200,0.5)', width=1),
                name='Association'
            ))

    else:
        # Show all hits (filtered by playback)
        fig = plot_3d_tracks(plot_tracks, plot_hits, show_hits=True, x_range=x_range, y_range=y_range, z_range=z_range)

    st.plotly_chart(fig, use_container_width=True)

    # 2D Views
    st.markdown("### 2D Projections")
    col2d_1, col2d_2 = st.columns(2)

    with col2d_1:
        # For 2D views, use same filtered hits
        hits_2d = track_hits_subset if selected_track_id else plot_hits

        # Pass full tracks list if no specific track selected, but if selected, pass only that track for clearer connection lines
        tracks_2d = plot_tracks

        fig_top = plot_2d_view(tracks_2d, hits_2d, view_type='top', x_range=x_range, y_range=y_range,
                               show_connections=(selected_track_id is not None))
        st.plotly_chart(fig_top, use_container_width=True)

    with col2d_2:
        hits_2d = track_hits_subset if selected_track_id else plot_hits
        tracks_2d = plot_tracks

        fig_side = plot_2d_view(tracks_2d, hits_2d, view_type='side', x_range=x_range, z_range=z_range,
                                show_connections=(selected_track_id is not None))
        st.plotly_chart(fig_side, use_container_width=True)

# === Tab 2: Doppler Analysis ===
with tab2:
    st.subheader("Doppler Analysis")

    if selected_track_id:
        track_hits_subset = merged_df[merged_df['TrackID'] == selected_track_id]

        if track_hits_subset.empty:
            st.info("No hits associated with this track.")
        else:
            # 1. STFT (Spectrogram)
            spectrogram, time_labels = prepare_doppler_spectrogram(track_hits_subset, doppler_cols)
            st.write("### Track Spectrogram (STFT)")
            st.plotly_chart(plot_doppler_spectrogram(spectrogram, time_labels), use_container_width=True)

            # 2. Point-wise Inspection
            st.write("### Point-wise Doppler Inspection")
            selected_scan = st.selectbox("Select Scan Number", sorted(track_hits_subset['ScanNum'].unique()))

            scan_hits = track_hits_subset[track_hits_subset['ScanNum'] == selected_scan]

            cols = st.columns(2)
            for idx, (i, hit) in enumerate(scan_hits.iterrows()):
                # Extract doppler bin values
                d_vals = hit[doppler_cols].values

                with cols[idx % 2]:
                    st.plotly_chart(plot_doppler_spectrum(d_vals, title=f"Hit {hit['hitId']} (Scan {selected_scan})"), use_container_width=True)

    else:
        st.info("Please select a Track ID in the sidebar or Tab 1 to view Doppler analysis.")

# === Tab 3: AI Classification ===
with tab3:
    st.subheader("AI Object Classification")

    if st.button("Run Classification Analysis"):
        with st.spinner("Analyzing Track Features..."):
            classified_tracks, feature_df = analyze_tracks_ai(tracks_df)

        st.success("Analysis Complete")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### Cluster Groups")
            st.dataframe(feature_df[['TrackID', 'Cluster', 'AvgV', 'AvgZ']])

        with col2:
            st.write("### Cluster Visualization")
            # Scatter plot of features
            import plotly.express as px
            fig_ai = px.scatter(
                feature_df,
                x='AvgV', y='AvgZ',
                color='Cluster',
                hover_data=['TrackID'],
                title="Clustering Result: Velocity vs Altitude"
            )
            st.plotly_chart(fig_ai, use_container_width=True)

        st.markdown("""
        **Methodology:**
        - **Features**: Average Velocity, Max Altitude, Velocity Standard Deviation (Maneuver), Average RCS.
        - **Algorithm**: K-Means Clustering (k=3).
        - **Purpose**: Group unidentified tracks into potential classes (e.g., Birds, Drones, Aircraft) based on kinematics.
        """)

# === Tab 4: Data Export ===
with tab4:
    st.subheader("Data Export")

    if selected_track_id:
        st.write(f"Export data for **Track {selected_track_id}**")

        # Prepare Data
        t_export = tracks_df[tracks_df['TrackID'] == selected_track_id]
        h_export = merged_df[merged_df['TrackID'] == selected_track_id]

        # Convert to CSV
        t_csv = t_export.to_csv(index=False).encode('utf-8')
        h_csv = h_export.to_csv(index=False).encode('utf-8')

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label=f"Download Track {selected_track_id} Data",
                data=t_csv,
                file_name=f"track_{selected_track_id}.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                label=f"Download Hits for Track {selected_track_id}",
                data=h_csv,
                file_name=f"hits_track_{selected_track_id}.csv",
                mime="text/csv"
            )

    else:
        st.info("Select a Track to enable export options.")
