# Radar Data Analysis Tool

This application analyzes and visualizes radar track and hit data.

## Features
- **3D Visualization**: Interactive orbit and hit display.
- **Doppler Analysis**: Spectrogram (STFT) and point-wise Doppler inspection.
- **AI Analysis**: K-Means clustering of tracks based on kinematic features.
- **Data Export**: Download processed Hit/Track data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```
2. Upload your Hit, Track, and Relation CSV files in the sidebar.
   - If no files are uploaded, dummy data is used for demonstration.
3. Adjust Radar Geometry (Height, Tilt, Direction) to correct Hit coordinates.
4. Explore the tabs for visualization and analysis.

## File Formats

- **Hit Data**: Tab/Space separated. Columns after header are treated as Doppler bins.
- **Track Data**: Standard CSV format with columns like `X`, `Y`, `Z`, `VX`, `VY`, `VZ`, `RCS`.
- **Relation Data**: Maps `TrackID` to a comma-separated list of `HitList`.
