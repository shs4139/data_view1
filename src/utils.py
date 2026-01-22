import pandas as pd
import numpy as np
import random
import os

def generate_dummy_data(output_dir="."):
    """
    Generates dummy Hit, Track, and Relation CSV files for testing.
    """

    # 1. Define Headers
    hit_headers = [
        "LogTime", "ID", "ScanNum", "dwellNum", "PanelId", "hitId", "valid",
        "range", "doppler_bin", "power", "azimuth", "elevation", "hitMap",
        "sumRe", "sumIm", "delazRe", "delazIm", "delelRe", "delelIm",
        "slbRe", "slbIm", "slb", "DeltaAz", "DeltaEl"
    ]

    track_headers = [
        "LogTime", "ID", "TrackID", "ScanNum", "Status", "Score", "Cost",
        "Doppler", "TV_sec", "TV_usec", "X", "Y", "Z", "VX", "VY", "VZ",
        "Power", "RCS", "AssocFlag", "UpdateFlag", "UpdateDist", "Class",
        "Pd", "SumZ", "SumZ2", "SumR", "SumR2", "SumV"
    ]

    relation_headers = ["TrackID", "HitList"]

    # 2. Simulation Settings
    num_tracks = 5
    time_steps = 20
    doppler_bins_count = 70 # Increased to test truncation (limit 60)

    hits_data = []
    tracks_data = []
    relations_data = []

    hit_id_counter = 10000

    for t_id in range(1, num_tracks + 1):
        # Initial state for track
        x, y, z = np.random.uniform(-1000, 1000), np.random.uniform(500, 2000), np.random.uniform(100, 500)
        vx, vy, vz = np.random.uniform(-20, 20), np.random.uniform(-10, 10), np.random.uniform(-2, 2)

        track_hits_map = [] # Store list of hits per scan to build relations later if needed per scan,
                            # but the relation file format seems to link TrackID -> All Hits or Hits per frame?
                            # The prompt says: "TrackID, HitList".
                            # And the example shows: "1 \t 739210,739211..."
                            # This implies one row per TrackID containing ALL hits for that track across time?
                            # Or one row per scan? The example only shows one row for TrackID 1.
                            # I will assume it collects ALL hits for that track ID.

        all_track_hit_ids = []

        for step in range(time_steps):
            # Update Track Position
            x += vx
            y += vy
            z += vz

            scan_num = step + 1
            log_time = f"14:22.{step}"

            # Create Track Row
            track_row = [
                log_time, 100, t_id, scan_num, 1, 100, 0, 0, 0, 0,
                x, y, z, vx, vy, vz,
                200000, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            tracks_data.append(track_row)

            # Generate associated Hits (Clutter around track)
            num_hits = random.randint(3, 8)
            current_scan_hit_ids = []

            for _ in range(num_hits):
                hit_id = hit_id_counter
                hit_id_counter += 1
                current_scan_hit_ids.append(hit_id)
                all_track_hit_ids.append(hit_id)

                # Hit position (Spherical approx from Cartesian + noise)
                r = np.sqrt(x**2 + y**2 + z**2) + np.random.normal(0, 5)
                az = np.degrees(np.arctan2(x, y)) + np.random.normal(0, 0.5) # Assuming Y is North/Forward
                el = np.degrees(np.arcsin(z / r)) + np.random.normal(0, 0.5)

                # Doppler Data (32 bins) - Simulate a peak
                doppler_vals = np.random.normal(100, 50, doppler_bins_count).astype(int).tolist()
                peak_idx = random.randint(10, 20)
                doppler_vals[peak_idx] = 10000 # Peak

                hit_row = [
                    log_time, 100, scan_num, 0, 0, hit_id, 1,
                    r, peak_idx, 200000, az, el, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ] + doppler_vals # Append Doppler bins at the end

                hits_data.append(hit_row)

        # Add Relation Row
        relations_data.append([t_id, ",".join(map(str, all_track_hit_ids))])

    # 3. Create DataFrames
    # Hit DataFrame: Headers + Extra Doppler Cols
    # Note: The prompt says "Columns without headers are Doppler".
    # So we write the fixed headers, and then the data rows will have more columns.
    # We shouldn't add headers for the doppler columns in the CSV file itself to match requirements.

    # Save Hits
    with open(os.path.join(output_dir, "hit_data.csv"), "w") as f:
        f.write("\t".join(hit_headers) + "\n")
        for row in hits_data:
            f.write("\t".join(map(str, row)) + "\n")

    # Save Tracks
    df_tracks = pd.DataFrame(tracks_data, columns=track_headers)
    df_tracks.to_csv(os.path.join(output_dir, "track_data.csv"), sep="\t", index=False)

    # Save Relations
    df_relations = pd.DataFrame(relations_data, columns=relation_headers)
    df_relations.to_csv(os.path.join(output_dir, "relation_data.csv"), sep="\t", index=False)

    print("Dummy data generated.")

if __name__ == "__main__":
    generate_dummy_data()
