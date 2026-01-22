import pandas as pd
import io

def load_hit_data(file_buffer):
    """
    Loads Hit Data.
    Handles dynamic columns (Doppler bins) that exceed the header length.
    """
    # Read the first line to get headers
    # file_buffer can be a file object or bytes. Streamlit uploader returns a BytesIO-like object.

    # We need to read lines. Streamlit UploadedFile is seekable.
    file_buffer.seek(0)
    first_line = file_buffer.readline().decode('utf-8').strip()
    headers = [h.strip() for h in first_line.split('\t')] # Assuming tab separated based on prompt

    # Check if headers are separated by spaces, tabs, or commas
    if len(headers) > 1:
        sep = '\t'
    else:
        # Try comma
        headers = [h.strip() for h in first_line.split(',')]
        if len(headers) > 1:
            sep = ','
        else:
            # Try splitting by whitespace
            headers = [h.strip() for h in first_line.split()]
            sep = r'\s+'

    # Now read the full data
    file_buffer.seek(0)

    # Strategy: Read into a list of lines, then parse.
    # Or use pandas with 'header=0' and then fix the extra columns?
    # Pandas will drop extra data or error if header has fewer cols than data.
    # Actually, pandas `read_csv` has `names` parameter.
    # But we don't know the number of doppler bins beforehand.

    # Let's inspect the first data row to determine column count.
    first_data_line = file_buffer.readline().decode('utf-8').strip() # Skip header
    first_data_line = file_buffer.readline().decode('utf-8').strip() # First data

    if not first_data_line:
        # Empty file
        return pd.DataFrame(columns=headers)

    if sep == '\t':
        data_cols = len(first_data_line.split('\t'))
    elif sep == ',':
        data_cols = len(first_data_line.split(','))
    else:
        data_cols = len(first_data_line.split())

    num_headers = len(headers)
    num_doppler = data_cols - num_headers

    # Create new header list
    new_headers = headers + [f"doppler_{i}" for i in range(num_doppler)]

    file_buffer.seek(0)
    df = pd.read_csv(file_buffer, sep=sep, names=new_headers, header=0)

    # Collect all doppler columns into a single column of lists/arrays for easier processing?
    # Or keep them as columns?
    # For STFT, having them as an array in a single cell is nice, or just filter columns.
    # Let's keep them as columns but identify them.

    return df, [f"doppler_{i}" for i in range(num_doppler)]

def load_track_data(file_buffer):
    file_buffer.seek(0)
    # Detect separator
    first_line = file_buffer.readline().decode('utf-8')
    if '\t' in first_line:
        sep = '\t'
    elif ',' in first_line:
        sep = ','
    else:
        sep = r'\s+'
    file_buffer.seek(0)
    return pd.read_csv(file_buffer, sep=sep)

def load_relation_data(file_buffer):
    file_buffer.seek(0)
    first_line = file_buffer.readline().decode('utf-8')
    # Relations might be comma or tab.
    # Example: "TrackID HitList" could be tab.
    # Data: "1, 123,456" or "1 \t 123,456"
    if '\t' in first_line:
        sep = '\t'
    elif ',' in first_line:
        # Caution: The HitList column itself contains commas.
        # But usually standard CSV handles quoted strings "123,456".
        # Or it might be semicolon separated?
        # Given the prompt, let's assume standard CSV or Tab.
        sep = ','
    else:
        sep = r'\s+'

    file_buffer.seek(0)
    df = pd.read_csv(file_buffer, sep=sep)

    # Parse HitList
    def parse_hitlist(x):
        if pd.isna(x): return []
        return [int(h) for h in str(x).split(',')]

    if 'HitList' in df.columns:
        df['HitList'] = df['HitList'].apply(parse_hitlist)

    return df

def merge_data(tracks, hits, relations, doppler_cols):
    """
    Merges Track, Hit, and Relation data.
    Returns a dictionary or object with linked data.
    """
    # 1. Explode Relations to map TrackID -> HitID
    rel_exploded = relations.explode('HitList')
    rel_exploded = rel_exploded.rename(columns={'HitList': 'hitId'})

    # 2. Merge with Hits
    # Ensure hitId types match
    if not rel_exploded.empty and not hits.empty:
        rel_exploded['hitId'] = rel_exploded['hitId'].astype(hits['hitId'].dtype)

        # Merge Hits into Relations
        track_hits = pd.merge(rel_exploded, hits, on='hitId', how='left')

        # Now we have rows of (TrackID, Hit Data...)
        # We can group by TrackID to get all hits for a track
        return track_hits
    return pd.DataFrame()
