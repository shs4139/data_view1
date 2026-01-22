import numpy as np

# Constant provided by user
RANGE_CONVERSION_FACTOR = 7.629395

def calculate_coordinates(range_vals, az_deg, el_deg, radar_height, radar_direction, radar_tilt=0.0):
    """
    Converts raw radar data to Cartesian coordinates based on user-provided logic.

    Args:
        range_vals: Raw range data.
        az_deg: Azimuth in degrees.
        el_deg: Elevation in degrees.
        radar_height: Height of the radar (m).
        radar_direction: Direction offset of the radar (degrees, adds to Azimuth).
        radar_tilt: Tilt offset of the radar (degrees, adds to Elevation).

    Returns:
        x, y, z arrays in meters.
    """
    # 1. Apply Range Conversion
    r = range_vals * RANGE_CONVERSION_FACTOR

    # 2. Apply Angles (Degrees to Radians)
    # Azimuth = Input + Direction
    # Elevation = Input + Tilt
    az = np.radians(az_deg + radar_direction)
    el = np.radians(el_deg + radar_tilt)

    # 3. Spherical to Cartesian (User Formula)
    # x = r * cos(el) * sin(az)
    # y = r * cos(el) * cos(az)
    # z = r * sin(el) + height

    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el) + radar_height

    return x, y, z
