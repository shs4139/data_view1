import numpy as np

def spherical_to_cartesian(r, az, el):
    """
    Converts spherical coordinates (Range, Azimuth, Elevation) to Cartesian (X, Y, Z).
    Azimuth and Elevation are in degrees.
    Coordinate System Assumption:
    - X: East
    - Y: North
    - Z: Up
    - Azimuth: 0 is North (Y), 90 is East (X) -> This is typical Compass.
      However, in math, Az 0 is usually X-axis.
      Let's assume standard math physics:
      x = r * cos(el) * cos(az)
      y = r * cos(el) * sin(az)
      z = r * sin(el)

      BUT, Radar Azimuth is often 0 at Boresight.
      Let's stick to standard Physics conventions first, but Azimuth 0 usually means Y (North) in navigation.
      Let's use:
      Azimuth 0 = +Y axis.
      Azimuth 90 = +X axis.

      x = r * cos(el) * sin(az)
      y = r * cos(el) * cos(az)
      z = r * sin(el)
    """
    az_rad = np.radians(az)
    el_rad = np.radians(el)

    # Using Navigation convention: Az 0 = Y (North), Az 90 = X (East)
    x = r * np.cos(el_rad) * np.sin(az_rad)
    y = r * np.cos(el_rad) * np.cos(az_rad)
    z = r * np.sin(el_rad)

    return x, y, z

def transform_radar_coords(x, y, z, height, tilt, direction):
    """
    Applies radar mounting configuration to local coordinates.
    1. Rotate by Direction (Azimuth offset).
    2. Rotate by Tilt (Elevation offset).
    3. Translate by Height (Z offset).

    Order matters: Typically, you rotate the frame, then translate.
    """

    # 1. Apply Direction (Azimuth rotation around Z axis)
    # direction is degrees offset from North? Or Boresight direction?
    # Let's assume 'direction' adds to the azimuth.
    dir_rad = np.radians(direction)

    # Rotate around Z (Yaw)
    # x' = x cos(th) - y sin(th)
    # y' = x sin(th) + y cos(th)
    # But if we treat 'direction' as the bore-sight azimuth:
    # We essentially rotate the vector (x,y) by the direction angle.
    x_rot = x * np.cos(dir_rad) + y * np.sin(dir_rad)
    y_rot = -x * np.sin(dir_rad) + y * np.cos(dir_rad)
    # Wait, rotation matrix for CCW is:
    # [cos -sin]
    # [sin  cos]
    # If we simply add the angle to the azimuth in spherical_to_cartesian, it's easier.
    # But doing it in Cartesian allows for independent tilt.

    # Let's simplify:
    # If the user provides Direction/Tilt/Height, they likely want to correct the Hit data
    # to match the Track data (which is usually in a Global or Local-ENU frame).

    # Let's assume the Hit data (r, az, el) is RELATIVE to the Radar Boresight.
    # And we want to convert to the World Frame where the Tracks live.

    # Step 1: Convert (r, az, el) to Cartesian in Radar Frame.
    # (Done before calling this, passed as x, y, z)

    # Step 2: Rotate Radar Frame to World Frame.
    # Rotation 1: Tilt (Pitch) - usually around X-axis if Az=0 is Y.
    tilt_rad = np.radians(tilt)
    # Pitch up/down
    # y' = y cos(tilt) - z sin(tilt)
    # z' = y sin(tilt) + z cos(tilt)
    y_tilt = y * np.cos(tilt_rad) - z * np.sin(tilt_rad)
    z_tilt = y * np.sin(tilt_rad) + z * np.cos(tilt_rad)
    x_tilt = x

    # Rotation 2: Direction (Yaw) - around Z-axis.
    # x'' = x' cos(dir) + y' sin(dir)
    # y'' = -x' sin(dir) + y' cos(dir)  <-- This rotates the point.

    # If 'Direction' is the angle the radar is facing (Azimuth of Boresight):
    # Then we rotate the point BY that angle.
    x_final = x_tilt * np.cos(dir_rad) + y_tilt * np.sin(dir_rad) # Check signs
    y_final = -x_tilt * np.sin(dir_rad) + y_tilt * np.cos(dir_rad)

    # Wait, standard rotation matrix Rz(theta) * v:
    # x' = x cos - y sin
    # y' = x sin + y cos
    # If direction is 90 deg (facing East), and we have a point at (0, 10, 0) (Ahead 10m):
    # We want it to be at (10, 0, 0) (East 10m).
    # Rz(-90)?
    # Let's assume standard rotation:
    x_final = x_tilt * np.cos(dir_rad) - y_tilt * np.sin(dir_rad)
    y_final = x_tilt * np.sin(dir_rad) + y_tilt * np.cos(dir_rad)
    z_final = z_tilt

    # Step 3: Translate
    z_final += height

    return x_final, y_final, z_final
