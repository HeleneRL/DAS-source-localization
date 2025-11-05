"""
Functions that handle GNSS coordinates.
"""

import json
import os

import numpy as np
from scipy.signal import lfilter

from .exceptions import CSNotSupportedError
from .saveandload import load_cable_geometry, load_source_table
from .utils import dt2num

_OWN_DIR = os.path.realpath(os.path.dirname(__file__))


def to_cartesian(lat, lon, alt=0, r=6378000):
    """Convert GPS coordinates to Cartesian coordinates.

    The basis vectors for the Cartesian coordinate system are defined such that

    .. math::

            \\Theta(\\hat{x}) &= (0, \\pi/2) \\\\
            \\Theta(\\hat{y}) &= (\\pi/2, \\text{undef.}) \\\\
            \\Theta(\\hat{z}) &= (0, 0),

    where :math:`\\Theta(x)` gives the elevation and azimuth angles of
    :math:`x`, using a line segment from the sphere's centre to the sphere's
    surface at (0.N, 0.E) as reference. In other words, the basis vectors
    :math:`\\hat{x}, \\hat{y}, \\hat{z}` point
    to (0.N, 90.E), (90.N, ---) and (0.N, 0.E), respectively.

    :param lat: Latitude coordinates in degrees.
    :type lat: array_like
    :param lon: Longitude coordinates in degrees.
    :type lon: array_like
    :param alt: Height over sea level in metres. Assumes sea level if left out.
    :type alt: array_like
    :param r: Radius of the celestial body of interest, expressed in metres.
        Defaults to Earth's radius.
    :type r: float
    :returns: An (..., 3) array of the Cartesian representation of the
        provided coordinates.

    """
    complex_az = np.exp(1j*np.radians(lon))
    complex_el = np.exp(1j*np.radians(lat))
    x = complex_az.imag*complex_el.real
    y = complex_el.imag
    z = complex_az.real*complex_el.real
    return np.atleast_2d(alt+r).T * np.stack([x, y, z], axis=lat.ndim)


def fix_depth(lat, lon, alt, missing_value=0, starting_at=1):
    """Estimate missing altitude points.

    Takes the latitude and longitude values, calculates distance assuming
    straight line segments between points and setting altitude to zero and
    interpolates the depth as a function of distance from start.

    :param lat: Latitude coordinates in degrees.
    :type lat: 1-D sequence of floats
    :param lon: Longitude coordinates in degrees.
    :type lon: 1-D sequence of floats
    :param alt: Height over sea level in metres.
    :type alt: 1-D sequence of floats
    :param missing_value: Which altitude is considered missing data.
        Default is 0.
    :type missing_value: float
    :param starting_at: Where the missing data are known to begin.
        Defaults to 1, i.e. the second data point and on may have missing data.
    :type starting_at: int
    :returns: Altitude coordinates with missing values interpolated.
    """
    distance_2d = distance_gps(lat, lon)
    sus_alt = np.hstack([np.zeros(starting_at, dtype='bool'),
                         alt[starting_at:] == missing_value])
    unknown_dist = distance_2d[sus_alt]
    interp_dist = np.interp(unknown_dist,
                            distance_2d[~sus_alt],
                            alt[~sus_alt])
    alt[sus_alt] = interp_dist
    return alt


def distance_gps(lat, lon, alt=0, r=6378000):
    """Distance along a sequence of GPS coordinates.

    :param lat: Latitude coordinates in degrees.
    :type lat: 1-D sequence of floats
    :param lon: Longitude coordinates in degrees.
    :type lon: 1-D sequence of floats
    :param alt: Height over sea level in metres.
    :type alt: 1-D sequence of floats
    :param r: Radius of the celestial body of interest in metres.
        Defaults to Earth's radius.
    :type r: float, optional
    :returns: Cumulative distance along the sequence of coordinates in metres.

    """
    points = to_cartesian(lat, lon, alt, r)
    segments = np.sqrt(np.sum(np.diff(points, axis=-2)**2, axis=-1))
    return np.cumsum(np.insert(segments, 0, 0))


def distance_gps_2(source, target, r=6378000, also_cosines=True,
                   smooth_corners=5):
    """Distance between source and target.

    :param source: Source positions, preferably a transducer.
        Last axis (lon, lat, alt).
    :type source: shape-(N,3) array
    :param target: Target positions, preferably a cable.
        Last axis (lon, lat, alt).
    :type target: shape-(M,3) array
    :param r: Radius of the celestial body in metres.
        Defaults to Earth's radius.
    :type r: float, optional
    :param also_cosines: Whether to output cosines of grazing angles, too.
    :type also_cosines: bool
    :param smooth_corners: How much to smooth abrupt changes in direction
        in the target.
    :type smooth_corners: float
    :returns: Distance between all source positions and target positions in
        metres. If also returning angles, it is the second output.

    """
    source_cartesian = to_cartesian(source[:, 1],
                                    source[:, 0],
                                    source[:, 2],
                                    r)
    target_cartesian = to_cartesian(target[:, 1],
                                    target[:, 0],
                                    target[:, 2],
                                    r)
    distance_vectors = source_cartesian[..., None]-target_cartesian.T
    distances = np.sqrt(np.sum(distance_vectors**2, axis=-2))

    if also_cosines:
        unit_distances = distance_vectors/distances[:, None, :]
        target_forward = np.diff(target_cartesian, axis=0)
        target_forward = np.vstack((target_forward[0], target_forward))
        # want same length
        if smooth_corners > 1:
            target_forward = lfilter(np.ones(smooth_corners)/smooth_corners,
                                     1,
                                     target_forward,
                                     axis=0)

        target_forward /= np.sqrt(np.sum(target_forward**2,
                                         axis=-1,
                                         keepdims=True))
        direction_cos = np.sum(unit_distances*target_forward.T, axis=-2)
        return distances, direction_cos

    return distances


def interpolate_coordinates(distances, known_distances, known_coords,
                            input_cs='Cartesian', output_cs='Cartesian',
                            r=6378000):
    """Interpolate coordinates along a segment.

    :param distances: Desired distances to interpolate coordinates.
    :type distances: 1-D sequence of floats
    :param known_distances: Known distances to interpolate.
    :type known_distances: 1-D sequence of floats
    :param known_coords: Known coordinates on the form specified by
        ``input_cs``.
    :type known_coords: 2-D sequence of floats
    :param input_cs: Coordinate system of input.
    :type input_cs: {'Cartesian', 'GPS'}
    :param output_cs: Coordinate system of output.
    :type output_cs: {'Cartesian', 'GPS'}
    :param r: Radius of the sphere in metres. Relevant when using GPS
        coordinates. Defaults to Earth's radius.
    :type r: float, optional
    :returns: Interpolated coordinates on the requested form.
    :raises CSNotSupportedError: if an unsupported CS is specified.

    .. note ::

        The interpolation is always carried out in Cartesian coordinates, which
        means the interpolation is linear. Curvature is ignored.

    """
    if input_cs not in ["Cartesian", "GPS"]:
        raise CSNotSupportedError("expected input CS to be 'Cartesian' or "
                                  f"'GPS', but got {input_cs}")
    elif output_cs not in ["Cartesian", "GPS"]:
        raise CSNotSupportedError("expected output CS to be 'Cartesian' or "
                                  f"'GPS', but got {output_cs}")

    if input_cs == "GPS":
        known_coords = to_cartesian(known_coords[:, 0], known_coords[:, 1],
                                    known_coords[:, 2], r)

    out = np.zeros([distances.shape[-1], 3])
    for it in range(out.shape[-1]):
        out[:, it] = np.interp(distances, known_distances, known_coords[:, it])

    if output_cs == "GPS":
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        return to_gps(x, y, z, r)

    return out


def to_gps(x, y, z, r=6378000):
    """Convert Cartesian coordinates to GPS coordinates.

    :param x: :math:`x` coordinates.
    :type x: array_like
    :param y: :math:`y` coordinates.
    :type y: array_like
    :param z: :math:`z` coordinates.
    :type z: array_like
    :param r: Radius of the celestial body. Significant for altitude.
        Defaults to Earth's radius.
    :type r: float
    :returns: (..., 3) array of GPS coordinates.
    """
    return np.stack([np.degrees(np.arctan2(x, z)),  # longitude
                     np.degrees(np.arctan(y/np.sqrt(x**2+z**2))),  # latitude
                     np.sqrt(x**2 + y**2 + z**2)-r],  # altitude
                    axis=x.ndim)


def get_transmission_locations(source_gps, run, force_calculate=False):
    """Find where the signals were transmitted, one by one.

    :param source_gps: Timetable with position data.
    :type source_gps: DataFrame with columns 'datetime', 'lat', 'lon'
    :param run: Run data for the requested run.
        See :py:func:`dasprocessor.constants.get_run`.
    :type run: dict
    :return: Array of GPS coordinates of each transmission.

    .. caution ::
        Time-varying depth is not yet supported.

    """
    if "source_position_file" in run.keys() and not force_calculate:
        # no need to repeat work here
        return np.load(_OWN_DIR + f'/../resources/'
                       f'{run["source_position_file"]}')['elly']
    tx_times = source_gps['datetime'].apply(lambda x:
                                            dt2num(x, True)).to_numpy()
    # TODO implement a way to handle time-varying depth (e.g. 2024-05-03 run 3)
    altitude = run['transmitter_altitude']
    elly_lat = source_gps['lat'].to_numpy()
    elly_lon = source_gps['lon'].to_numpy()
    samples_cartesian = to_cartesian(elly_lat, elly_lon, altitude)
    time_start = np.array([3600, 60, 1]) @ np.array(run["sequence_start"])
    target_times = np.arange(run["sequence_count"])*run["sequence_period"]\
        / run["sample_rate"] + time_start
    tx_coordinates = np.column_stack([np.interp(target_times,
                                                tx_times,
                                                samples_cartesian[:, it])
                                      for it in range(samples_cartesian
                                                      .shape[-1])])
    return to_gps(tx_coordinates[:, 0],
                  tx_coordinates[:, 1],
                  tx_coordinates[:, 2])


def main():
    """Test the functions.

    Not expected to be run.
    """
    from .constants import get_run
    import matplotlib.pyplot as plt

    cablegps = load_cable_geometry('../resources/cable-layout.json')

    corrected_alt = fix_depth(cablegps[:, 0], cablegps[:, 1], cablegps[:, 2])
    cablegps[:, 2] = corrected_alt
    distance_3d = distance_gps(cablegps[:, 0], cablegps[:, 1], cablegps[:, 2])

    myrun = get_run("2024-05-03", 1)
    wantedpts = np.arange(1200)*myrun['channel_distance']
    gpspts = interpolate_coordinates(wantedpts, distance_3d, cablegps,
                                     "GPS", "GPS")
    print(gpspts)

    txtable = load_source_table('../resources/source-position.csv')
    print(txtable['datetime'].loc[500])
    txspots = get_transmission_locations(txtable, myrun, True)
    print(txspots)
    distance, cosines = distance_gps_2(txspots, gpspts, smooth_corners=10)
    plt.pcolor(cosines, cmap='RdBu')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
