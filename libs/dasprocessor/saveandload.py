"""
Input and output tailored for DAS processing.
"""

import json
import os

import numpy as np
from pandas import DataFrame, read_csv
from scipy.io import wavfile
from scipy.io import savemat, loadmat
from scipy.signal import sosfilt

from .simpleDASreader8 import load_DAS_files
from .exceptions import NotInCacheError, NoSuchPreambleError
from .utils import strip_trailing, get_bandpass_filter, to_time_list

_OWN_DIR = os.path.realpath(os.path.dirname(__file__))


def load_builtin_detection_signal(kind, rate):
    """Load predefined detection signal.

    :param kind: Which signal to load.
    :type kind: str
    :param rate: Sampling rate of the signal.
    :type rate: int
    :returns: The detection signal of the specified kind and with given
        sampling rate.
    :raises NoSuchPreambleError: if there is no predefined detection signal
        under that name and sampling rate.

    """
    try:
        return wavfile.read(_OWN_DIR + f"/../resources/{kind}-{rate}.wav")[1]
    except FileNotFoundError:
        raise NoSuchPreambleError(f"detection signal of type {kind} and sample"
                                  f" rate {rate} is not predefined; check "
                                  "spelling or use "
                                  "`load_user_detection_signal`.")


def load_user_detection_signal(path):
    """Load user-defined detection signal.

    Wrapper for :py:func:`scipy.io.wavfile.read`.

    :param path: Path to the user-defined detection signal.
    :type path: path-like
    :returns: 2-tuple with ``(rate, signal)``.

    """
    return wavfile.read(path)


def load_janus_autocorrelation(band, which, rate=25000, force_overwrite=False):
    """Load autocorrelation spectrum of a JANUS packet.

    Best used with :py:func:`dasprocessor.sensitivity.get_sensitivity`.

    :param band: Frequency band of JANUS signal.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'}
    :param which: Packet number.
    :type which: int
    :param rate: Sampling rate.
    :type rate: int, optional
    :param force_overwrite: If ``True``, raises a
        :py:exc:`NotInCacheError` even if the file exists.
    :type force_overwrite: bool
    :returns: Autocorrelation spectrum.

    """
    if force_overwrite:
        raise NotInCacheError("user requests to generate spectrum again")
    return np.load(_OWN_DIR
                   + f"/../resources/signal-psds/janus-{band}-x{which:02x}-"
                   f"{rate}.npz")['y']


def load_hydrophone(fname, date, start, stop, step=300):
    """Load hydrophone recordings.

    The filename should match the pattern
    ``/([A-Za-z0-9_-])+?_(\\d+\\d{2}\\d{2})_(\\d{2})(\\d{2})(\\d{2})\\.wav/``,
    where the capturing groups are (1) the base filename from ``fname``,
    (2) the date from ``date`` on the form yyyymmdd and (3)-(5) the hours,
    minutes and seconds, respectively. For example, specifying
    ``fname='foo'``, ``date='20201029'`` and ``start=(12, 34, 56)`` should give
    the filename ``foo_20201029_123456.wav``

    :param fname: Base name of hydrophone recordings, including any
        relative path.
    :type fname: path-like
    :param date: Date part of recording. Accepts ``yyyy-mm-dd``
        and ``yyyymmdd`` format. (Actually accepts arbitrary hyphens.
        However, the format after removing all
        hyphens *is* expected as ``yyyymmdd``.)
    :type date: str
    :param start: 3-tuple of (hours, minutes, seconds) of the first hydrophone
        data section.
    :type start: 3-tuple of ints
    :param stop: 3-tuple of (hours, minutes, seconds) one ``step`` beyond the
        start of the last hydrophone data section.
    :type stop: 3-tuple of ints
    :param step: Length of one hydrophone data section in seconds.
    :type step: int, optional
    :returns: A dictionary with the following entries.

        * **fs** -- The sampling rate of the hydrophone recordings.
        * **y** -- The length of hydrophone recordings starting at ``start``
            and ending immediately before ``stop`` If ``stop`` is not an
            integer multiple of ``step`` seconds past ``start``, ``stop``
            is effectively rounded up to the next time that satisfies
            this condition.

    """
    if '-' in date:
        date = date.replace('-', '')

    hydrotime = to_time_list(start, stop, step=step)
    hydrosig = []
    for time in hydrotime:
        rate, hydroblock = wavfile.read(f'{fname}_{date}_{time}.wav')
        hydrosig.append(hydroblock)

    return {'fs': rate, 'y': np.hstack(hydrosig)}


def load_interrogator_data(basename, start, stop, step=10,
                           channels=slice(None), on_fnf="raise", out="mat",
                           verbose=False, cachepath=None, filter_band=None,
                           force_overwrite=False, direct_mode=False):
    """Load interrogator data.

    Default behaviour is to raise an exception if the requested data are not
    available directly on disk. The behaviour can be overridden by setting
    ``on_fnf`` to one of `"cache"` or `"nocache"`, in which case the loading
    order is (1) filtered data, if requesting that; (2) cached data, possibly
    to be filtered; (3) from HDF5 files, possibly to be filtered.

    .. warning ::
            The total trial data could easily exceed the amount of RAM
            plus swap you have available. Be doubly careful with the number of
            channels you load at once. The processing demands a lot of memory
            already at ten channels. I never loaded more than twelve channels.

    :param basename: Path to the interrogator data directory.
    :type basename: path-like
    :param start: Start of time sequence. Each
        (3,)-sequence is expected on the form `(hours, minutes, seconds)`.
    :type start: 3-tuple of ints
    :param stop: End of time sequence, exclusive.
    :type stop: 3-tuple of ints
    :param step: seconds between interrogator data blocks
    :type step: int, optional
    :param channels: Interrogator channels to load. Must have ``step=None``
        or ``step=1``.

        .. warning ::

            Using excessively wide-spanning slices with tall interrogator data
            and ``out="mat"`` may cause the caching process to fail due to
            2 GB size restrictions on variables in older versions of MATLAB
            data files.

    :type channels: slice
    :param on_fnf: Determines the behaviour if there is no cached version of
        the requested DAS data for the selected slice of channels. If "raise"
        is given or argument is left out, a :py:exc:`NotInCacheError` is
        raised. If "cache" is given, the desired channels are loaded from the
        raw interrogator in `hdf5` files and concatenated in time, then written
        to disk. If "nocache" is given, it behaves like "cache" except data
        are not written to disk.
    :type on_fnf: {'raise','cache','nocache'}, optional
    :param out: Output format. If "mat" or omitted, operates on saved MATLAB
        data files. If "npz", operates on saved NumPy archives.
    :type out: {'mat','npz'}, optional
    :param verbose: Give more information of what is going on if `True`.
    :type verbose: bool, optional
    :param cachepath: Directory to check for cached data. Defaults to a
        ``backups`` directory inside the directory specified by ``basename``.
    :type cachepath: path-like, optional
    :param filter_band: Passband of desired filter. If `None`, loads
        unfiltered data.
    :type filter_band: None or 2-sequence of floats, optional
    :param force_overwrite: Whether to forcibly overwrite existing cache data.
        Overrides ``on_fnf``. Useful if cached data are corrupted or wrong.
    :type force_overwrite: bool, optional
    :param direct_mode: Whether to attempt to load data directly from
        ``basename``.
        Overrides all other parameters.
    :type direct_mode: bool, optional
    :returns: Loaded data.
    :raises NotInCacheError: if the preloaded file does not exist on disk
        and ``on_fnf="raise"`` is specified.

    """
    # sanity check arguments
    if direct_mode:
        if basename.lower().endswith(".mat"):
            return loadmat(basename)
        elif basename.lower().endswith(".npz"):
            out = np.load(basename)
            return {"y": out["y"], "fs": out["fs"]}
        else:
            raise ValueError(f"file type {basename[basename.rindex('.'):]} "
                             "is not supported")

    if channels.start is None and channels.stop is None:
        # either 'channels' was not given or its 'start' and 'stop' are None
        raise ValueError("loading all channels of interrogator data is not "
                         "allowed; was 'channels' given?")
    elif channels.start is None or channels.stop is None:
        # 'channels' was given, but either 'start' or 'stop' are None
        raise ValueError("using implicit start or stop points is not allowed;"
                         " please specify them explicitly")
    elif channels.step is not None and channels.step != 1:
        raise ValueError("loading non-contiguous or reversed blocks of"
                         "channels is not supported")

    supported_filetypes = ("mat", "npz")
    if out not in supported_filetypes:
        raise ValueError(f'file type {out} is not supported')

    timelist = to_time_list(start, stop, step=step)
    basename = strip_trailing(basename)
    if cachepath is not None:
        cachepath = strip_trailing(cachepath)

    defaultbackups = f'{basename}/backups'
    choice_of_prefix = (f'filtered-{filter_band[0]}-{filter_band[1]}_'
                        if filter_band is not None else 'backup-')
    kind_str = 'filtered' if filter_band is not None else 'raw'

    # check for filtered data
    cached_filename = (f'{cachepath or defaultbackups}/{choice_of_prefix}'
                       f'{timelist[0]}-{timelist[-1]}_{channels.start}-'
                       f'{channels.stop}.{out}')
    try:
        # do not even try to load data if forcibly overwriting them
        if force_overwrite:
            on_fnf = "cache"
            raise NotInCacheError('we are forcibly overwriting cached items')

        match out:
            case "mat":
                data = loadmat(cached_filename)
            case "npz":
                data = np.load(cached_filename)
    except FileNotFoundError:
        match on_fnf:
            case "raise":
                raise NotInCacheError(f'desired {kind_str} data file was not'
                                      ' found')
            case "cache" | "nocache":
                will_cache = "no" not in on_fnf
                if filter_band is None:
                    # not filtered data, load raw instead
                    data = load_DAS_files([f'{basename}/{it}.hdf5'
                                           for it in timelist],
                                          chIndex=channels,
                                          showProgress=verbose)
                    data = {'y': data, 'fs': 1/data.meta['dt']}
                else:
                    # a recursive call to retrieve unfiltered data
                    print(f'so now we are going to load {channels}'
                          f' {will_cache}')
                    data = load_interrogator_data(basename, start, stop, step,
                                                  channels, on_fnf, out,
                                                  verbose, cachepath, None,
                                                  force_overwrite)
                    # create a bandpass filter and apply it to the DAS data
                    bandpass = get_bandpass_filter(*filter_band,
                                                   np.round(data['fs']))
                    print(data['y'].shape)
                    data['y'] = sosfilt(bandpass, data['y'], axis=0)\
                        .astype('float32')
                    print(f'ok, we got channels {channels}')
                # cache filtered or raw data to disk
                if will_cache:
                    print(f'we should now save to {cached_filename}')
                    if verbose:
                        print(f"Making backup of {kind_str} data to a "
                              f"{out.upper()} file")

                    match out:
                        case "mat":
                            savemat(cached_filename, data, do_compression=True)
                        case "npz":
                            np.savez_compressed(cached_filename, **data)
    # at last, return the data we recovered
    finally:
        return {
            'y': data.to_numpy() if isinstance(data, DataFrame) else data['y'],
            'fs': 1/data.meta['dt'] if isinstance(data, DataFrame)
            else data['fs']
        }


class NumpyEncoder(json.JSONEncoder):

    """JSON encoder with support for NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super.default(obj)


def save_peaks(file, peaks):
    """Write peak data to a JSON file.

    Wrapper for :py:func:`json.dump`.

    :param file: Target file. Will overwrite any existing file.
    :type file: path-like
    :param peaks: Peak data.
    :type peaks: dict or sequence of dicts

    .. seealso ::

        * :py:func:`dasprocessor.detection.matched_filter_detector`
    """
    with open(file, 'w') as fid:
        json.dump(peaks, fid, cls=NumpyEncoder)


def load_peaks(file):
    """Read peak data from a JSON file.

    Wrapper for :py:func:`json.load`.

    :param file: Target file.
    :type file: path-like
    :returns: Peak data.

    .. seealso ::

        * :py:func:`dasprocessor.detection.matched_filter_detector`
    """
    with open(file, 'r') as fid:
        out = json.load(fid)
    return out


def load_experiment_metadata(file):
    """Alias for :py:func:`load_peaks`. Intended for loading experiment
    metadata.

    :param file: Target file.
    :type file: path-like
    :returns: Experiment metadata.

    """
    return load_peaks(file)


def load_cable_geometry(path,
                        key_sequence=("features",
                                      0,
                                      "geometry",
                                      "coordinates"),
                        lat_lon_alt_order=[1, 0, 2]):
    """Load cable geometry from JSON file.

    :param path: Path to cable JSON file.
    :type path: path-like
    :param key_sequence: Tuple of indices to follow to reach the coordinates.
    :type key_sequence: tuple of :py:class:`collections.abc.Hashable`
    :param lat_lon_alt_order: Which order the GPS coordinates are given in
        the file. For example, the default ``[1, 0, 2]`` means
        "latitude is the second column,
        longitude is the first column,
        and altitude is the third column".
    :type lat_lon_alt_order: any permutation of [0, 1, 2]
    :returns: N-sequence of GPS coordinates on the form (lat, lon, alt) as an
        array.

    """
    with open(path, 'r') as fid:
        obj = json.load(fid)

    for k in key_sequence:
        obj = obj[k]

    return np.squeeze(obj)[:, lat_lon_alt_order]


def load_source_table(path):
    """Load source trajectory data from CSV file.

    Wrapper for :py:func:`pandas.read_csv`.

    :param path: Path to the CSV file with source trajectory data.
    :type path: str, path object or file-like object
    :returns: DataFrame with the source trajectory data.

    """
    return read_csv(path)


def load_experiment_run_table(path):
    """Load run details from CSV file. Wrapper for :py:func:`pandas.read_csv`.

    The ``run`` column is used as row index.

    :param path: Path to the CSV file with run details.
    :type path: str, path object or file-like object
    :returns: DataFrame with run details.

    """
    return read_csv(path, index_col="run")


def main():
    pass
