"""
General-purpose utility functions for other modules to use.
"""

from datetime import datetime, timezone, timedelta

import numpy as np
from pandas import DataFrame
from scipy.signal import butter
from scipy.sparse import dok_array
from sk_dsp_comm.fec_conv import FECConv

from .constants import janus_frequency_bands, baseline_chips


def to_time_list(myrange, maybestop=None, *, step: int = 10):
    """Build time list for loading many blocks of interrogator data.

    Use them to load raw interrogator data.

    :param myrange: Start of time sequence, or both start and stop of time
        sequence. Each time sequence is given on the form
        (hours, minutes, seconds).
    :type myrange: 3-tuple of ints or 2-tuple of 3-tuple of ints
    :param maybestop: Stop of time sequence.
        If given, ``myrange`` must be start of time sequence.
    :type maybestop: 3-tuple of ints, optional
    :param step: Time between interrogator data blocks in seconds.
    :type step: int

    :returns: A list of "hhmmss" strings from the start time, inclusive, to the
        stop time, exclusive.
    """
    _res = np.array([3600, 60, 1])
    stop = np.asarray(maybestop or myrange[1])@_res
    start = np.asarray(myrange[0])@_res if maybestop is None else\
        np.asarray(myrange)@_res
    return [f"{it//3600:02d}{(it % 3600)//60:02d}{it % 60:02d}"
            for it in range(start, stop, step)]


def dt2num(dt, time_only=True):
    """Convert date-time to number.

    :param dt: Date-time represented by an ISO-formatted string.
    :type dt: datetime-like string
    :returns: A numeric value corresponding to the datetime.

    """
    dt = datetime.fromisoformat(dt + "+00:00")
    h = dt.hour * 3600
    m = dt.minute * 60
    s = dt.second
    frac = dt.microsecond / 1000000
    if time_only:
        return h+m+s+frac  # below is a reference to https://xkcd.com/2867/
    return NotImplemented  # IT IS IMPOSSIBLE TO KNOW AND A SIN TO ASK!


def strip_trailing(string, char='/'):
    """Strip all trailing instances of a specified character.

    :param string: Input string.
    :type string: str
    :param char: Character to strip from the end of the string.
    :type char: length-1 str
    :returns: The input string with trailing ``char`` stripped.
    """
    while string[-1] == char:
        string = string[:-1]

    return string


def get_bandpass_filter(lo, hi, rate, order=6):
    """Retrieve Butterworth filter. Wrapper for :py:func:`scipy.signal.butter`.

    :param lo: Lower limit of passband.
    :type lo: float
    :param hi: Upper limit of passband.
    :type hi: float
    :param rate: Sampling rate of intended application data.
    :type rate: int
    :param order: Butterworth filter order.
    :type order: int, optional
    :returns: A discrete-time Butterworth filter with specified cut-off
        frequencies, sampling rate and order.

    .. seealso ::

        * :py:func:`scipy.signal.butter`

    """
    return butter(order, [lo, hi], 'bandpass', output='sos', fs=rate)


def get_janus_duration_samples(band, rate=25000):
    """Find length of a JANUS baseline packet in samples.

    :param band: JANUS frequency band.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'}
    :param rate: Sampling rate.
    :type rate: int
    :returns: Duration of a JANUS baseline packet in the specified frequency
        band and under the specified sampling rate.

    """
    return round(rate*baseline_chips/janus_frequency_bands[band]['df'])


def get_decoder():
    """Obtain JANUS decoder.

    :returns: Viterbi decoder for decoding received JANUS packets.
    """
    return FECConv(('111101011', '101110001'), 9)


def bit_array_to_int(x, flatten=False):
    """Convert array of bit sequences to 64-bit signed integers.

    Operates on the last axis.

    :param x: Input bit sequences. Must not be longer than 64 along the last
        axis.
    :type x: array_like
    :param flatten: Whether the output should be flattened.
    :type flatten: bool, optional
    :returns: Integer representations of input bit sequences.
    :raises OverflowError: if the last axis of ``x`` is longer than 64, which
        means it cannot be represented as a primitive integer in C.

    """
    if x.shape[-1] > 64:
        raise OverflowError(f'bit sequence of length {x.shape[-1]} cannot be'
                            f' accurately represented as a C int64_t')
    out = (x @ np.flip(2**np.arange(x.shape[-1]))).astype('int64')
    return out.flatten() if flatten else out


def to_pandas(packets, crcs, start=0, combined=False):
    """Save decoded JANUS packets to a Pandas DataFrame.

    :param packets: Decoded bits as numeric zeroes and ones.
    :type packets: array_like with shape (..., 64)
    :param crcs: Results from running a
        :py:func:`CRC <dasprocessor.janusprocessing.crc8>` on
        the first 56 bits of the packets.
    :type crcs: array_like
    :param start: First channel in the decoded packets.
    :type start: int, optional
    :param combined: If the packet array is the result of combining channels.
    :type combined: bool, optional
    :returns: A DataFrame with row indexes on the form
        "C[channel number]P[packet index]" and the following columns:

        version
            JANUS version

        mobility
            Mobility flag

        schedule
            Scheduling flag

        txRxFlag
            TX/RX flag

        canForward
            Forwarding capability

        classUserID
            JANUS class user ID (CUID)

        appType
            Application type (CUID specific)

        appData
            Application data block (CUID/appType specific)

        crc
            CRC8 field in decoded packet

        valid
            Whether the CRC field checks out. Does not warrant that the packet
            was correctly decoded.

    """
    decoded_crc = bit_array_to_int(packets[..., -8:], True)
    index_range = [f"C{x//packets.shape[-2] + start}P{x % packets.shape[-2]}"
                   for x in range(packets[..., 0].size)]
    return DataFrame({"version": bit_array_to_int(packets[..., :4], True),
                      "mobility": packets[..., 4].flatten(),
                      "schedule": packets[..., 5].flatten(),
                      "txRxFlag": packets[..., 6].flatten(),
                      "canForward": packets[..., 7].flatten(),
                      "classUserID": bit_array_to_int(packets[..., 8:16],
                                                      True),
                      "appType": bit_array_to_int(packets[..., 16:22], True),
                      "appData": bit_array_to_int(packets[..., 22:56], True),
                      "crc8": decoded_crc,
                      "valid": (crcs.flatten() == decoded_crc)
                      & np.any(packets != 0, axis=-1).flatten()},
                     index=index_range,
                     )


def moving_average(x, xp, fp, tol=5, window=None, output_dB=True,
                   is_fp_dB=True):
    """Moving average of a collection of data points.

    Call signature deliberately resembles that of :py:func:`numpy.interp`.

    :param x: Desired output points.
    :type x: array_like
    :param xp: Location of input data points.
    :type xp: array_like
    :param fp: Input data at these locations.
    :type fp: array_like
    :param tol: Width of the averaging window in data-point-location units.
    :type tol: positive float, optional
    :param window: Window function.
        If not specified, defaults to a rectangular window.
    :type window: function with signature
        ``(t: np.typing.ArrayLike, span: float) -> np.ndarray``, optional
    :param output_dB: Whether to output averaged data in dB.
    :type output_dB: bool, optional
    :param is_fp_dB: Whether ``fp`` are given in dB.
    :type is_fp_dB: bool, optional
    :returns: Moving-average values at the desired output points.

    """
    def cast_to_dB(y): return 20*np.log10(y) if output_dB else y
    def cast_from_dB(y): return 10**(y/20) if is_fp_dB else y
    if window is None:
        def window(t, span): return np.where(np.abs(t) < span, 1., 0.)

    # This approach is not as heavy on memory any more
    # Constructing a matrix of all weights was not a bright idea, memory-wise
    f = np.zeros(len(x))
    for it in range(len(x)):
        # could be that sparse arrays will help with efficiency
        weights = dok_array(window(x[it]-xp, tol))
        weights /= weights.sum()
        weights[~np.isfinite(weights.toarray())] = 0  # clear inf and NaN
        f[it] = cast_to_dB((np.nan_to_num(cast_from_dB(fp))*weights).sum())

    return f


def hann_window(t, span):
    """Hann window from :math:`-\\mathrm{span}` to :math:`+\\mathrm{span}`.

    :param t: Evaluation points.
    :type t: array_like
    :param span: Width of the window.
    :type span: float
    :returns: The Hann window with desired span evaluated at the given points.

    """
    return np.where(np.abs(t) < span, 0.5+0.5*np.cos(np.pi/span*t), 0)


def ensure_numpy(arr):
    """Ensures that the input is a NumPy array.

    :param arr: Input, not necessarily a NumPy array.
    :type arr: array-like
    :returns: The input as a NumPy array.

    """
    if isinstance(arr, DataFrame):
        return arr.to_numpy()
    return np.asarray(arr)
