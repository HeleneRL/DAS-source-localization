"""Module for detecting received packets and pinpointing their timing."""

import numpy as np
from scipy.signal import correlate, find_peaks

from .saveandload import load_builtin_detection_signal, save_peaks


def matched_filter_detector(rx, preamble, peak_properties={},
                            get_output_instead=False):
    """Apply a matched filter detector to DAS data.

    Correlates the received signal with a reference signal. A relatively high
    correlation peak indicates a possible start of a packet, so
    :py:func:`scipy.signal.find_peaks` is used to locate them. The correlation
    output is normalised before searching for peaks.

    :param rx: Received signal.
    :type rx: shape-(N,[k]) array
    :param preamble: Reference signal.
    :type preamble: 1-D sequence of floats
    :param peak_properties: Dictionary of peak properties. This is passed to
        :py:func:`scipy.signal.find_peaks`.
    :type peak_properties: dict
    :param get_output_instead: If ``True``, outputs the cross-correlation
        function. Skips peak detection and hence ``peak_properties``.
    :type get_output_instead: bool
    :returns: A k-tuple of outputs from ``scipy.signal.find_peaks``.

    """
    if rx.ndim == 1:
        rx = rx[:, None]
    elif rx.ndim != 2:
        raise TypeError('expected received signal to be a 1-D or 2-D array,'
                        f' but it is {rx.ndim}-D')

    out = []
    # loop for better memory usage
    for it in range(rx.shape[-1]):
        xc = correlate(rx[:, it], preamble, 'valid')
        if get_output_instead:
            out.append(xc)
        else:
            xc /= np.max(np.abs(xc))
            out.append(find_peaks(xc, **peak_properties))

    return np.column_stack(out) if get_output_instead else tuple(out)


def map_peaks_to_packets(peaks, start, run_data=None, tol=None):
    """Determine which peaks belong to which packet.

    :param peaks: Detected peaks.
    :type peaks: sequence of :py:func:`scipy.signal.find_peaks` results
    :param start: Index of first channel.
    :type start: int
    :param run_data: Run info. If omitted, will be estimated from the peak
        locations.
    :type run_data: :py:func:`dasprocessor.constants.get_run` output
    :param tol: Maximum allowed deviation, up or down, from expected peak
        locations.

        .. caution ::

            If omitted, uses three times the estimated standard deviation of
            the first detected peak in samples. Rely on it with caution.

    :type tol: int, optional
    :returns: A map of peak indices to possible packets. Only peaks that
        could represent a packet are included.
    """
    # estimate important parameters
    if run_data is None:
        med_first = np.median([x[0][0] for x in peaks])
        med_diff = np.median(np.hstack([np.diff(x[0]) for x in peaks]))
        max_packets = np.round(np.median([np.diff(x[0][[0, -1]])
                                          for x in peaks]))+1
    # use known numbers
    else:
        med_first = run_data["offset_in_samples"]
        med_diff = run_data["sequence_period"]
        max_packets = run_data["sequence_count"]

    tol = 3*np.std([x[0][0] for x in peaks]) if tol is None else tol
    pmap = {x: None for x in range(start, start+len(peaks))}
    target = np.arange(max_packets)*med_diff + med_first
    for it in range(len(peaks)):
        mapmat = np.isclose(peaks[it][0][:, None], target, rtol=0, atol=tol)
        rows, cols = np.nonzero(mapmat)
        pmap[start+it] = {"peak_map": {r: c for r, c in zip(rows, cols)}}

    return pmap


def get_packet_sample_indexes(rx, preamble, start, peak_properties={},
                              run_data=None, tol=None):
    """Find the positions of received packets in a received signal.

    :param rx: Received signal. Passed to :py:func:`matched_filter_detector`.
    :type rx: shape-(N,[k]) array
    :param preamble: Detection preamble.
        Passed to :py:func:`matched_filter_detector`.
    :type preamble: 1-D sequence of floats
    :param start: DAS channel number of first channel.
        Passed to :py:func:`map_peaks_to_packets`.
    :type start: int
    :param peak_properties: Keyword arguments to
        :py:func:`scipy.signal.find_peaks` in
        :py:func:`matched_filter_detector`.
    :type peak_properties: dict
    :param run_data: Run info. Passed to :py:func:`map_peaks_to_packets`.
    :type run_data: :py:func:`dasprocessor.constants.get_run` output
    :param tol: How far away a peak can be from an expected location. Passed to
        :py:func:`map_peaks_to_packets`.
    :type tol: int
    :returns: A map of packet indexes to sample numbers.

    .. seealso ::

        * :py:func:`matched_filter_detector`
        * :py:func:`map_peaks_to_packets`

    """
    detector_hits = matched_filter_detector(rx, preamble, peak_properties)
    packets_found = map_peaks_to_packets(detector_hits, start, run_data, tol)
    # TODO get dict on form {channel: {pk1: idx1, pk2: idx2, ..., pkN: idxN}}
    out = {}
    for it in range(start, start+len(detector_hits)):
        out[it] = {int(packets_found[it]["peak_map"][x]):
                   detector_hits[it-start][0][x]
                   for x in packets_found[it]["peak_map"].keys()}

    return out


def main():
    """Demonstrate that the module can detect JANUS packets.

    Testing purposes only.
    """
    from sys import argv
    from copy import deepcopy

    from .saveandload import load_interrogator_data, save_peaks, load_peaks
    from .constants import get_run, frequency_bands, get_trial_day_metadata
    choicerun = int(argv[1]) if len(argv) > 1 else 1
    band = argv[2] if len(argv) > 2 else "B_4"
    myrun = deepcopy(get_run("2024-05-03", choicerun))
    meta = get_trial_day_metadata("2024-05-03")
    myrun["offset_in_samples"] += meta["signal_starting_points"][
        meta["signal_sequence"].index(band)]
    for it in range(300, 420, 12):
        wantedchans = slice(it, it+12)
        mydata = load_interrogator_data(
                "/home/emil/Dokument/Data/DASOneWay/DasKabel/rawdata/DASComms_"
                "25kHz_GL_2m/20240503/dphi",
                *myrun["time_range"],
                on_fnf="cache",
                channels=wantedchans,
                filter_band=frequency_bands[band],
                cachepath="/home/emil/Dokument/Data/DASOneWay/DasKabel/backups",
                out="npz",
                verbose=True)
        # distance is 20 seconds times 25000 samples per second
        mypeaks = get_packet_sample_indexes(mydata['y'],
                                            load_builtin_detection_signal(
                                                    f"preamble-{band}", 25000),
                                            wantedchans.start,
                                            {
                                                "distance": 500000,
                                                "height": 0.01
                                            },
                                            myrun,
                                            2500)
        savepath = f"/../resources/{band}/peaks-{wantedchans.start}-{wantedchans.stop}-"\
                   f"run{choicerun}.json"
        save_peaks(savepath, mypeaks)
        print(it)


if __name__ == "__main__":
    main()
