"""Module for detecting received packets and pinpointing their timing."""

import numpy as np
from scipy.signal import correlate, find_peaks, hilbert
from pathlib import Path
import os

from .saveandload import load_builtin_detection_signal, save_peaks

import matplotlib.pyplot as plt




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
    envs = []
    # loop for better memory usage
    for it in range(rx.shape[-1]):
        xc = correlate(rx[:, it], preamble, 'valid')
        if get_output_instead:
            out.append(xc)
        else:
            #xc /= np.max(np.abs(xc))
            env  = np.abs(hilbert(xc))          # amplitude envelope
            env /= env.max() or 1.0
            envs.append(env)
            out.append(find_peaks(env, **peak_properties))
            #out.append(find_peaks(np.sqrt(xc**2), **peak_properties))
    
    return np.column_stack(out) if get_output_instead else tuple(out), envs

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

    #big Helene tweak, changed the loop structure, see original file for reference
    for it in range(len(peaks)): 
         # For each packet index c, find the earliest peak row r within tol
        peak_idx = peaks[it][0]  # 1-D array of peak sample indices (slice timebase)
        peak_map = {}
        for c, tgt in enumerate(target):
            cand = np.where(np.abs(peak_idx - tgt) <= tol)[0]
            if cand.size:
                r = int(cand.min())  # earliest in time
                peak_map[r] = c
        pmap[start+it] = {"peak_map": peak_map}
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
    detector_hits, envs = matched_filter_detector(rx, preamble, peak_properties)
    packets_found = map_peaks_to_packets(detector_hits, start, run_data, tol)

    # TODO get dict on form {channel: {pk1: idx1, pk2: idx2, ..., pkN: idxN}}
    out = {}
    for it in range(start, start+len(detector_hits)): #Helene tweak, see original file for reference
        out[it] = {
            int(packets_found[it]["peak_map"][x]):
            int(detector_hits[it-start][0][x])
            for x in packets_found[it]["peak_map"].keys()
        }

    return out


def main():
    print("=== DAS packet detection demo ===")
    """Demonstrate that the module can detect JANUS packets.

    Testing purposes only.
    """
    from sys import argv
    from copy import deepcopy

    from .saveandload import load_interrogator_data, save_peaks, load_peaks
    from .constants import get_run, frequency_bands, get_trial_day_metadata
    from dasprocessor.debugging import plot_channel_corr_and_peaks, plot_channel_corr_with_selected
    choicerun = int(argv[1]) if len(argv) > 1 else 2
    band = argv[2] if len(argv) > 2 else "B_4"
    myrun = deepcopy(get_run("2024-05-03", choicerun))


    #print("Using time_range:", myrun["time_range"])
    meta = get_trial_day_metadata("2024-05-03")
    myrun["offset_in_samples"] += meta["signal_starting_points"][
        meta["signal_sequence"].index(band)]
    #for it in range(300, 312, 12):
    for it in [148]: 
        wantedchans = slice(it, it+12)
        mydata = load_interrogator_data(
                r"D:\DASComms_25kHz_GL_2m\20240503\dphi",
                *myrun["time_range"],
                on_fnf="cache",
                channels=wantedchans,
                filter_band=frequency_bands[band],
                cachepath=r"D:\backups",
                out="npz",
                verbose=True)
        


        # --- plotting inputs (use same params you pass to detector) ---  # <<< ADDED
        preamble = load_builtin_detection_signal(f"preamble-{band}", 25000)   # <<< ADDED
        # peak_properties = {                                                  # <<< ADDED
        #     "prominence": 0.3,
        #     "height": 0.15,
        #     "distance": 500000
        # }
        # Targets in RAW indices (run/global timebase):                    # <<< ADDED
        targets_raw = (
            myrun["offset_in_samples"]
            + np.arange(myrun["sequence_count"]) * myrun["sequence_period"]
        )

        # 1) Full-trace overview for the FIRST channel in the slice        # <<< ADDED
        ch_local = 0  # 0..(wantedchans.stop - wantedchans.start - 1)
        rx_col = mydata['y'][:, ch_local]

        # plot_channel_corr_and_peaks(
        #     rx_col,
        #     preamble,
        #     peak_properties=peak_properties,
        #     targets=targets_raw,
        #     tol=5000,                 # just to visualize your current tol band
        #     fs=int(mydata['fs']),
        #     zoom_center=None,         # full correlation trace
        #     title_prefix=f"ch {wantedchans.start + ch_local}"
        # )

        # # --- end plotting ---          

        print(f"Detecting packets in channels {wantedchans.start} to {wantedchans.stop-1}...")

        mypeaks = get_packet_sample_indexes(mydata['y'],
                                            load_builtin_detection_signal(
                                                    f"preamble-{band}", 25000),
                                            wantedchans.start,
                                            {
                                                "prominence": 0.025, #I added prominence
                                                "height": 0.05, 
                                                "distance": 500000                                    
                                        
                                            },
                                            myrun,
                                            7500) #tolerance in samples, 500m/1475m/s*25000sps ≈ 8475 samples
        print(f"Detected peaks map for channels {wantedchans.start} to {wantedchans.stop-1}:")
        
        savepath = Path(__file__).resolve().parent / f"../resources/{band}/peaks-{wantedchans.start}-{wantedchans.stop}-run{choicerun}-HeleneTweaks.json"
        savepath = savepath.resolve()
        os.makedirs(savepath.parent, exist_ok=True)
        save_peaks(savepath, mypeaks)
        print(f"✅ Saved: {savepath}")

        # choose a channel inside the slice to inspect (e.g., first in the slice)
        ch_global = wantedchans.start          # e.g., 148
        ch_local  = ch_global - wantedchans.start  # 0-based within the loaded block
        selected_map = mypeaks.get(ch_global, {})  # {packet_id: corr_index}



        # plot
        plot_channel_corr_with_selected(
            rx_col=mydata['y'][:, ch_local],
            preamble=preamble,
            selected_peaks_map=selected_map,
            targets=targets_raw,        # already in corr coords
            tol=5000,                    # same tol you used in mapping
            fs=25000,
            targets_are_raw=False,       # we passed corr targets above
            zoom_center=None,            # full trace
            title_prefix=f"ch {ch_global}",
            annotate=True
        )

        plot_channel_corr_with_selected(
            rx_col=mydata['y'][:, ch_local+1],
            preamble=preamble,
            selected_peaks_map=mypeaks.get(ch_global+1, {}),
            targets=targets_raw,        # already in corr coords
            tol=5000,                    # same tol you used in mapping
            fs=25000,
            targets_are_raw=False,       # we passed corr targets above
            zoom_center=None,            # full trace
            title_prefix=f"ch {ch_global+1}",
            annotate=True
        )


        plot_channel_corr_with_selected(
            rx_col=mydata['y'][:, ch_local+2],
            preamble=preamble,
            selected_peaks_map=mypeaks.get(ch_global+2, {}),
            targets=targets_raw,        # already in corr coords
            tol=5000,                    # same tol you used in mapping
            fs=25000,
            targets_are_raw=False,       # we passed corr targets above
            zoom_center=None,            # full trace
            title_prefix=f"ch {ch_global+2}",
            annotate=True
        )

        
        plot_channel_corr_with_selected(
            rx_col=mydata['y'][:, ch_local+8],
            preamble=preamble,
            selected_peaks_map=mypeaks.get(ch_global+8, {}),
            targets=targets_raw,        # already in corr coords
            tol=5000,                    # same tol you used in mapping
            fs=25000,
            targets_are_raw=False,       # we passed corr targets above
            zoom_center=None,            # full trace
            title_prefix=f"ch {ch_global+8}",
            annotate=True
        )






if __name__ == "__main__":
    main()






