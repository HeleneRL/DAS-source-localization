"""
Collection of functions for estimating sensitivity.
"""

import os

import numpy as np
from pandas import read_csv
from scipy.fft import rfft, next_fast_len

from .constants import get_run, get_trial_day_metadata
from .detection import matched_filter_detector
from .saveandload import load_interrogator_data, load_user_detection_signal, \
    load_janus_autocorrelation
from .utils import moving_average, hann_window

_OWN_DIR = os.path.realpath(os.path.dirname(__file__))


class Transducer(object):

    """Class representing a transducer with a known sensitivity.

    Each transducer instance uses a CSV file with columns *f* and *tx*,
    where *f* is the frequency in kilohertz and *tx* is the transducer
    sensitivity in dB re 1 µPa/V.
    """

    def __init__(self, sensitivityFile):
        """Construct a transducer object.

        :param sensitivityFile: Path to the sensitivity file.
        :type sensitivityFile: path-like

        """
        self._sensitivityFile = read_csv(sensitivityFile)

    def getSensitivity(self, frequency):
        """Transducer sensitivity in dB re uPa/V.

        :param frequency: Frequency in hertz.
        :type frequency: float or sequence of floats
        :returns: Transducer sensitivity, interpolated from the data.

        """
        return np.interp(frequency, self._sensitivityFile['f']*1000,
                         self._sensitivityFile['tx'])


defaultTransducer = Transducer(_OWN_DIR
                               + "/../resources/transducer-datasheet.csv")
"""Default transducer object.
This make was used in the 2024-05-03 experiments."""


def correlation_spectrum(data_fname, run_data, meta, band, search_space=5000):
    """Estimate cross-spectral density.

    The output has units [relative strain]·[units of reference signal]/Hz.
    Consider that

    .. math ::

        \\frac{1}{f_s} \\sum\\limits_{k=-\\infty}^{\\infty} x[k] y[k]
        \\approx
        \\int\\limits_{-\\infty}^{\\infty} x(t)y(t) \\mathrm{d}t.

    To go from energy times sample rate to power-like, we can divide the
    cross-correlation output by the length of the JANUS packet in samples
    because it is :math:`T` seconds long. Hence it has :math:`T f_s` samples
    under sample rate :math:`f_s`.

    It is the user's responsibility to scale the output to obtain the right
    units. For instance, he or she must multiply the output by :math:`10^9`
    to obtain [nanostrain]·[units of reference signal]/Hz if the strain data
    are in [relative strain].

    .. note ::

        This is a power-type quantity by itself. When used to estimate
        sensitivity, however, it should be treated as a product of two
        voltage-type quantities.

    :param data_fname: Path to the file to load filtered data.
    :type data_fname: path-like
    :param run_data: Information about the associated run.
    :type run_data: output from :py:func:`dasprocessor.constants.get_run`
    :param meta: Run-day metadata.
    :type meta: output from
        :py:func:`dasprocessor.constants.get_trial_day_metadata`
    :param band: JANUS band or path to a transmitted signal.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'} or path-like
    :param search_space: Amount of samples to look behind the presumed
        detection peak when calculating the correlation-spectrum estimate.
        The actual search space is roughly three times this number;
        the exact length is chosen to provide an efficient real-signal FFT.
    :type search_space: int, optional
    :returns: FFTs of the correlation spectra.

    """
    filtered_data = load_interrogator_data(data_fname, None, None,
                                           direct_mode=True)['y']
    nfft = next_fast_len(search_space*3, real=True)
    output_spectra = np.zeros((filtered_data.shape[1],
                               run_data["sequence_count"],
                               nfft//2+1),
                              dtype='complex64')
    base_offset = run_data["offset_in_samples"]\
        + meta["signal_starting_points"][meta["signal_sequence"].index(band)]
    print(base_offset)
    base_period = run_data["sequence_period"]
    sr = 'sample_rate'  # necessary to comply with PEP 8 on line length
    for it in range(run_data["sequence_count"]):
        current_focus = base_offset + it*base_period
        # less important TODO write JANUS packet loading function
        try:
            rate, signal = load_user_detection_signal(_OWN_DIR
                                                      + "/../resources/signals"
                                                      f"/janus-{band}-"
                                                      f"x{it:02x}"
                                                      f"-{run_data[sr]}.wav")
        except FileNotFoundError:
            # path to a user-defined detection signal was given
            rate, signal = load_user_detection_signal(band)
        # this is x(-t) * y(t) without scaling x nor y
        # divided by signal length, it becomes a (biased) correlation estimate
        xc = matched_filter_detector(filtered_data[current_focus-rate:
                                                   current_focus+len(signal)
                                                   + rate],
                                     # normalise to unit RMS naively
                                     # (assumes sinusoid-like signal)
                                     np.sqrt(2)*signal/np.max(np.abs(signal)),
                                     get_output_instead=True)
        output_spectra[:, it] = rfft(xc[rate-search_space:rate-search_space
                                        + nfft]
                                     * np.hanning(nfft)[..., None], axis=0).T\
            / len(signal) / run_data[sr]
    # From Wikipedia (if only I knew a better source),
    # this is the right scaling to get PSD
    # The scaling factor is namely :math:`\\frac{(\\Delta t)^2}{T}`.
    # Equivalently, it is :math:`\\frac{1}{f_s^2T}`.

    return output_spectra


def transmission_loss(source, receivers=None, exponent=2):
    """Transmission loss due to geometrical spreading.

    :param source: Source positions in Cartesian coordinates, or
        distances from source (axis 0) to receiver (axis 1).
        If given as an array of distances, ``receivers`` must be ``None``.
    :type source: 2-D sequence of floats or 1-D sequence of (3,)-floats
    :param receivers: Receiver positions in Cartesian coordinates.
        If not given, assumes source to be a 2-D array of distances.
    :type receivers: None or 1-D sequence of (3,)-floats
    :param exponent: Spreading exponent. Should be between 1 and 2, inclusive:
        1 for cylindrical spreading, 2 for spherical spreading.
    :type exponent: float, optional
    :returns: 2-D array of estimates of transmission loss.

    """
    if receivers is not None:
        source = np.sqrt(np.sum((source[..., None] - receivers.T)**2, axis=-2))

    return 10*exponent*np.log10(source)


def get_sensitivity(cross_psd, distances, transducer=defaultTransducer,
                    input_scaling=100, band="B_4", rate=25000,
                    force_overwrite=False, verbosity=0):
    """Estimate DAS receiver sensitivity.

    Getting the dB conversions right was most likely the most difficult task.

    The full chain is

    .. math ::

        U(f) \\to G_{amp}(f) \\to G_{trans}(f) \\to \\mathit{TL}
        \\underset{X(f)}{\\to} H(f) \\to Y(f)

    for which we know the input :math:`U(f)` and have an idea of how
    :math:`G_{amp}(f)` affects :math:`U`.
    We have a model for :math:`G_{trans}(f)` and can estimate
    :math:`\\mathit{TL}`, so :math:`X(f)` is
    obtainable. :math:`Y(f)` is observed, so we find :math:`H(f)=Y(f)/X(f)`.
    Or, as long as :math:`U^\\ast(f) \\neq 0`,

    .. math ::

        H(f)=\\frac{Y(f)U^\\ast(f)}{X(f)U^\\ast(f)}=\\frac{Y(f)U^\\ast(f)}
        {G_{amp}(f) G_{trans}(f) \\mathit{TL}\\, U(f)U^\\ast(f)}.

    :param cross_psd: Cross-spectral density from
        :py:func:`correlation_spectrum`, scaled to correct units.
        Shape (nchan, npack, nfft//2+1). Equivalent to
        :math:`Y(f)U^\\ast(f)`. To get :math:`Y(f)X^\\ast(f)`,
        apply the other three blocks.
    :type cross_psd: array_like
    :param distances: Distances from transmitter to receiving channels
        per packet. Shape (nchan, npack). Ensure that the slice matches the
        channel range that was used to find the CSD.
    :type distances: array_like
    :param transducer: Object representing the transducer used.
    :type transducer: :py:class:`Transducer`, optional
    :param input_scaling: RMS voltage of the input signal.
    :type input_scaling: float, optional
    :param band: Which JANUS band was used, or a path to a detection signal.
    :type band: {'B_4','B_2','B','C','A'} or path-like, optional
    :param rate: Sampling rate of the original DAS data.
    :type rate: int, optional
    :param force_overwrite: Whether existing correlation spectra should be
        forcibly overwritten. Useful if this function is changed such that
        it would produce different spectra.
    :type force_overwrite: bool, optional
    :param verbosity: How much is reported to calls to :py:func:`print`.
        Higher means more.
    :type verbosity: int, optional
    :returns: Sensitivity estimates. Same shape as ``cross_psd``.

    """
    nfft = (cross_psd.shape[-1] - 1) * 2
    target_f = np.linspace(0, rate/2, cross_psd.shape[-1])
    out = np.zeros(cross_psd.shape)
    for it in range(cross_psd.shape[1]):
        cache_path = _OWN_DIR\
                     + f"/../resources/signal-psds/janus-{band}-x{it:02x}-"\
                     f"{rate}.npz"
        # u(t)
        try:
            signal_psd_mov_avg = load_janus_autocorrelation(band, it, rate,
                                                            force_overwrite)
        except FileNotFoundError:
            cache_miss = True
            try:
                _, signal = load_user_detection_signal(_OWN_DIR
                                                       + "/../resources/"
                                                       f"signals/janus-{band}-"
                                                       f"x{it:02x}-{rate}.wav")
            except FileNotFoundError:
                cache_path = f"{band[:band.rindex('.')]}-spectrum"\
                             f"{band[band.rindex['.']:]}"
                try:
                    _, signal = load_user_detection_signal(cache_path)
                    cache_miss = False
                except FileNotFoundError:
                    _, signal = load_user_detection_signal(band)

            if cache_miss:
                signal = signal/(np.max(np.abs(signal))/np.sqrt(2))
                # above is u[k], next we want it as U(f)
                signal_psd = np.abs(rfft(signal))**2/(len(signal)*rate)\
                    * input_scaling
                # above is G_{amp}(f) U*(f)U(f)
                # moving_average uses $20 \log_{10}$
                # should be cached!
                signal_psd_mov_avg = moving_average(target_f,
                                                    np.linspace(0,
                                                                rate/2,
                                                                len(signal_psd)
                                                                ),
                                                    signal_psd,
                                                    window=hann_window,
                                                    is_fp_dB=False)
                np.savez_compressed(cache_path, y=signal_psd_mov_avg)

        # that was G_{amp}(f) U(f), next is G_{trans}(f) G_{amp}(f) U(f)
        signal_psd_at_td = signal_psd_mov_avg\
            + transducer.getSensitivity(target_f)
        # applying TL gives us X(f)
        signal_psd_at_cable = signal_psd_at_td\
            - transmission_loss(distances[:, it])[..., None]
        # and we can now find H(f) as Y(f) U*(f) / X(f) U*(f) (working in dB)
        out[:, it] = 20*np.log10(np.abs(cross_psd[:, it]))\
            - signal_psd_at_cable
        if verbosity > 0:
            print(f"Packet {it} processed")

    return out


def get_plottable_sensitivity(frequency, sensitivity, angles, frequency_bins,
                              snr_dB, desired_angles=None, minimum_snr_dB=3,
                              angle_hann_width=2, quantities=("mean", "std")):
    """Find sensitivity per angle and frequency bin.

    If results from multiple runs are to be used in the same estimate, they
    must be concatenated. This applies to all arrays with ``angles.shape``
    in their shape.

    The function can also be used to find average and spread of SNR.

    :param frequency: Frequency array.
    :type frequency: n-sequence of floats
    :param sensitivity: Sensitivity estimates.
    :type sensitivity: array_like with shape ``angles.shape + (n,)``
    :param angles: Grazing angles in degrees associated with sensitivity.
    :type angles: array_like
    :param frequency_bins: Edges of the desired frequency bins.
    :type frequency_bins: sequence of floats
    :param snr_dB: Channel-level SNR estimates in dB. Same shape as ``angles``.
    :type snr_dB: array_like
    :param desired_angles: Desired sensitivity angles in degrees. If not given,
        is set to a :py:func:`numpy.linspace` from 30 degrees to 75 degrees
        with step 0.5 degrees.
    :type desired_angles: sequence of floats, optional
    :param minimum_snr_dB: Receptions with less than this SNR in dB
        are discarded when estimating the sensitivity.
    :type minimum_snr_dB: float, optional
    :param angle_hann_width: Width of the angular moving-average window
        in degrees.
    :type angle_hann_width: float, optional
    :param quantities: Which statistical quantities to estimate.
    :type quantities: sequence of {"mean", "std", "var", "median", "quartile"},
        optional
    :returns: Average sensitivity per frequency band and standard deviation,
        smoothed by grazing angles.

    """
    allowed_quantities = {"mean", "std", "var", "median", "quartile"}
    bad_quantities = set(quantities) - allowed_quantities
    if len(bad_quantities):
        raise ValueError(f"quantities {','.join(bad_quantities)} unsupported")
    # now the desired quantities are verified

    if desired_angles is None:
        desired_angles = np.linspace(30, 75, 45*2+1)

    kept_sensitivity = sensitivity[snr_dB >= minimum_snr_dB]
    kept_angles = angles[snr_dB >= minimum_snr_dB]
    output = dict()
    output_base = np.zeros(desired_angles.shape + frequency_bins[1:].shape)
    for k in quantities:
        if k == "quartile":
            output[k] = output_base[..., None].copy() + np.zeros(5)
        else:
            output[k] = output_base.copy()

    for it in range(output_base.shape[-1]):
        # collapse relevant frequencies to a single band for memory reasons
        this_band_mean = np.mean(kept_sensitivity[...,
                                                  (frequency
                                                   >=
                                                   frequency_bins[it])
                                                  &
                                                  (frequency
                                                   <
                                                   frequency_bins[it+1])
                                                  ], axis=-1)
        for k in quantities:
            if k == "mean":
                output[k][:, it] = moving_average(desired_angles, kept_angles,
                                                  this_band_mean,
                                                  tol=angle_hann_width,
                                                  window=hann_window)
            else:
                func = eval(f"np.{k}")
                output[k][:, it] = np.hstack([func(this_band_mean[
                    (kept_angles > desired_angles[it]-angle_hann_width) &
                    (kept_angles < desired_angles[it]+angle_hann_width)
                ], ddof=1) for it in range(len(desired_angles))])

    return output


def main():
    """Demonstrate that a user can estimate sensitivity.
    Also test sensitivity plotting.

    Should not be called by the end user.

    """
    import matplotlib.pyplot as plt
    from .visualisation import plot_sensitivity_curves
    from .constants import frequency_bands, janus_frequency_bands

    band = "B_2"
    fband = frequency_bands[band]
    jband = janus_frequency_bands[band]
    basepath = "resources/base-sens-{}-run{}.npz"
    frange = np.linspace(0, 15000, 151)
    chrange = slice(48, 300)
    excluderange = None  # slice(180, 240)
    run_no = 2
    with_other = True
    min_snr = 3
    loadrange = range(chrange.start, chrange.stop, 12)
    rundict = get_run("2024-05-03", run_no)
    metadata = get_trial_day_metadata("2024-05-03")
    rebuild = True
    stop_after_rebuild = False
    if rebuild:
        test = np.vstack([correlation_spectrum("/home/emil/Dokument/Data/DASOn"
                                               "eWay/DasKabel/backups/filtered"
                                               f"-{fband[0]}-{fband[1]}_"
                                               f"105919-114719_{it}-{it+12}"
                                               ".npz", rundict, metadata, band)
                          for it in loadrange if excluderange is None
                          or excluderange is not None and (it
                                                           < excluderange.start
                          or it >= excluderange.stop)])
        # fixed time interval needs to not be so hardcoded when doing run 2

        distdata = np.load(_OWN_DIR
                           + "/../resources/"
                           f"{rundict['source_position_file']}")
        dftdata = [np.load(_OWN_DIR
                           + f"/../resources/{band}/dfts-{it}-{it+12}-"
                           f"run{run_no}"
                           ".npz")
                   for it in loadrange if excluderange is None or
                   excluderange is not None and (it < excluderange.start
                   or it >= excluderange.stop)]
        if excluderange is not None:
            chrange = np.hstack([np.arange(chrange.start, excluderange.start),
                                 np.arange(excluderange.stop, chrange.stop)])
        isthissensitivity = get_sensitivity(test*1e9,
                                            distdata['dist'][chrange],
                                            verbosity=1, band=band)
        # the 1e9 changes strain to nanostrain
        snr_data = np.vstack([x['snr'] for x in dftdata])
        coses = distdata['cos'][chrange]
        np.savez(basepath.format(band, run_no), sens=isthissensitivity,
                 snr=snr_data,
                 cos=coses)
    else:
        cached_data = np.load(basepath.format(band, run_no))
        isthissensitivity = cached_data['sens']
        snr_data = cached_data['snr']
        coses = cached_data['cos']

    if stop_after_rebuild:
        return
    other_run = 2 if run_no == 1 else 1 if run_no == 2 else -1
    if with_other:
        other_run_data = np.load(basepath.format(band, other_run))

    fbins = np.linspace(jband["fmin"] + jband["df"]*0.5,
                        jband["fmin"] + jband["df"]*25.5,
                        6)
    idxes = slice(None) if excluderange is None\
        else np.hstack([np.arange(excluderange.start - chrange.start),
                        np.arange(excluderange.stop, chrange.stop)
                        - chrange.start])
    # dynamic frequency range, no longer hard coded
    cosdata = np.vstack([coses[idxes], other_run_data['cos']]) if with_other\
        else coses[idxes]
    sensdata = np.vstack([isthissensitivity[idxes],
                          other_run_data['sens']]) if with_other\
        else isthissensitivity[idxes]
    snrstack = np.vstack([snr_data[idxes], other_run_data['snr']])\
        if with_other else snr_data[idxes]
    print(sensdata.shape, cosdata.shape, snrstack.shape)
    desired_angles = np.linspace(0, 90, 181)
    plotdata = get_plottable_sensitivity(
        np.linspace(0, 12500, isthissensitivity.shape[-1]),
        sensdata,
        np.degrees(np.arccos(np.abs(cosdata))),
        fbins,
        snrstack,
        minimum_snr_dB=min_snr,
        desired_angles=desired_angles
    )
    isthisplotfriendly, isthisagoodstd = plotdata["mean"], plotdata["std"]
    snr_dict = get_plottable_sensitivity(
        np.zeros(1),
        snrstack[..., None],
        np.degrees(np.arccos(np.abs(cosdata))),
        np.array([-1, 1]),
        snrstack,
        minimum_snr_dB=-100,
        desired_angles=desired_angles
    )
    snr_mean, snr_std = snr_dict["mean"], snr_dict["std"]
    plotwindow = desired_angles
    # plt.matshow(isthissensitivity[:, 30, :1500])
    # plt.xlabel('Packet number')
    # plt.ylabel('Frequency bin')
    # plt.title('Packet 30')
    # plt.colorbar(location="bottom")
    # plt.show()
    # Gives -140 dB and similar. Actually makes more sense now.

    colour_sequence = ["C3", "C1", "C2", "C0", "C4"]
    outname = f"resources/full-sensitivity-{band}-run{run_no}n{other_run}.npz"
    np.savez(outname, fbins=fbins, theta=plotwindow,
             exp_sensitivity=isthisplotfriendly,
             std_sensitivity=isthisagoodstd,
             exp_snr=snr_mean, std_snr=snr_std)
    fsens, axsens = plot_sensitivity_curves(plotwindow, isthisplotfriendly,
                                            isthisagoodstd,
                                            fbins, colours=colour_sequence,
                                            give_handles=True)
    # fsnr, axsnr = plot_sensitivity_curves(plotwindow, snr_mean, snr_std,
    #                                       [-np.inf, np.inf], colours=["C7"],
    #                                       give_handles=True)
    plt.show()


if __name__ == "__main__":
    main()
