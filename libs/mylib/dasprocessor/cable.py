"""Theoretical models of a fibre-optic cable."""
import numpy as np
from scipy.fft import rfftfreq
from scipy.signal import convolve, ShortTimeFFT

from dasprocessor.constants import get_run
from dasprocessor.janusprocessing import get_all_packets
from dasprocessor.saveandload import load_interrogator_data
from dasprocessor.utils import ensure_numpy
import dasprocessor.spatial_sampling_response as ssr


def mechanical_strain(grazing_angle, spl_dB_upa,
                      bulk_modulus=2.25e9,
                      fibre_modulus=7e10,
                      young_modulus_ratio=0.1,
                      poisson_ratio=0.2,
                      angle_units="degrees"):
    """Stress-to-strain conversion on a fibre-optic cable.
    :math:`\\Delta L_0/L_0` from Eq. (20) in `Taweesintananon21`_.

    .. _Taweesintananon21: https://library.seg.org/doi/10.1190/geo2020-0834.1

    :param grazing_angle: Grazing angle of the incident wave.
    :type grazing_angle: array-like
    :param spl_dB_upa: Sound pressure level of the incident wave
        in dB re 1 µPa.
    :type spl_dB_upa: float
    :param bulk_modulus: Young's modulus of water in Pascals.
    :type bulk_modulus: float, optional
    :param fibre_modulus: Young's modulus of the fibre in Pascals.
    :type fibre_modulus: float, optional
    :param young_modulus_ratio: Ratio of Young's moduli of the cable and the
        fibre. Young's modulus of the cable equals this number times
        ``fibre_modulus``.
    :type young_modulus_ratio: [0., 1.], optional
    :param poisson_ratio: Poisson's ratio for the fibre. Causes strain in
        other directions when strain is applied in one direction.
    :type poisson_ratio: [-1., 0.5], optional
    :param angle_units: Which angular unit ``grazing_angle`` has.
    :type angle_units: {'degrees', 'radians'}
    :returns: Modelled RMS strain expressed in dB re :math:`10^{-9}`.

    """
    grazing_angle = validate_angles(grazing_angle, angle_units)

    if young_modulus_ratio < 0 or young_modulus_ratio > 1:
        raise ValueError("coupling coefficient must be between 0 and 1"
                         " inclusive")
    if poisson_ratio < -1 or poisson_ratio > 0.5:
        raise ValueError("Poisson's ratio must be between -1 and 0.5"
                         " inclusive")

    cable_modulus = fibre_modulus*young_modulus_ratio
    # cable takes the stress, fibre takes 1/alpha as much stress (more)

    # strain = stress (pressure) / Young's modulus
    # equations referenced from Kuvshinov, 2016
    base_strain = (
                   spl_dB_upa
                   - 120 + 180  # go from uPa/strain to Pa/nanostrain
                   - 20*np.log10(bulk_modulus)  # stress = modulus × strain
                  )
    longitudinal = (
                    base_strain
                    + 40*np.log10(np.abs(np.cos(grazing_angle)))  # it is cos^2
                    + 20*np.log10(0.7 + 2*0.2*poisson_ratio)
                    # 0.7 from (6), other term from (8)
                   )
    transversal = (
                   base_strain
                   + 40*np.log10(np.abs(np.sin(grazing_angle)))  # it is sin^2
                   + 20*np.log10(0.2)  # factor 0.2 from Eq. (6)
                   + 20*np.log10(young_modulus_ratio)  # higher modulus in den
                  )
    return 20*np.log10(0j + 10**(longitudinal/20)
                       - 10**(transversal/20))  # 0j for complex-valued out


def directivity(grazing_angle, source_frequency,
                angle_units="degrees", c=1500, points_per_gl=16,
                gl_mult=2, freq_count=64, skip_pulse=False, skip_space=False):
    """Frequency-dependent directivity effects of an interrogator.

    *I am on the right track, but I need to revisit the moments I have.*

    :param grazing_angle: Grazing angle of the incident wave.
    :type grazing_angle: array-like
    :param source_frequency: Temporal frequency of the incident wave.
    :type source_frequency: array-like
    :param angle_units: Units of the grazing angle.
    :type angle_units: {'degrees', 'radians'}, optional
    :param c: Speed of sound in water.
    :type c: int or float, optional
    :param points_per_gl: Number of spatial-response points per channel.
    :type points_per_gl: int, optional
    :param gl_mult: Gauge-length multiplier. The channel spacing in metres is
        found as this number times 1.02. The width of the window function is
        twice of this.
    :type gl_mult: (int or float) >= 1, optional
    :param freq_count: Number of frequency (wavenumber) elements to use.
    :type freq_count: int, optional
    :param skip_pulse: Whether to omit the autocorrelation effects of the
        pulse from the interrogator.
    :type skip_pulse: bool, optional
    :param skip_space: Whether to omit the effects of the spatial windowing
        taking effect on the interrogator.
    :type skip_space: bool, optional
    :returns: Signal gain due to directivity.

    """
    if skip_pulse and skip_space:
        raise ValueError('cannot skip both spatial windowing effects')

    grazing_angle = validate_angles(grazing_angle, angle_units)
    # we go from Hz [1/s] to 1/m
    axial_wavenumber = np.atleast_1d(source_frequency)[..., None]\
        * np.cos(grazing_angle[..., None, None])/c
    # if we skip neither property, extend either domain to avoid tapering
    gl_freq_count = freq_count if skip_pulse or skip_space else 2*freq_count
    interrogator_pos, interrogator_space\
        = ssr.instrument_spatial_response(interpolation=points_per_gl,
                                          nFreq=gl_freq_count,
                                          method='linear')
    # why do I overcomplicate matters here?
    # I could just find the wavenumber and insert it into the wavenumber resp
    gaugelength_pos, gaugelength_space\
        = ssr.gaugelength_spatial_response(gaugeLenghtChs=gl_mult,
                                           interpolation=points_per_gl,
                                           nFreq=freq_count,
                                           method='linear')
    if skip_pulse:
        combined_space = gaugelength_space
    elif skip_space:
        combined_space = interrogator_space
    else:
        combined_space = convolve(interrogator_space, gaugelength_space,
                                  'valid')[1:]
        # ensure we get a scalar from the conv
    conv_output = np.empty(axial_wavenumber.shape[:-1])
    for it in range(conv_output.shape[0]):
        for jit in range(conv_output.shape[1]):
            out = convolve(combined_space/points_per_gl,
                           np.exp(2j*np.pi*axial_wavenumber[it, jit]
                                  * gaugelength_pos),
                           'valid') / points_per_gl
            conv_output[it, jit] = np.squeeze(np.abs(out))
    return conv_output


def directivity_wn(grazing_angle, source_frequency,
                   angle_units="degrees", c=1500, gl_base=1.02, gl_mult=2,
                   skip_pulse=False, skip_space=False):
    """Frequency-dependent directivity effects of an interrogator.
    Works on wavenumber responses instead of spatial responses, laddering on
    the Fourier transform property that convolution in time ~ multiplication
    in frequency.

    *I am on the right track, but I need to revisit the moments I have.*

    :param grazing_angle: Grazing angle of the incident wave.
    :type grazing_angle: array-like
    :param source_frequency: Temporal frequency of the incident wave.
    :type source_frequency: array-like
    :param angle_units: Units of the grazing angle.
    :type angle_units: {'degrees', 'radians'}, optional
    :param c: Speed of sound in water.

        .. note ::
            Acrylic plastic has sound speed near 2.75 km/s.
            Silica glass has a pressure-dependent sound speed. In (Zha et al,
            1994), it has a sound speed around 6 km/s at low pressure.

    :type c: int or float, optional
    :param gl_base: Channel spacing. Default gives a typical value.
    :type gl_base: float > 0, optional
    :param gl_mult: Gauge-length multiplier. The channel spacing in metres is
        found as this number times ``gl_base``. The width of the window
        function is twice of this. For example, if ``gl_base=1.02`` and
        ``gl_mult=2``, channels are spaced 2.04 m apart and the gauge length
        equals 4.08 m.
    :type gl_mult: (int or float) >= 1, optional
    :param skip_pulse: Whether to omit the autocorrelation effects of the
        pulse from the interrogator.
    :type skip_pulse: bool, optional
    :param skip_space: Whether to omit the effects of the spatial windowing
        taking effect on the interrogator.
    :type skip_space: bool, optional
    :returns: Signal gain due to directivity.

    """
    if skip_pulse and skip_space:
        raise ValueError('cannot skip both spatial windowing effects')
    if gl_base <= 0:
        raise ValueError('channel spacing must be positive')

    grazing_angle = validate_angles(grazing_angle, angle_units)
    # we go from Hz [1/s] to 1/m
    axial_wavenumber = np.atleast_1d(source_frequency)\
        * np.cos(grazing_angle[..., None])/c

    def pulse_wavelen(norm_wavenum, periodic=False,
                      norm_cos_bounds=[0.4, 0.45]):
        """Reconstruct the wavelength response."""
        if periodic:
            norm_wavenum = (norm_wavenum + 0.5) % 1 - 0.5  # modulus to +/- 0.5
        return np.where(np.abs(norm_wavenum) > norm_cos_bounds[1],
                        0,
                        np.where(np.abs(norm_wavenum) < norm_cos_bounds[0],
                                 1,
                                 (1+np.cos(np.pi*(np.abs(norm_wavenum)
                                                  - norm_cos_bounds[0])
                                           / (np.diff(norm_cos_bounds))))/2))

    interrogator_wavenumber = pulse_wavelen(axial_wavenumber*gl_base)

    def gaugelength_wavelen(norm_wavenum):
        """Response of the gauge-length effects. Effectively a sinc-squared
        with zero crossings at integer multiples of 2 over gauge length."""
        return np.sinc(norm_wavenum*gl_base/gl_mult)**2

    gaugelength_wavenumber = gaugelength_wavelen(axial_wavenumber)
    if skip_pulse:
        return gaugelength_wavenumber
    elif skip_space:
        return interrogator_wavenumber
    else:
        return interrogator_wavenumber * gaugelength_wavenumber


def estimate_noise(data_path, run_data, start, stop,
                   extra_offset=0, band="B_2", *,
                   cachepath=None,
                   delegate=False,
                   stft_params={'win': np.hanning(1024),
                                'hop': 512,
                                'fs': 25000,
                                'scale_to': 'psd'}):
    """Estimate noise spectral density on the cable.

    :param data_path: Path to the raw data.
    :type experiment: path-like
    :param run_data: Dictionary of run data.
    :type run_data: dict from :py:func:`dasprocessor.constants.get_run`
    :param start: Start of channel slice.
    :type start: int
    :param stop: Stop of channel slice.

        .. warning ::
            Memory requirements scale with ``stop - start``. Using even a
            modest number of channels requires a lot of memory. If you want
            to use many channels, consider using ``delegate=True`` and
            performing the averaging yourself.

    :type stop: int
    :param extra_offset: Additional offset for extracting segments of noise
        in samples.
    :type extra_offset: int, optional
    :param band: JANUS frequency band. Used for selecting the length of the
        slice.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'}, optional
    :param cachepath: Path to cached data. May speed up loading if given.
        Passed to :py:func:`dasprocessor.saveandload.load_interrogator_data`.
    :type cachepath: path-like, optional
    :param delegate: Whether to delegate the averaging step to the user.
        The data will not be trimmed nor converted to dB in this case.
        Practical if using a large sample size. Does not delegate by default.
    :type delegate: bool, optional
    :param stft_params: Dictionary of arguments to
        :py:func:`scipy.signal.ShortTimeFFT`.
    :type stft_params: dict, optional
    :returns: Estimate of noise spectral density in
        :math:`\\mathrm{dB~re~1~nanostrain/\\sqrt{Hz}}` if ``delegate=False``,
        or collection of spectrograms in strain²/Hz if ``delegate=True``.
    :returns: If ``delegate=False``, also the frequency axis of the spectral
        density estimates.

    """
    all_data = load_interrogator_data(data_path, run_data['time_range'][0],
                                      run_data['time_range'][1],
                                      channels=slice(start, stop),
                                      on_fnf='cache', out='npz',
                                      cachepath=cachepath)
    cable_data = ensure_numpy(all_data['y'])
    start_idx = np.arange(run_data['sequence_count'])\
        * run_data['sequence_period'] + run_data['offset_in_samples']\
        + extra_offset
    print(cable_data.shape, start_idx[-1])
    noise_data = np.vstack([get_all_packets(cable_data[:, it], start_idx, band)
                            for it in range(stop-start)])
    stft = ShortTimeFFT(**stft_params)
    frequency_axis = rfftfreq(stft.mfft, 1/all_data['fs'])
    spectrograms = stft.spectrogram(noise_data)  # shape (npack, freq, time)
    if delegate:
        return spectrograms
    return (10*np.log10(np.mean(spectrograms[..., 2:-2], axis=(-3, -1)))
            + 180, frequency_axis)
    # the 2:-2 slice along the last axis is there to trim artifacts at the ends


def validate_angles(grazing_angle, angle_units):
    """Validate angle units and convert angles to radians on success.

    :param grazing_angle: Grazing angle of the incident wave.
    :type grazing_angle: array-like
    :param angle_units: Units of the grazing angle.
    :type angle_units: {'degrees', 'radians'}
    :returns: Grazing angles in radians.
    :raises ValueError: if the angle unit is not supported.

    """
    angle_units = angle_units.lower()
    if angle_units == "degrees":
        grazing_angle = np.radians(grazing_angle)
    elif angle_units != "radians":
        raise ValueError(f"'{angle_units}' is not a recognised angular unit")

    return grazing_angle


def main():
    """Testing function.
    :returns: TODO

    """
    import matplotlib.pyplot as plt
    plt.rc("font", size=12, family="Nimbus Sans")
    res = 40
    ccaxis = np.linspace(0, res, res + 1)
    ccaxis[0] = 0.001  # fix div-by-zero issues
    theta = np.linspace(0, 90, 451)
    mechastrain = np.vstack([mechanical_strain(theta, -60,
                                               young_modulus_ratio=it/res)
                             for it in ccaxis]).real
    # mechastrain += 20*np.log10(ccaxis[..., None])
    # mechanical_strain: pressure/stress transfers, cf pressure continuity BC
    # above line changes that to assume strain to transfer
    # plt.pcolor(theta, ccaxis/res, mechastrain, vmin=-60, vmax=0,
    #            cmap='inferno')
    # plt.colorbar(label='Sensitivity (dB re X nanostrain/µPa)')
    clist = [["#330000", "#770000", "C3"],
             ["#7b3000", "#ca6b39", "C1"],
             ["#405612", "#5b6a00", "C2"],
             ["#122559", "#284e82", "C0"],
             ["#292159", "#5e468c", "C4"]]
    # colour list with groups of colours, some from the MMORPG Guild Wars 2
    clist_monochrome = ["#0E1C25", "#284e82", "C0", "#85B3C1", "#82C8FF"]
    plt.figure()
    for it, dx in enumerate([0]):  # , 4, 8, 20, 40]):
        series = mechanical_strain(theta, -66,
                                   young_modulus_ratio=dx/res)
        plt.plot(theta, series,  # + 20*np.log10(dx/res),
                 label=f"{dx/res:.2f}",
                 zorder=2.5 - 0.1*it,
                 color=clist_monochrome[it])

    plt.grid()
    plt.ylim(bottom=-245, top=-195)
    plt.xlim(left=0, right=90)
    plt.title('Stress-strain mechanisms')
    plt.xlabel('Grazing angle (degrees)')
    plt.ylabel('Relative OPL change (dB re X billionths)')
    plt.legend(title="Young's modulus ratio")
    plt.show()
    # basic stress and strain up to this line

    # theta = np.linspace(0, 90, 451)  # we changed that to become "common"
    f = np.linspace(0, 10000, 1001)
    freq_count = 256
    points_per_gl = 4
    gl_mult = 1
    out = directivity_wn(theta, f,  # freq_count=freq_count,
                         # points_per_gl=points_per_gl,
                         gl_mult=gl_mult, skip_pulse=False)
    print(out.shape)
    out_dB = 20*np.log10(np.abs(out))

    # plt.pcolor(theta, f, out_dB, cmap='inferno', vmin=-60)
    # plt.colorbar(label='Relative gain (dB)')
    outnogauge = directivity_wn(theta, f,  # freq_count=freq_count,
                                # points_per_gl=points_per_gl,
                                gl_mult=gl_mult, skip_space=True)
    outng_dB = 20*np.log10(np.abs(outnogauge))

    outonlygauge = directivity_wn(theta, f,  # freq_count=freq_count,
                                  # points_per_gl=points_per_gl,
                                  gl_mult=gl_mult, skip_pulse=True)
    outgo_dB = 20*np.log10(np.abs(outonlygauge))

    # wavenumber on X axis
    figwn, axwn = plt.subplots()
    axwn.grid()
    gl_base = 1.02
    c = 1500
    wnaxis = f*gl_base/c
    labels = ["Pulse shape", "Gauge length", "Both effects"]
    for it, which in enumerate((outng_dB, outgo_dB, out_dB)):
        axwn.plot(wnaxis, which[0], label=labels[it],
                  color=clist_monochrome[it])

    axwn.set_ylim(bottom=-60, top=10)
    axwn.set_xlim(left=0, right=6)
    axwn.set_xlabel('Normalised wavenumber (times 1/1.02 m)')
    axwn.set_ylabel('Frequency response (dB)')
    axwn.legend(title="Phenomena")
    plt.show()

    # angle on X axis
    figpulse, axp = plt.subplots()
    figgauge, axg = plt.subplots()
    figboth, axboth = plt.subplots()

    # used to plot with frequency on X axis, now want angle there
    for it, dx in enumerate([80, 150, 300, 600, 1000]):
        axp.plot(theta, outng_dB[:, dx], label=f"{dx*10} Hz",
                 color=clist_monochrome[it])
        axg.plot(theta, outgo_dB[:, dx], label=f"{dx*10} Hz",
                 color=clist_monochrome[it])
        axboth.plot(theta, out_dB[:, dx], label=f"{dx*10} Hz",
                    color=clist_monochrome[it])

    for ax in [axp, axg, axboth]:
        ax.grid()
        ax.set_title('Beam-pattern effects: '
                     f'{"instrument pulse" if ax is axp else "gauge length"}'
                     f'{" and instrument pulse" if ax is axboth else ""}')
        ax.set_ylim(bottom=-45, top=5)
        ax.set_xlim(left=0, right=90)
        ax.set_xlabel('Grazing angle (\u00b0)')
        ax.set_ylabel('Frequency response (dB)')
        ax.legend(title="Source frequency")

    plt.show()

    # interrogator additional effects up to this line

    combinedeffect = mechastrain[..., None] + out_dB
    combinedeffect[np.isneginf(combinedeffect)] = -340
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 6)
    ax = ax.flatten()
    for it, dx in enumerate([4, 8, 20, 40]):
        toplot = combinedeffect[dx]
        for clr, jit in enumerate([80, 150, 300, 600, 1000]):
            ax[it].plot(theta, toplot[:, jit], color=clist_monochrome[clr],
                        label=f"{jit/100:.1f} kHz")

        ax[it].set_title(f'Young\'s modulus ratio {dx/res:.1f}')
        ax[it].set_ylim(bottom=-100, top=-20)
        ax[it].set_xlim(left=0, right=90)
        ax[it].grid()
        if it > 1:
            ax[it].set_xlabel('Grazing angle (deg)')
        if it % 2 == 0:
            ax[it].set_ylabel('Frequency response\n(dB re X nstr/µPa)')
        if it == 3:
            ax[it].legend()

    plt.show()

    area = range(120, 240, 12)
    specgram = np.vstack([estimate_noise("/media/emil/JohnDisk_1p8TB/DASComms"
                                         "_25kHz_GL_2m/20240503/dphi/",
                                         get_run("2024-05-03", 1),
                                         it, it + 12, round(14.2*25000),
                                         cachepath="./resources/backups",
                                         delegate=True)
                          for it in area])
    estspecgram = np.mean(specgram[..., 2:-2], axis=(-3, -1))
    freqaxis = rfftfreq(1024, 4e-5)
    plt.plot(freqaxis, 10*np.log10(estspecgram)+180)
    plt.title('Noise spectral density estimates\nbased on channels 120 to 240'
              '\nand the first run')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Noise PSD (dB re 1 nanostrain/sqrt(Hz))')
    plt.show()


if __name__ == "__main__":
    main()
