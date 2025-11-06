"""Signal processing specifically tailored for JANUS packets."""

import numpy as np

from .constants import janus_frequency_count, janus_frequency_bands, \
    baseline_chips, hopping_pattern
from .utils import get_janus_duration_samples, get_decoder, to_pandas


def get_all_packets(rx, indexes, band, rate=25000):
    """Extract all JANUS packets from a single channel.

    :param rx: One channel of data.
    :type rx: 1-D sequence of floats
    :param indexes: Starting indexes of detected packets.
    :type indexes: 1-D sequence of ints
    :param band: Target JANUS band.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'}
    :param rate: Sampling rate.
    :type rate: int
    :returns: A shape-``(N,l)`` array of detected JANUS packets, where ``N``
        is the number of detected packets and ``l`` is the length of one packet
        in samples.

    """
    window_length = round(rate*baseline_chips
                          / janus_frequency_bands[band]["df"])
    out = np.zeros([len(indexes), window_length], dtype='float32')
    for it, k in enumerate(indexes):
        out[it] = rx[k:k+window_length]

    return out


def chipwise_dft(packets, band, window_function=np.ones, nchip=baseline_chips,
                 rate=25000):
    """Discrete Fourier-transform many packets on a chip-by-chip basis.

    :param packets: Packets in time domain.
    :type packets: n-D-sequence of length-(k*26*nchip) sequences of floats
    :param band: JANUS band to use.
    :type band: {'B_4', 'B_2', 'B', 'C', 'A'}
    :param window_function: Windowing to apply to the data, chip by chip,
        before DFT. Silently ignored.
    :type window_function: function with signature (int) -> array_like
    :param nchip: Number of chips in the packet. Provide if the packet contains
        cargo.
    :type nchip: int, optional
    :param rate: Sampling rate. If ``None``, the function estimates the
        sampling rate.
    :type rate: int or None, optional
    :returns: n-D sequence of (26,nchip)-arrays of chips detected.

    """
    if rate is None:
        rate = packets.shape[-1]*janus_frequency_bands[band]['df']/nchip

    frequency_range = (np.arange(janus_frequency_count)
                       * janus_frequency_bands[band]['df']
                       + janus_frequency_bands[band]['fmin'])
    chip_length = round(rate/janus_frequency_bands[band]['df'])
    packets = np.reshape(packets, packets.shape[:-1] + (nchip, chip_length))
    vertical = np.arange(chip_length)[:, None]
    dft_matrix = np.exp(2j*np.pi*frequency_range/rate*vertical)
    dft_output = packets @ dft_matrix
    return dft_output.astype('complex64')


def baseline_snr(dfts):
    """Signal-to-noise ratio (SNR) of DFT-ed JANUS baseline packets.

    Useful for determining which packets should be considered in a sensitivity
    analysis, as well as for array combination of many receptions.

    :param dfts: DFTs of JANUS baseline packets.
    :type dfts: n-D sequence of (nchip,26)-arrays of complex
    :returns: Both estimated SNR and estimated noise RMS (incoherent sum).
        Both are n-D arrays with shape ``dfts.shape[:-2]``.

    .. note ::
        The SNR is estimated by summing the energy in the frequency bins
        corresponding to permitted chips, then dividing the result by the
        sum of the same pattern shifted six steps forward.

    """
    sigpower = np.abs(dfts[..., np.tile(np.arange(dfts.shape[-2])
                                        .astype('int16'), 2),
                           (hopping_pattern.flatten()*2+np.array([[0], [1]]))
                           .flatten()])
    noisepower = np.abs(dfts[..., np.tile(np.arange(dfts.shape[-2])
                                          .astype('int16'), 2),
                        np.roll((hopping_pattern.flatten()*2+np.array([[0],
                                                                       [1]]))
                        .flatten(), 6)])
    # previous iterations made the first sample very weird, possibly due to
    # not filtering the data, so for historical reasons I keep that fix here
    return (10*np.log10(np.sum(sigpower[...,
                                        np.hstack(
                                            [np.arange(1, len(sigpower)//2),
                                             np.arange(-len(sigpower)//2+1, 0)]
                                            )
                                        ]**2, axis=-1)
                        / np.sum(noisepower[...,
                                            np.hstack(
                                                [np.arange(1,
                                                           len(sigpower)//2),
                                                 np.arange(-len(sigpower)//2+1,
                                                           0)]
                                                 )
                                            ]**2, axis=-1)),
            # above is SNR, below is noise RMS
            np.sqrt(np.sum(noisepower[...,
                                      np.hstack([np.arange(1,
                                                           len(sigpower)//2),
                                                 np.arange(-len(sigpower)//2+1,
                                                           0)]
                                                )
                                      ]**2, axis=-1))
            )


def stack_packets(dfts, snr, noise, min_snr=3):
    """Combine multiple receptions with sufficient SNR.

    :param dfts: :py:func:`chipwise_dft` estimates.
    :type dfts: shape-(nchannels, npackets, nchip, 26) array of complex
    :param snr: Signal-to-noise ratio from :py:func:`baseline_snr`. In dB.
    :type snr: shape-(nchannels, npackets) array of float
    :param noise: Root-mean-square value of noise from :py:func:`baseline_snr`.
        **Not** in dB.
    :type noise: shape-(nchannels, npackets) array of float
    :param min_snr: Lowest signal-to-noise ratio to consider for stacking.
        In dB.
    :type min_snr: float, optional
    :returns: A (npackets, nchip, 26)-shape array with stacked channel data.

    """
    combinable = snr >= min_snr
    # Scale the DFTs to the same noise level
    return np.sum(np.abs(dfts)*(combinable.astype('float32')/noise)[...,
                                                                    None,
                                                                    None],
                  axis=0)


def deinterleave(bits, interest=slice(32, None), *, q=13):
    """Deinterleave a JANUS baseline packet.

    :param bits: Bit sequence.
    :type bits: nchip-length sequence of array_like of floats
    :param interest: Which part of the bit sequence is actually the packet.
    :type interest: slice, optional
    :param q: Prime number in modulo operator.
    :type q: int, optional
    :returns: Deinterleaved bit sequence with preamble bit sequence discarded.
    """
    outbits = bits[interest]
    rightbits = np.zeros(outbits.shape)
    rightbits[((np.arange(len(outbits))*q) % len(outbits)).astype('int64')]\
        = outbits
    return rightbits.astype('int8')


def decode(dfts):
    """Decode JANUS packet from DFTs.

    :param dfts: DFTs of packets to decode.
    :type dfts: output from :py:func:`chipwise_dft`
    :returns: All packets that were decoded as arrays of ones and zeroes.

    .. seealso ::
        * :py:func:`chipwise_dft`

    """
    # since nondetections are all-zero and detections are non-zero,
    # looking at the first chip and first band of the DFTs will reveal
    # where we should decode
    isrelevant = np.nonzero(dfts[..., 0, 0])
    # 32 chips for the preamble, 8·2 chips for giving the decoder one way out
    outarray = -np.ones(dfts.shape[:-2] + ((dfts.shape[-2]-32)//2 - 8,),
                        dtype='int8')
    length = len(isrelevant[0])
    for it in range(len(isrelevant[0])):
        target_idx = tuple(x[it] for x in isrelevant)
        my_dft = np.abs(dfts[target_idx]).T
        investigated = [2*hopping_pattern, 2*hopping_pattern+1]
        # Hard decision-making per chip: was it a zero or a one?
        # Can be made soft decision by dividing each element by the sum of
        # the two possible bins, then scaling to desired precision.
        foundbits = np.argmax(my_dft[np.hstack(investigated),
                              np.arange(hopping_pattern.shape[0])
                                     .astype('int64')[:, None]], axis=-1)
        # Deinterleave the bits
        convcodedbits = deinterleave(foundbits)
        # Convolutional decoding
        decbitshard = get_decoder().viterbi_decoder(convcodedbits,
                                                    metric_type='hard')\
            .astype('int8')
        # String-like presentation
        outarray[target_idx] = decbitshard

    return outarray


def crc8(code, poly=np.array([1, 0, 0, 0, 0, 0, 1, 1, 1])):
    """Cyclic redundancy check of baseline packet.

    :param code: The first 56 bits of a baseline packet.
        If an ``int`` is given, the version number
        occupies the most significant bits. In other words, bit 1 in the
        specification is considered the most significant.
    :type code: int or str or 56-sequence of {0,1}
    :param poly: CRC8 polynomial. Defaults to ANEP-87's CRC8 polynomial.
    :type poly: 9-sequence of {0,1}
    :returns: The calculated CRC8 of the packet as an integer.

    """
    if isinstance(code, str):
        codeasint = int(code, 2)
    elif isinstance(code, np.typing.ArrayLike):
        codeasint = int(np.sum(code*2**np.arange(55, -1, -1)))
    else:
        codeasint = code
    # how much to shift the baseline packet sans CRC to leave only the CRC
    # when finished
    paddingfactor = 2**(len(poly)-1)
    polyasint = int(np.sum(poly*2**(len(poly)-np.arange(len(poly))-1)))
    codeasint *= paddingfactor
    exp = 64
    # do bitwise XOR every time the MSB of the CRC polynomial hits a one
    while codeasint >= paddingfactor:
        if codeasint & (1 << (exp+len(poly)-1)) != 0:
            codeasint ^= polyasint*2**exp
        # shift CRC polynomial one step to the right
        exp -= 1
    return codeasint


def main():
    """Demonstrate the features in this module.

    Intended for internal testing. Feel free to study the source.

    """
    from sys import argv

    from .constants import get_run, frequency_bands
    from .saveandload import load_interrogator_data, load_peaks

    import matplotlib.pyplot as plt

    skipprocessing = False
    choicerun = int(argv[1]) if len(argv) > 1 else 1
    band = argv[2] if len(argv) > 2 else "B_4"
    myrun = get_run("2024-05-03", choicerun)
    for it in range(24, 300, 12):
        wantedchans = slice(it, it+12)
        dftoutfile = f"../resources/{band}/dfts-{wantedchans.start}-{wantedchans.stop}-"\
                     f"run{choicerun}.npz"
        if not skipprocessing:
            mydata = load_interrogator_data("/home/emil/Dokument/Data/DASOne"
                                            "Way/DasKabel/rawdata/DASComms"
                                            "_25kHz_GL_2m/20240503/dphi",
                                            *myrun["time_range"],
                                            channels=wantedchans,
                                            on_fnf="cache",
                                            filter_band=frequency_bands[band],
                                            cachepath="/home/emil/Dokument/Data/"
                                            "DASOneWay/DasKabel/backups",
                                            verbose=True,
                                            out="npz")
            savepath = f"../resources/{band}/peaks-{wantedchans.start}-"\
                       f"{wantedchans.stop}-run{choicerun}.json"
            peaks = load_peaks(savepath)
            goalarray = np.zeros([wantedchans.stop-wantedchans.start,
                                  myrun['sequence_count'],
                                  get_janus_duration_samples(band)])
            for it in range(goalarray.shape[0]):
                goalarray[it, [int(x) for x in peaks[f"{wantedchans.start+it}"]
                               .keys()]]\
                    = get_all_packets(mydata['y'][:, it],
                                      peaks[f"{wantedchans.start+it}"].values(),
                                      band)
            dfts = chipwise_dft(goalarray, band)
            np.savez_compressed(dftoutfile, dft=dfts)
            everything = {'dft': dfts}
        else:
            everything = np.load(dftoutfile)
            dfts = everything['dft']

        test_snr = True
        if test_snr:
            snrtest = baseline_snr(dfts)
            np.savez_compressed(dftoutfile, dft=dfts, snr=snrtest[0],
                                noise=snrtest[1])
            everything['snr'] = snrtest[0]
            everything['noise'] = snrtest[1]

        print(wantedchans.stop)

        rundecode = True
        if rundecode:
            packets = decode(dfts)
            np.savez_compressed(dftoutfile, dft=dfts, snr=everything['snr'],
                                noise=everything['noise'], packets=packets)
            everything['packets'] = packets
        else:
            packets = everything['packets']

        demostack = False
        if demostack:
            stacked = stack_packets(dfts, everything['snr'], everything['noise'])
            stackpacket = decode(stacked)
            print(stackpacket)
            np.savez_compressed(dftoutfile, dft=dfts, snr=everything['snr'],
                                noise=everything['noise'], packets=packets,
                                stackpack=stackpacket)

        crc8s = np.zeros(packets.shape[:-1])
        checkedout = np.zeros(crc8s.shape, dtype='bool')
        print(everything['snr'].shape)
        detections = np.nonzero(np.isfinite(everything['snr']))
        for r, c in zip(*detections):
            crc8s[r, c] = crc8(packets[r, c, :-8])
            checkedout[r, c] = int(''.join(str(x) for x in packets[r, c, -8:]), 2)\
                == crc8s[r, c]
        print(np.sum(checkedout, axis=0))
        table = to_pandas(packets, crc8s, start=wantedchans.start)

        singleresult = f"../resources/{band}/packets-{wantedchans.start}-"\
                       f"{wantedchans.stop}-run{choicerun}.csv"
        table.to_csv(singleresult, index_label="CxxxPxxx")

    print(packets.shape)


if __name__ == "__main__":
    main()
