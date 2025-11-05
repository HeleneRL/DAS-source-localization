import numpy as np
from scipy.signal import resample, butter, sosfilt


def bpsk(x, operation, zero_phase=0):
    """Binary phase-shift keying.

    :param x: Received symbols when ``operation == 'demodulate'``,
        bit stream to modulate when ``operation == 'modulate'``.
    :type x: sequence of complex or ``{0, 1}``
    :param operation: Operation to perform.
    :type operation: ``{'modulate', 'demodulate'}``
    :param zero_phase: Phase of the symbol representing zero.
    :type zero_phase: ``[-np.pi, +np.pi)``, optional
    :returns: Depends on the choice of ``operation``.

        * When ``operation == 'demodulate'``: Bit stream as hard decisions.
        * When ``operation == 'modulate'``: Symbols in baseband.

    """
    match operation:
        case 'modulate':
            return bpsk_encode(x, zero_phase)
        case 'demodulate':
            return bpsk_hard_decode(x, zero_phase)
        case _:
            raise ValueError(f'operation {operation} is not supported')


def bpsk_hard_decode(x, zero_phase=0):
    """Hard decision decoding for binary phase-shift keyed symbols.

    :param x: Received symbols.
    :type x: array-like of complex, shape ``(foo)``
    :param zero_phase: Phase shift of the zero symbol, in radians.
    :type zero_phase: float in [-pi, +pi), optional
    :returns: Maximum-likelihood data, shape ``(foo, 1)``.

    """
    return (np.abs((np.angle(x) + np.pi - zero_phase) % (2*np.pi) - np.pi)
            <= np.pi/2)[..., np.newaxis]


def bpsk_encode(x, zero_phase=0):
    """Encode bit stream as binary phase-shift keyed symbols.

    :param x: Bit stream to modulate.
    :type x: array-like of ``{0, 1}``, shape ``(foo[, 1])``
    :param zero_phase: Phase shift of the zero symbol, in radians.
    :type zero_phase: float in [-pi, +pi), optional
    :returns: Symbols, shape ``(foo)``.

    """
    if x.shape[-1] == 1:
        x = x[..., 0]
    return np.exp(1j*(np.pi*x + zero_phase))


def decision_feedback_demodulation(passband, fc,
                                   rate=44100, modulation='bpsk',
                                   feedforward_len=10, feedback_len=10,
                                   taps_per_symbol=2, training_sequence=None,
                                   **lms_kwargs):
    """Demodulate coherently modulated packets by decision feedback
        equalisation.

    :param passband: Received signal in passband. May be filtered in advance.
        Expected to be reasonably aligned in time.
    :type passband: array-like of floats
    :param fc: Carrier frequency in hertz.
    :type fc: float
    :param rate: Sample rate of the input stream.
    :type rate: int, optional
    :param modulation: How the packet was modulated.
    :type modulation: str or ``Callable``, optional
    :param feedforward_len: Length of the feedforward filter, in taps.
    :type feedforward_len: int > 0, optional
    :param feedback_len: Length of the feedback filter, in taps.
    :type feedback_len: int > 0, optional
    :param taps_per_symbol: Filter coefficients per symbol. Setting this
        higher than one enables fractionally spaced equalisation and shortens
        the filters in absolute time.
    :type taps_per_symbol: int > 0, optional
    :param training_sequence: Training sequence used, if any.
    :type training_sequence: sequence of ``{0, 1}`, optional
    :returns: TODO

    -----
    Notes
    -----

    A ``Callable`` passed to ``modulation`` is expected to accept a
    ``complex`` argument, or an array-like thereof with shape ``(foo)``,
    and return a ``(foo, nbits)`` shaped array-like of zeroes and ones.
    For binary modulation (implies ``nbits == 1``), the return value may
    have the same shape as the argument, in which case a new axis will
    be appended to the return value.

    """
    if isinstance(modulation, str):
        modulation = eval(modulation)  # expects this to be a function

    lowpass = butter(10, fc/2, output='sos', fs=rate)  # baseband filter

    baseband = np.exp(-2j*np.pi*fc/rate)*passband
    baseband = sosfilt(lowpass, baseband)
    baseband = resample(baseband, round(len(passband)*taps_per_symbol*fc/rate))
    bb_len = len(baseband)
    baseband = np.hstack([np.zeros(feedback_len),
                          baseband,
                          np.zeros(feedforward_len-1)])
    decided_symbols = np.empty(bb_len, dtype='complex64')*np.nan
    decided_symbols[:len(training_sequence)] = modulation(training_sequence,
                                                          "modulate",
                                                          zero_phase=np.pi/2)
    # above are already decided symbols
    def shifted_cir(shift_amount):
        if shift_amount == 0:
            return channel_estimate
        elif shift_amount > 0:
            return np.hstack([np.zeros(shift_amount*taps_per_symbol),
                              channel_estimate[:-(shift_amount
                                                  * taps_per_symbol)]])
        elif shift_amount < 0:
            return np.hstack([channel_estimate[-(shift_amount
                                                 * taps_per_symbol):],
                              np.zeros(-shift_amount*taps_per_symbol)])
        else:
            raise TypeError(f'illegal shift amount {shift_amount}')


    # self-study help: h(k) is simply h(0) shifted a bit, and if h(0) covers
    # all of the impulse response, we shift in taps_per_symbol zeroes. yes?
    # follows Stojanovic, Freitag and Johnson (1999): init channel estimate
    channel_estimate = np.zeros(feedforward_len + feedback_len)
    channel_estimate[feedback_len] = 1  # h[-1]
    forget_factor = lms_kwargs.get("forget_factor", 0.99)
    for it in range(bb_len):
        if np.isnan(decided_symbols[it]):
            pass  # here goes a symbol decision
        channel_estimate *= forget_factor
        channel_estimate += (1-forget_factor)\
                * baseband[it:it+len(channel_estimate)]\
                * np.conj(decided_symbols[it])


def main():
    x = np.exp(np.linspace(0, 2*np.pi, 100, endpoint=False)*1j)
    print(bpsk_hard_decode(x).flatten())
    print(bpsk_hard_decode(x, zero_phase=2).flatten())


if __name__ == "__main__":
    main()
