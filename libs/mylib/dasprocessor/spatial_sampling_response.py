from typing import Tuple

import numpy as np
from scipy import constants
from scipy import signal

c = constants.c
refractiveIndex = 1.4677
samplingRate = 100e6
dx_sampling = c / (2 * refractiveIndex * samplingRate)


def instrument_wavenumber_response(nFreq: int = 64) -> Tuple[np.ndarray,
                                                             np.ndarray]:
    """Compute the instrument wavenumber response.

    :param nFreq: The number of frequency points to compute the response at.
    :type nFreq: int, optional

    :returns wavenumber: The wavenumber array.
    :returns wavenumber_response: The instrument wavenumber response.
    """
    swrange = np.r_[-45e6, 45e6]
    napod = int(np.diff(swrange)/samplingRate*nFreq)
    df = samplingRate/nFreq
    apodratio = 1.0 / 9
    apod = np.zeros(nFreq)
    nstart = int((swrange[0]+samplingRate/2) / df)
    apod[nstart:nstart+napod] = signal.windows.tukey(napod, apodratio)
    wavenumber_response = np.fft.fftshift(apod)
    wavenumber = np.fft.fftfreq(nFreq, d=dx_sampling)
    return wavenumber, wavenumber_response


def instrument_spatial_response(nFreq: int = 64,
                                interpolation: int = 1,
                                method: str = "resample") -> Tuple[np.ndarray,
                                                                   np.ndarray]:
    """Compute the instrument spatial response.

    :param nFreq: The number of frequency points to compute the response at.
    :type nFreq: int, optional
    :param interpolation: The factor by which to interpolate
        the spatial response.
    :type interpolation: int, optional
    :param method: Which method to use for interpolation.
    :type method: {'resample', 'linear'}, optional

    :returns pos: The position array.
    :returns spatial_response: The instrument spatial response.
    """
    wavenumber, wavenumber_response = instrument_wavenumber_response(nFreq)
    spatial_response = np.fft.ifftshift(np.fft.ifft(wavenumber_response).real)
    if interpolation > 1:
        match method:
            case "linear":
                spatial_response = np.interp(np.arange(nFreq * interpolation),
                                             np.arange(0,
                                                       nFreq * interpolation,
                                                       interpolation),
                                             spatial_response)
            case "resample":
                spatial_response = signal.resample(spatial_response,
                                                   interpolation * nFreq)
            case _:
                raise ValueError('unsupported interpolation method '
                                 f'{method}')

    N = len(spatial_response)
    pos = np.arange(-N // 2, N // 2) * dx_sampling / interpolation
    return pos, spatial_response


def gaugelength_wavenumber_response(gaugeLenghtChs: int = 4,
                                    nFreq: int = 64) -> Tuple[np.ndarray,
                                                              np.ndarray]:
    """Compute the gauge length wavenumber response.

    :param gaugeLenghtChs: The number of channels in the gauge length.
    :type gaugeLenghtChs: int, optional
    :param nFreq: The number of frequency points to compute the response at.
    :type nFreq: int, optional

    :returns wavenumber: The wavenumber array.
    :returns wavenumber_response: The gauge length wavenumber response.
    """
    win = np.zeros(nFreq)
    win[:gaugeLenghtChs*2 - 1] = np.convolve(np.ones(gaugeLenghtChs),
                                             np.ones(gaugeLenghtChs),
                                             mode='full')/gaugeLenghtChs
    wavenumber_response = np.abs(np.fft.fft(win))
    wavenumber = np.fft.fftfreq(nFreq, d=dx_sampling)

    return wavenumber, wavenumber_response


def gaugelength_spatial_response(
    gaugeLenghtChs: int = 4, nFreq: int = 64, interpolation: int = 1,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the gauge length spatial response.

    :param gaugeLenghtChs: The number of channels in the gauge length.
    :type gaugeLenghtChs: int, optional
    :param nFreq: The number of frequency points to compute the response at.
    :type nFreq: int, optional
    :param interpolation: The factor by which to interpolate
        the spatial response.
    :type interpolation: int, optional
    :param method: Which method to use for interpolation.
    :type method: {'resample', 'linear'}, optional

    :returns pos: The position array.
    :returns spatial_response: The gauge length spatial response.
    """
    wavenumber, wavenumber_response = gaugelength_wavenumber_response(
        gaugeLenghtChs, nFreq
    )
    spatial_response = np.fft.ifftshift(np.fft.ifft(wavenumber_response).real)
    if interpolation > 1:
        match method:
            case "linear":
                spatial_response = np.interp(np.arange(nFreq * interpolation),
                                             np.arange(0,
                                                       nFreq * interpolation,
                                                       interpolation),
                                             spatial_response)
            case "resample":
                spatial_response = signal.resample(spatial_response,
                                                   interpolation * nFreq)
            case _:
                raise ValueError('unsupported interpolation method '
                                 f'{method}')


    N = len(spatial_response)
    pos = np.arange(-N // 2, N // 2) * dx_sampling / interpolation

    return pos, spatial_response
