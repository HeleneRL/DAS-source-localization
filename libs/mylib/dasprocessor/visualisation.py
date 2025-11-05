"""
Module containing functions for drawing plots related to the DAS experiments.
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import numpy as np
from pandas import read_csv


def blend_colours(a, b, /, amount_of_a=50):
    """Blend colours together.

    :param a: First colour.
    :type a: color_like
    :param b: Second colour.
    :type b: color_like
    :param amount_of_a: How many percent of the new colour that should be
        ``a``.
    :type amount_of_a: float between 0 and 100, both inclusive
    :returns: The blended colour.

    """
    if amount_of_a < 0 or amount_of_a > 100:
        raise ValueError("cannot blend a negative percentage or more than 100%"
                         " of any colour")

    return (np.array(to_rgba(a))*amount_of_a
            + np.array(to_rgba(b))*(100-amount_of_a))/100


def style_axes(ax):
    """Apply default style to axes.

    :param ax: Axes handle.
    :type ax: :py:class:`matplotlib.axes.Axes`

    """
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(0.1)
    ax.tick_params(width=0.1, length=3, labelsize=5, pad=1)


def style_colorbar(cb):
    """Apply default style to colorbar.

    :param cb: Colorbar handle.
    :type cb: :py:class:`matplotlib.colorbar.Colorbar`

    """
    cb.outline.set_linewidth(0.1)


def plot_snr_rx_map(snr, successful, cosine_data, /,
                    margin_pt=24, fig_handle=None, first_channel=0,
                    marker_aspect_ratio=1, fill_angles=[75, np.nan],
                    give_handles=False, *,
                    figkwargs={}, scatterkwargs={}, colorbarkwargs={},
                    labelkwargs={}):
    """Generate a scatter plot in the style of UComms.

    :param snr: Signal-to-noise ratio estimates for each packet and channel.
    :type snr: array_like
    :param successful: Map of successful estimates.
    :type successful: array_like
    :param cosine_data: Cosines of grazing angles. Paints the area where cosine
        is small enough in magnitude in amber.
    :type cosine_data: array_like
    :param margin_pt: Margin of axes in points (1/72ths of an inch).
    :type margin_pt: float, optional
    :param fig_handle: Handle to the target figure. If not specified, creates
        a new figure. Overrides **figkwargs** if specified.
    :type fig_handle: :py:class:`matplotlib.figure.Figure`, optional
    :param first_channel: If plotting a subset of channels, gives the index
        of the first channel in the supplied data.
    :type first_channel: int, optional
    :param marker_aspect_ratio: Width-to-height ratio of the markers.
    :type marker_aspect_ratio: float, optional
    :param fill_angles: Range of grazing angles to paint. Use ``nan`` to
        remove the corresponding limit.
    :type fill_angles: 2-sequence of floats, optional
    :param give_handles: If ``True``, returns handles to the figure and axes
        instead of displaying them. If ``False``, displays the plot and
        returns ``None`` on close.
    :type give_handles: bool, optional
    :param figkwargs: Keyword arguments passed to ``plt.figure``.
    :type figkwargs: dict, optional
    :param scatterkwargs: Keyword arguments passed to ``plt.scatter``.
    :type scatterkwargs: dict, optional
    :param colorbarkwargs: Keyword arguments passed to ``plt.colorbar``.
    :type colorbarkwargs: dict, optional
    :param labelkwargs: ``fontdict`` passed to ``plt.xlabel``
        and ``plt.ylabel``.
    :type labelkwargs: dict, optional
    :returns: Handles to the created figure and axis,
        if ``give_handles`` is set.

    """
    figsize_pt = np.array(snr.shape)*np.array([marker_aspect_ratio, 1])\
        + 2*margin_pt
    if fig_handle is None:
        default_figkwargs = {"dpi": 288, "figsize": figsize_pt/72}
        default_figkwargs.update(**figkwargs)
        fig_handle = plt.figure(**default_figkwargs)

    fig_handle.clear()

    ax_handle = fig_handle.add_axes(np.hstack([margin_pt/figsize_pt,
                                               1-2*margin_pt/figsize_pt]),
                                    aspect=1/marker_aspect_ratio)
    style_axes(ax_handle)

    ch, pk = np.meshgrid(np.arange(snr.shape[0])+first_channel,
                         np.arange(snr.shape[1]), indexing="ij")
    detectmask = ~np.isnan(snr)
    transform_marker = MarkerStyle(scatterkwargs.get("marker", "s"),
                                   transform=Affine2D
                                   .from_values(marker_aspect_ratio, 0,
                                                0, 1,
                                                0, 0))
    default_scatterkwargs = {"cmap": "magma", "s": 1,
                             "marker": transform_marker,
                             "vmin": 0, "vmax": 18, "linewidths": 0}
    scatterkwargs.pop("marker", None)  # enforce the aspect-ratioed marker
    default_scatterkwargs.update(**scatterkwargs)
    # first fill the area with large enough grazing angle in amber
    cos_bounds = np.where(np.isnan(fill_angles), [1, 0],
                          np.cos(np.radians(fill_angles)))
    in_bounds = np.logical_and(np.abs(cosine_data) <= cos_bounds[0],
                               np.abs(cosine_data) >= cos_bounds[1])
    ax_handle.plot(ch[in_bounds], pk[in_bounds], linestyle='none',
                   marker=transform_marker, markersize=1,
                   markeredgewidth=0, color="#fc0",
                   zorder=-1)
    goodscat = ax_handle.scatter(ch[detectmask & successful],
                                 pk[detectmask & successful],
                                 c=snr[detectmask & successful],
                                 **default_scatterkwargs)
    default_scatterkwargs.update(s=0.25)
    badscat = ax_handle.scatter(ch[detectmask & ~successful],
                                pk[detectmask & ~successful],
                                c=snr[detectmask & ~successful],
                                **default_scatterkwargs)
    ax_handle.set_xlim(first_channel-0.5, first_channel-0.5+snr.shape[0])
    ax_handle.set_ylim(-0.5, -0.5+snr.shape[1])

    default_fontdict = {"size": 6}
    default_fontdict.update(**labelkwargs)
    ax_handle.set_xlabel('Channel number',
                         fontdict=default_fontdict,
                         labelpad=1)
    ax_handle.set_ylabel('Packet index',
                         fontdict=default_fontdict,
                         labelpad=1)

    # colourbar
    ax_cbar = fig_handle.add_axes((margin_pt/figsize_pt[0],
                                   1 - 0.875*margin_pt/figsize_pt[1],
                                   1 - 2*margin_pt/figsize_pt[0],
                                   0.1875*margin_pt/figsize_pt[1]))
    cb = fig_handle.colorbar(goodscat, cax=ax_cbar, orientation='horizontal')
    style_axes(ax_cbar)
    ax_cbar.tick_params(top=True, labeltop=True,
                        bottom=False, labelbottom=False, pad=0)
    ax_cbar.set_xlabel("SNR (dB)" if figsize_pt[0] < 360
                       else "Signal-to-noise ratio (dB)",
                       fontdict=default_fontdict, labelpad=2)
    ax_cbar.xaxis.set_label_position('top')
    style_colorbar(cb)
    if give_handles:
        return fig_handle, ax_handle
    else:
        plt.show()


def plot_sensitivity_curves(angles, sensitivity_means, sensitivity_stdevs,
                            frequency_bins, /,
                            show_stdev=1, colours=None, give_handles=False, *,
                            plot_kwargs={}, labelkwargs={}):
    """Generate a line plot in the spirit of the WUWNet paper.

    :param angles: Grazing angles in degrees.
    :type angles: array_like with shape (m,)
    :param sensitivity_means: Mean sensitivity estimates from
        :py:func:`dasprocessor.sensitivity.get_plottable_sensitivity`.
    :type sensitivity_means: array_like with shape (m, k)
    :param sensitivity_stdevs: Sensitivity standard-deviation estimates from
        :py:func:`dasprocessor.sensitivity.get_plottable_sensitivity`.
    :type sensitivity_stdevs: array_like with shape (m, k)
    :param frequency_bins: Frequency bins that each series in
        ``sensitivity_means`` shows.
    :type frequency_bins: array_like with shape (k+1,)
    :param show_stdev: Number of standard deviations up and down to draw
        subdued "uncertainty" curves. Set to 0 to disable them.
    :type show_stdev: nonnegative float, optional
    :param colours: Sequence of colours to use for the data series. If the
        uncertainty curves are enabled, they follow the same sequence mixed
        with 50% white. The list wraps if there are more series than colours.
    :type colours: sequence of colour-like, optional
    :param give_handles: If ``True``, returns handles to the figure and axes
        instead of displaying them. If ``False``, displays the plot and
        returns ``None`` on close.
    :type give_handles: bool, optional
    :param plot_kwargs: Keyword arguments passed to ``plt.plot``.
    :type plot_kwargs: dict, optional
    :param labelkwargs: ``fontdict`` passed to ``plt.xlabel`` and
        ``plt.ylabel``.
    :type labelkwargs: dict, optional
    :returns: Handles to the created figure and axis, if ``give_handles`` is
        set.

    """
    base_linewidth = plot_kwargs.pop("linewidth", rcParams["lines.linewidth"])
    fig_handle, ax_handle = plt.subplots()
    fig_handle.set_dpi(288)
    style_axes(ax_handle)
    for it in range(sensitivity_means.shape[-1]):
        mymean = sensitivity_means[:, it]
        mystd = sensitivity_stdevs[:, it]
        plot_hdl = ax_handle.plot(angles, mymean, label=f"{frequency_bins[it]}"
                                  f"–{frequency_bins[it+1]} Hz",
                                  color=None if colours is None else
                                  colours[it % len(colours)],
                                  linewidth=base_linewidth, **plot_kwargs)
        this_colour = plot_hdl[0].get_color()
        if show_stdev > 0:
            ax_handle.plot(angles, mymean-show_stdev*mystd,
                           label=f"_{frequency_bins[it]}–"
                                 f"{frequency_bins[it+1]} Hz (low)",
                           color=blend_colours(this_colour if
                                               colours is None else
                                               colours[it % len(colours)],
                                               "#ffffff"),
                           linewidth=0.5*base_linewidth,
                           zorder=-1,
                           **plot_kwargs)
            ax_handle.plot(angles, mymean+show_stdev*mystd,
                           label=f"_{frequency_bins[it]}–"
                                 f"{frequency_bins[it+1]} Hz (high)",
                           color=blend_colours(this_colour if
                                               colours is None else
                                               colours[it % len(colours)],
                                               "#ffffff"),
                           linewidth=0.5*base_linewidth,
                           zorder=-1,
                           **plot_kwargs)

    default_fontdict = {"size": 6}
    default_fontdict.update(**labelkwargs)
    ax_handle.set_xlabel("Grazing angle (degrees)", fontdict=default_fontdict)
    ax_handle.set_xlim(np.min(angles), np.max(angles))
    ax_handle.set_ylabel("Estimated sensitivity (dB re nanostrain/µPa)",
                         fontdict=default_fontdict)
    ax_handle.legend(prop=default_fontdict)
    if give_handles:
        return fig_handle, ax_handle
    else:
        plt.show()


def main():
    """Demonstrates the UComms part of this module.

    Should not be run by the end user.

    """
    import pandas as pd
    slc = slice(24, 300)
    band = "B_2"
    rng = range(slc.start, slc.stop, 12)
    run = 2
    rundata = np.vstack([np.load(f'../resources/{band}/dfts-{it}-{it+12}'
                                 f'-run{run}.npz')['snr'] for it in rng])
    tabdata = pd.concat([read_csv(f'../resources/{band}/packets-{it}-{it+12}-'
                                  f'run{run}.csv') for it in rng])
    gooddata = np.reshape((tabdata['valid'] & (tabdata['version'] == 4))
                          .to_numpy(), rundata.shape)
    mask = rundata >= 3
    run_str = {1: "first", 2: "second"}
    cabledata = np.load(f'../resources/ellyandcable-{run_str[run]}run.npz')
    cosdata = cabledata['cos']
    rundata[~mask] = np.nan
    f, ax = plot_snr_rx_map(rundata, gooddata, cosdata[slc],
                            first_channel=slc.start, marker_aspect_ratio=1,
                            give_handles=True
                            )
    ax.axvspan(180, 239, color='C3', zorder=0.5, alpha=0.5)
    ax.axvspan(0, 47, color='C3', zorder=0.5, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
