import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator



'''

ignore this for now...

'''










def arrival_info_box(ax, starts, amps=None, title="Arrivals"):
    lines = [title]
    if amps is None:
        for s in starts:
            lines.append(f"• {s}")
    else:
        for s, a in zip(starts, amps):
            lines.append(f"• {s}  (A={a:g})")

    ax.text(
        0.99, 0.99, "\n".join(lines),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.85,
                  edgecolor="none")
    )


def peak_info_box(ax, y):
    k = int(np.argmax(y))
    yk = float(y[k])

    text = (
        "Matched-filter max peak\n"
        f"• index = {k}\n"
        f"• value = {yk:.3f}"
    )

    ax.text(
        0.99, 0.99, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.85,
                  edgecolor="none")
    )


def matched_filter(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.convolve(x, template[::-1], mode="valid")


def add_pulse(x, start, width, amp=1.0):
    start = int(start)
    end = int(start + width)
    if end <= 0 or start >= len(x):
        return
    x[max(0, start):min(len(x), end)] += amp


def style_axes(ax, x_minor=2, y_minor=2):
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.18)


def annotate_overlap_stipple_inside_pulse(ax, base, delays, width,
                                         ymin_frac=0.10, ymax_frac=0.90):
    ax.axvspan(base, base + width, alpha=0.08)
    for d in delays:
        t = base + d
        ax.axvline(t, linestyle=":", linewidth=1.4, alpha=0.95,
                   ymin=ymin_frac, ymax=ymax_frac)
        ax.text(t, 0.02, f"+{d}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)


def main():
    rng = np.random.default_rng(0)

    N = 800
    pulse_w = 52
    template = np.ones(pulse_w)
    noise_sigma = 0.05

    # same overlap offsets for ALL cases
    base = 320
    delays = [0, 15, 20]
    starts = [base + d for d in delays]

    cases = [
        ("Case 1: overlapping (equal amps)", [0.8, 0.8, 0.8]),
        ("Case 2: overlapping (arrival 1 much stronger)", [1.2, 0.25, 0.25]),
        ("Case 3: overlapping (last arrival much stronger)", [0.25, 0.25, 1.2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex="col")

    for c, (title, amps) in enumerate(cases):
        x = rng.normal(0, noise_sigma, size=N)
        for s, a in zip(starts, amps):
            add_pulse(x, start=s, width=pulse_w, amp=a)
        y = matched_filter(x, template)

        ax_sig = axes[0, c]
        ax_mf  = axes[1, c]

        ax_sig.plot(x, linewidth=1.0)
        ax_sig.set_title(title)
        ax_sig.set_ylabel("amplitude" if c == 0 else "")
        style_axes(ax_sig)

        arrival_info_box(ax_sig, starts, amps)
        annotate_overlap_stipple_inside_pulse(ax_sig, base, delays, pulse_w)

        ax_mf.plot(y, linewidth=1.0)
        ax_mf.set_title("Matched filter output")
        ax_mf.set_ylabel("corr" if c == 0 else "")
        ax_mf.set_xlabel("sample index (valid correlation)")
        style_axes(ax_mf)

        peak_info_box(ax_mf, y)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
