import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def arrival_info_box(ax, starts, amps=None, title="Arrivals"):
    lines = [title]
    if amps is None:
        for s in starts:
            lines.append(f"• {s}")
    else:
        for s, a in zip(starts, amps):
            lines.append(f"• {s}  (A={a:g})")

    text = "\n".join(lines)

    ax.text(
        0.99, 0.99, text,
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
    h = template[::-1]
    return np.convolve(x, h, mode="valid")

def add_pulse(x, start, width, amp=1.0):
    start = int(start)
    end = int(start + width)
    if end <= 0 or start >= len(x):
        return
    start_clip = max(0, start)
    end_clip = min(len(x), end)
    x[start_clip:end_clip] += amp

def style_axes(ax, x_minor=2, y_minor=2):
    """Report-style grids: major + minor."""
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.18)

def annotate_arrivals_on_signal(ax, starts, width, amps=None,
                                show_spans=True,
                                arrival_ls="--",
                                span_alpha=0.10,
                                text_y=0.97):
    """
    Mark pulse start arrivals and optionally shade each pulse duration.
    """
    if amps is None:
        amps = [None] * len(starts)

    # Use axis-fraction coordinates for consistent label placement.
    for s, a in zip(starts, amps):
        # Arrival line (full height)
        ax.axvline(s, linestyle=arrival_ls, linewidth=1.0, alpha=0.9)

        # Pulse duration span
        if show_spans:
            ax.axvspan(s, s + width, alpha=span_alpha)

        # Label near top
        lab = f"arrival @ {s}"
        if a is not None:
            lab += f"\nA={a:g}"



def annotate_overlap_stipple_inside_pulse(ax, base, delays, width,
                                         ymin_frac=0.10, ymax_frac=0.90):
    """
    For overlap: draw dotted 'stipple' lines *within the pulse window*.
    We restrict line height using axis-fraction so it looks like it's inside the pulse band.
    """
    # A lightly shaded "common window" to emphasize the overlap area (optional but helpful)
    ax.axvspan(base, base + width, alpha=0.08)

    for d in delays:
        t = base + d
        # axvline doesn't accept ymin/ymax in data coords; it uses axis-fraction.
        ax.axvline(t, linestyle=":", linewidth=1.4, alpha=0.95,
                   ymin=ymin_frac, ymax=ymax_frac)
        ax.text(t, 0.02, f"+{d}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)

def annotate_matched_filter(ax, y, expected_peaks=None, label_prefix=""):
    """
    Mark actual MF peak + optionally draw expected peak locations as dotted lines.
    """
    k = int(np.argmax(y))
    yk = float(y[k])

    # Actual peak: solid line + marker
    ax.axvline(k, linestyle="-", linewidth=1.2, alpha=0.9)
    ax.plot([k], [yk], marker="o", markersize=5)

    ax.annotate(
        f"peak @ {k}\ny={yk:.3f}",
        xy=(k, yk),
        xycoords="data",
        xytext=(8, 20),
        textcoords="offset points",
        ha="left", va="bottom", fontsize=8,
        arrowprops=dict(arrowstyle="-", linewidth=0.8)
    )


    # Expected peak locations (reference)
    if expected_peaks is not None:
        for e in expected_peaks:
            ax.axvline(e, linestyle=":", linewidth=1.1, alpha=0.7)
        ax.text(0.01, 0.03,
                "dotted = expected (from known arrivals)\nsolid = actual max",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, alpha=0.9)

def main():
    rng = np.random.default_rng(0)

    # ----- Setup -----
    N = 800
    pulse_w = 60
    template = np.ones(pulse_w)
    noise_sigma = 0.05

    # ===== Case 1: single pulse =====
    x1 = rng.normal(0, noise_sigma, size=N)
    starts1 = [300]
    amps1 = [1.0]
    add_pulse(x1, start=starts1[0], width=pulse_w, amp=amps1[0])
    y1 = matched_filter(x1, template)

    # ===== Case 2: well-separated pulse train =====
    x2 = rng.normal(0, noise_sigma, size=N)
    starts2 = [150, 350, 550]
    amps2 = [1.0, 0.8, 0.3]
    for s, a in zip(starts2, amps2):
        add_pulse(x2, start=s, width=pulse_w, amp=a)
    y2 = matched_filter(x2, template)

    # ===== Case 3: overlapping pulses (different amps) =====
    x3 = rng.normal(0, noise_sigma, size=N)
    base = 320
    delays = [0, 37, 76]
    amps3 = [0.9, 0.8, 0.9]
    starts3 = [base + d for d in delays]
    for s, a in zip(starts3, amps3):
        add_pulse(x3, start=s, width=pulse_w, amp=a)
    y3 = matched_filter(x3, template)
    

    # ===== Case 4: overlapping pulses (different amps) =====
    x4 = rng.normal(0, noise_sigma, size=N)
    base = 320
    delays = [0, 37, 76]
    amps4 = [0.9, 0.3, 0.3]
    starts4 = [base + d for d in delays]
    for s, a in zip(starts4, amps4):
        add_pulse(x4, start=s, width=pulse_w, amp=a)
    y4 = matched_filter(x4, template)

    # Expected peak indices in "valid" MF output:
    # y[k] corresponds to x[k : k+pulse_w], so a perfect square pulse starting at s peaks at k=s.
    def expected_in_valid(starts):
        return [s for s in starts if 0 <= s <= (N - pulse_w)]

    # ----- Plot (2 rows x 3 cols) -----
    fig, axes = plt.subplots(2, 4, figsize=(13, 8), sharex="col")

    cases = [
        ("Case 1: single pulse", x1, y1, starts1, amps1, None),
        ("Case 2: separated pulses", x2, y2, starts2, amps2, None),
        ("Case 3: overlapping pulses", x3, y3, starts3, amps3, (base, delays)),
        ("Case 4: overlapping pulses (diff. amps.)", x4, y4, starts4, amps4, (base, delays)),
    ]

    for c, (title, x, y, starts, amps, overlap_info) in enumerate(cases):
        ax_sig = axes[0, c]
        ax_mf  = axes[1, c]

        # --- Signal plot ---
        ax_sig.plot(x, linewidth=1.0)
        ax_sig.set_title(title)
        ax_sig.set_ylabel("amplitude" if c == 0 else "")
        style_axes(ax_sig)

        arrival_info_box(ax_sig, starts, amps)

        # Overlap-specific stippled lines inside the (shared) pulse window
        if overlap_info is not None:
            base0, delays0 = overlap_info
            annotate_overlap_stipple_inside_pulse(ax_sig, base0, delays0, pulse_w)

        # --- MF plot ---
        ax_mf.plot(y, linewidth=1.0)
        ax_mf.set_title("Matched filter output")
        ax_mf.set_ylabel("corr" if c == 0 else "")
        style_axes(ax_mf)

        peak_info_box(ax_mf, y)


        ax_mf.set_xlabel("sample index (valid correlation)")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
