import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = r"D:\Singapore Data\Cable\inversion_observations.csv"

# Columns used for prior consistency checks
PRIOR_COLUMNS = [
    "prior_x_m",
    "prior_y_m",
    "prior_u_m",
    "prior_lat",
    "prior_lon",
    "prior_z_m",
]

# Columns used for plotting in east-north plane
PLOT_X_COL = "prior_x_m"   # East
PLOT_Y_COL = "prior_y_m"   # North

CHANNEL_COL = "channel"


def load_table(file_path: str) -> pd.DataFrame:
    """
    Load the observations file.
    Tries normal CSV first, then falls back to whitespace-delimited.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Try normal CSV first
    try:
        df = pd.read_csv(file_path)
        if CHANNEL_COL in df.columns:
            return df
    except Exception:
        pass

    # Fallback: whitespace-delimited
    try:
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        if CHANNEL_COL in df.columns:
            return df
    except Exception:
        pass

    raise ValueError(
        "Could not parse the file as either a normal CSV or whitespace-delimited table."
    )


def validate_columns(df: pd.DataFrame):
    required = [CHANNEL_COL, PLOT_X_COL, PLOT_Y_COL] + PRIOR_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarize_channel_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each channel, count how many unique values it has in each prior column.
    If every prior column has exactly 1 unique value, that channel is consistent.
    """
    summary_rows = []

    for channel, g in df.groupby(CHANNEL_COL, dropna=False):
        row = {
            "channel": channel,
            "n_rows": len(g),
        }

        for col in PRIOR_COLUMNS:
            # dropna=False equivalent with pandas nunique(dropna=False)
            row[f"{col}_nunique"] = g[col].nunique(dropna=False)

        # Main criterion: exactly one unique value per prior field
        row["prior_consistent"] = all(row[f"{col}_nunique"] == 1 for col in PRIOR_COLUMNS)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("channel").reset_index(drop=True)
    return summary


def print_consistency_report(df: pd.DataFrame, summary: pd.DataFrame):
    total_channels = len(summary)
    consistent_channels = summary["prior_consistent"].sum()
    inconsistent_channels = total_channels - consistent_channels

    print("\n=== PRIOR CONSISTENCY REPORT ===")
    print(f"Total rows: {len(df)}")
    print(f"Total channels: {total_channels}")
    print(f"Channels with exactly one prior: {consistent_channels}")
    print(f"Channels with inconsistent prior values: {inconsistent_channels}")

    # Explicit check for x/y consistency per channel
    xy_bad = summary[
        (summary["prior_x_m_nunique"] > 1) | (summary["prior_y_m_nunique"] > 1)
    ]
    print(f"\nChannels where prior_x_m or prior_y_m changes: {len(xy_bad)}")

    if len(xy_bad) > 0:
        print("\n--- Channels with varying prior_x_m / prior_y_m ---")
        print(
            xy_bad[
                ["channel", "n_rows", "prior_x_m_nunique", "prior_y_m_nunique"]
            ].to_string(index=False)
        )

    bad = summary[~summary["prior_consistent"]]
    if len(bad) > 0:
        print("\n--- Detailed inconsistent channels ---")
        for channel in bad["channel"]:
            g = df[df[CHANNEL_COL] == channel].copy()
            print(f"\nChannel {channel}:")
            print(f"Rows: {len(g)}")

            for col in PRIOR_COLUMNS:
                unique_vals = pd.unique(g[col])
                print(f"  {col}: {len(unique_vals)} unique value(s) -> {unique_vals[:10]}")
    else:
        print("\nAll channels have exactly one unique prior.")


def build_unique_channel_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one point per channel for plotting.
    If a channel has inconsistent priors, we still keep one representative point
    (first row) so the plot can be inspected visually, while the text report flags it.
    """
    rows = []

    for channel, g in df.groupby(CHANNEL_COL, dropna=False):
        first = g.iloc[0]
        rows.append(
            {
                "channel": channel,
                "prior_x_m": first[PLOT_X_COL],
                "prior_y_m": first[PLOT_Y_COL],
                "n_rows": len(g),
                "x_nunique": g[PLOT_X_COL].nunique(dropna=False),
                "y_nunique": g[PLOT_Y_COL].nunique(dropna=False),
                "xy_consistent": (
                    g[PLOT_X_COL].nunique(dropna=False) == 1
                    and g[PLOT_Y_COL].nunique(dropna=False) == 1
                ),
            }
        )

    points = pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)
    return points


def plot_channel_priors(points: pd.DataFrame):
    """
    Plot one prior point per channel in the east-north plane.
    """
    plt.figure(figsize=(10, 8))

    consistent = points[points["xy_consistent"]]
    inconsistent = points[~points["xy_consistent"]]

    if len(consistent) > 0:
        plt.scatter(
            consistent["prior_x_m"],
            consistent["prior_y_m"],
            s=50,
            label="Consistent channel prior"
        )

    if len(inconsistent) > 0:
        plt.scatter(
            inconsistent["prior_x_m"],
            inconsistent["prior_y_m"],
            s=80,
            marker="x",
            label="Inconsistent channel prior"
        )

    # Label each point with channel number
    for _, row in points.iterrows():
        plt.annotate(
            str(int(row["channel"])) if pd.notna(row["channel"]) else "NaN",
            (row["prior_x_m"], row["prior_y_m"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8
        )

    plt.xlabel("East (prior_x_m)")
    plt.ylabel("North (prior_y_m)")
    plt.title("Channel priors in east-north plane")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print(f"Reading: {FILE_PATH}")
    df = load_table(FILE_PATH)
    validate_columns(df)

    # Clean obvious numeric columns in case they were read as strings
    numeric_cols = [CHANNEL_COL, PLOT_X_COL, PLOT_Y_COL] + PRIOR_COLUMNS
    numeric_cols = list(dict.fromkeys(numeric_cols))  # remove duplicates
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = summarize_channel_priors(df)
    print_consistency_report(df, summary)

    # Optional: specifically inspect channel 348
    ch = 348
    if (df[CHANNEL_COL] == ch).any():
        g348 = df[df[CHANNEL_COL] == ch]
        print(f"\n=== CHANNEL {ch} CHECK ===")
        print(f"Rows for channel {ch}: {len(g348)}")
        print(f"Unique prior_x_m: {pd.unique(g348['prior_x_m'])}")
        print(f"Unique prior_y_m: {pd.unique(g348['prior_y_m'])}")
        print(f"Unique prior_u_m: {pd.unique(g348['prior_u_m'])}")
    else:
        print(f"\nChannel {ch} was not found in the file.")

    points = build_unique_channel_points(df)
    plot_channel_priors(points)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)