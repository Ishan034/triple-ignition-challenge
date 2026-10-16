"""Static report figure: reference band vs observed trajectory, with alerts.

Small multiples (one panel per variable, each with its own y axis and a shared time
axis), so different units are never forced onto one scale.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alerts import Alert
from .reference import ReferenceProfile
from .schema import VARIABLES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#e4e3df"
OBSERVED = "#2a78d6"
ALERT = "#ec835a"  # status "serious": used only for alert markers


def _break_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN where readings are missing so the line is not drawn across the gap."""
    if len(x) < 3:
        return x, y
    step = np.median(np.diff(x))
    gaps = np.where(np.diff(x) > 1.5 * step)[0]
    return np.insert(x.astype(float), gaps + 1, np.nan), np.insert(
        y.astype(float), gaps + 1, np.nan
    )


def plot_batch(
    scores: pd.DataFrame,
    alerts: list[Alert],
    reference: ReferenceProfile,
    batch_id: str,
    path,
    band_z: float = 3.0,
    markers: dict[str, float] | None = None,
    title_note: str = "",
) -> None:
    """Save a figure for one batch. ``markers`` maps a label to an elapsed hour."""
    s = scores[scores["batch_id"] == batch_id]
    batch_alerts = [a for a in alerts if a.batch_id == batch_id]
    variables = list(VARIABLES)
    fig, axes = plt.subplots(len(variables), 1, figsize=(8, 2.0 * len(variables)), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, var in zip(axes, variables):
        ax.set_facecolor(SURFACE)
        ref = reference.stats[reference.stats["variable"] == var].sort_values("hour")
        ax.fill_between(
            ref["hour"],
            ref["median"] - band_z * ref["scale"],
            ref["median"] + band_z * ref["scale"],
            color=BAND,
            linewidth=0,
            label=f"Reference median ±{band_z:g} robust SD",
        )
        ax.plot(
            ref["hour"],
            ref["median"],
            color=INK_2,
            linewidth=1,
            linestyle=(0, (4, 3)),
            label="Reference median",
        )
        g = s[s["variable"] == var]
        x, y = _break_gaps(g["elapsed_h"].to_numpy(), g["value"].to_numpy())
        ax.plot(
            x,
            y,
            color=OBSERVED,
            linewidth=2,
            label="Observed",
            marker="o" if len(g) < 20 else None,
            markersize=4,
        )

        flagged_h = sorted(
            {
                h
                for a in batch_alerts
                if a.variable == var
                for h in g["elapsed_h"]
                if a.start_h <= h <= a.end_h
            }
        )
        if flagged_h:
            pts = g[g["elapsed_h"].isin(flagged_h)]
            ax.scatter(
                pts["elapsed_h"],
                pts["value"],
                s=40,
                color=ALERT,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
                label="Alert",
            )
        for label, hour in (markers or {}).items():
            if hour is not None and np.isfinite(hour):
                ax.axvline(hour, color=INK_2, linewidth=1, linestyle=":")
        ax.set_ylabel(f"{var}\n({VARIABLES[var].unit})", color=INK_2, fontsize=8)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    top = axes[0]
    ymax = top.get_ylim()[1]
    for label, hour in (markers or {}).items():
        if hour is not None and np.isfinite(hour):
            top.text(hour, ymax, f" {label}", color=INK_2, fontsize=7, va="top", ha="left")
    axes[-1].set_xlabel("Elapsed time (h)", color=INK_2, fontsize=8)
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"Batch {batch_id}: observed vs reference{title_note}", color=INK, fontsize=11, y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
