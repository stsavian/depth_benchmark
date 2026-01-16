#!/usr/bin/env python3
"""
Simple investigation plots for depth estimation error vs ground truth depth.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot error vs depth relationships")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/moge_skyscenes/results_per_sample.csv",
        help="Path to per-sample results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/moge_skyscenes/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Plot 1: RMSE vs GT Depth (scatter)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["gt_depth_mean"], df["rmse"], alpha=0.5, s=20)
    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("RMSE (meters)")
    ax.set_title("RMSE vs Ground Truth Depth")
    ax.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(df["gt_depth_mean"], df["rmse"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["gt_depth_mean"].min(), df["gt_depth_mean"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", label=f"Linear fit: y={z[0]:.3f}x + {z[1]:.3f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_vs_depth_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: rmse_vs_depth_scatter.png")

    # =========================================================================
    # Plot 2: Relative Error (RMSE / GT_depth) vs GT Depth
    # =========================================================================
    df["rmse_relative"] = df["rmse"] / df["gt_depth_mean"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["gt_depth_mean"], df["rmse_relative"], alpha=0.5, s=20)
    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("Relative RMSE (RMSE / GT_depth)")
    ax.set_title("Relative Error vs Ground Truth Depth")
    ax.grid(True, alpha=0.3)

    # Add horizontal line at mean
    mean_rel = df["rmse_relative"].mean()
    ax.axhline(y=mean_rel, color="r", linestyle="--", label=f"Mean: {mean_rel:.3f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_relative_vs_depth.png", dpi=150)
    plt.close()
    print(f"Saved: rmse_relative_vs_depth.png")

    # =========================================================================
    # Plot 3: RMSE vs GT Depth colored by altitude
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    altitudes = sorted(df["altitude"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(altitudes)))

    for alt, color in zip(altitudes, colors):
        mask = df["altitude"] == alt
        ax.scatter(
            df.loc[mask, "gt_depth_mean"],
            df.loc[mask, "rmse"],
            alpha=0.6,
            s=20,
            c=[color],
            label=f"H={alt}m",
        )

    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("RMSE (meters)")
    ax.set_title("RMSE vs Ground Truth Depth (by Altitude)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_vs_depth_by_altitude.png", dpi=150)
    plt.close()
    print(f"Saved: rmse_vs_depth_by_altitude.png")

    # =========================================================================
    # Plot 4: RMSE vs GT Depth colored by pitch
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    pitches = sorted(df["pitch"].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(pitches)))

    for pitch, color in zip(pitches, colors):
        mask = df["pitch"] == pitch
        ax.scatter(
            df.loc[mask, "gt_depth_mean"],
            df.loc[mask, "rmse"],
            alpha=0.6,
            s=20,
            c=[color],
            label=f"P={pitch}°",
        )

    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("RMSE (meters)")
    ax.set_title("RMSE vs Ground Truth Depth (by Pitch)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_vs_depth_by_pitch.png", dpi=150)
    plt.close()
    print(f"Saved: rmse_vs_depth_by_pitch.png")

    # =========================================================================
    # Plot 5: Binned RMSE vs GT Depth (to see trend more clearly)
    # =========================================================================
    # Create depth bins
    df["depth_bin"] = pd.cut(df["gt_depth_mean"], bins=10)
    binned = df.groupby("depth_bin", observed=True).agg({
        "rmse": ["mean", "std", "count"],
        "gt_depth_mean": "mean",
    })
    binned.columns = ["rmse_mean", "rmse_std", "count", "depth_mean"]
    binned = binned.reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        binned["depth_mean"],
        binned["rmse_mean"],
        yerr=binned["rmse_std"],
        fmt="o-",
        capsize=5,
        markersize=8,
    )
    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("RMSE (meters)")
    ax.set_title("Binned RMSE vs Ground Truth Depth (mean ± std)")
    ax.grid(True, alpha=0.3)

    # Add count annotations
    for _, row in binned.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (row["depth_mean"], row["rmse_mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_vs_depth_binned.png", dpi=150)
    plt.close()
    print(f"Saved: rmse_vs_depth_binned.png")

    # =========================================================================
    # Plot 6: AbsRel vs GT Depth (common metric)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["gt_depth_mean"], df["abs_rel"], alpha=0.5, s=20)
    ax.set_xlabel("Ground Truth Depth (mean, meters)")
    ax.set_ylabel("Absolute Relative Error")
    ax.set_title("AbsRel vs Ground Truth Depth")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "absrel_vs_depth_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: absrel_vs_depth_scatter.png")

    # =========================================================================
    # Summary stats
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Correlation between error and depth
    corr_rmse = df["gt_depth_mean"].corr(df["rmse"])
    corr_absrel = df["gt_depth_mean"].corr(df["abs_rel"])
    corr_rmse_rel = df["gt_depth_mean"].corr(df["rmse_relative"])

    print(f"Correlation (GT_depth, RMSE):          {corr_rmse:.4f}")
    print(f"Correlation (GT_depth, AbsRel):        {corr_absrel:.4f}")
    print(f"Correlation (GT_depth, RMSE/GT_depth): {corr_rmse_rel:.4f}")

    # Linear fit coefficients
    print(f"\nLinear fit RMSE = a*depth + b:")
    print(f"  a = {z[0]:.4f} (RMSE increases by {z[0]:.4f}m per 1m depth)")
    print(f"  b = {z[1]:.4f}")

    # Per-altitude stats
    print("\nPer-altitude RMSE:")
    for alt in altitudes:
        mask = df["altitude"] == alt
        print(f"  H={alt}m: RMSE={df.loc[mask, 'rmse'].mean():.2f}±{df.loc[mask, 'rmse'].std():.2f}")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
