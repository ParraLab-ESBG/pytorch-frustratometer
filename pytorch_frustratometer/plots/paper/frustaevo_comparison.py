"""
Compare FrustraEvo IC outputs between a re-implementation run and originals.

New layout:   {new_dir}/{funfam}/FrustraEvo_{funfam}/OutPutFiles/
Original:     {orig_dir}/FrustraEvo_{funfam}/OutPutFiles/

Files compared per funfam:
  IC_SingleRes_{funfam}  -> ICTot column  (frustration IC from predicted classes)

SeqIC is identical in both runs (same MSA) so is not compared.

Produces a 1×3 scatter figure:
  [IC_SingleRes: new vs orig per position] [Timing: new vs orig] [Time vs MSA length]

Usage
-----
python -m frustrai_fold.plots.frustaevo_comparison \\
    --new_dir  .../FrustrAI-Fold/baseline_comparison/FrustaEvo \\
    --orig_dir .../seqdist3 \\
    --out      frustaevo_ic_comparison.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── metrics ───────────────────────────────────────────────────────────────────

def pearson_r2(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    return float(r ** 2)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float("nan") if denom == 0 else float(np.dot(a, b) / denom)


# ── loaders ───────────────────────────────────────────────────────────────────

def _load(path: Path, sep: str, col: str) -> "pd.Series | None":
    try:
        df = pd.read_csv(path, sep=sep)
        if "Res" in df.columns:
            return df.set_index("Res")[col].astype(float)
        return pd.Series(df[col].to_numpy(dtype=float))
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def load_time(path: Path) -> float | None:
    try:
        return float(path.read_text().strip().split()[0])
    except Exception:
        return None


FILE_SPECS = [
    # (label,          filename_template,    sep,   column)
    ("IC_SingleRes", "IC_SingleRes_{fid}",  "\t",  "ICTot"),
]


# ── per-funfam comparison ─────────────────────────────────────────────────────

def compare_funfam(funfam: str, new_out: Path, orig_out: Path) -> tuple[list[dict], list[dict]]:
    """Return (summary_rows, scatter_rows).

    summary_rows: one row per (funfam, metric) with R² / cosine_sim.
    scatter_rows: one row per position per metric with new_val / orig_val / funfam.
    """
    summary, scatter = [], []
    for label, tmpl, sep, col in FILE_SPECS:
        fname     = tmpl.format(fid=funfam)
        new_path  = new_out  / fname
        orig_path = orig_out / fname

        missing = [s for s, p in (("new", new_path), ("orig", orig_path)) if not p.exists()]
        if missing:
            print(f"  [SKIP] {label}: missing {', '.join(missing)}")
            continue

        new_v  = _load(new_path,  sep, col)
        orig_v = _load(orig_path, sep, col)
        if new_v is None or orig_v is None:
            continue

        if len(new_v) != len(orig_v):
            # Positional comparison is only valid when lengths match exactly.
            # Even a 1-residue difference shifts all subsequent positions, giving
            # near-zero R² (same issue documented in data/io.py lines 127-129).
            # There is no shared absolute coordinate (new uses AlphaFold TED domains,
            # original may use different reference proteins), so skip rather than
            # report a misleading metric.
            print(f"  [SKIP] {label} length mismatch new={len(new_v)} orig={len(orig_v)} "
                  f"— no shared position coordinate, cannot align")
            summary.append({"funfam": funfam, "metric": label,
                            "n_positions": None, "pearson_r2": float("nan"),
                            "cosine_sim": float("nan"),
                            "note": f"length_mismatch_{len(new_v)}_vs_{len(orig_v)}"})
            continue
        new_v  = new_v.to_numpy()
        orig_v = orig_v.to_numpy()
        n = len(new_v)

        r2  = pearson_r2(new_v, orig_v)
        cos = cosine_sim(new_v, orig_v)
        print(f"  {label:15s}  n={n:4d}  R²={r2:.4f}  cos={cos:.4f}")

        summary.append({"funfam": funfam, "metric": label,
                        "n_positions": n, "pearson_r2": r2, "cosine_sim": cos})

        ok = np.isfinite(new_v) & np.isfinite(orig_v)
        for nv, ov in zip(new_v[ok], orig_v[ok]):
            scatter.append({"funfam": funfam, "metric": label,
                            "new_val": float(nv), "orig_val": float(ov)})

    return summary, scatter


# ── discovery ─────────────────────────────────────────────────────────────────

def find_funfams(new_dir: Path):
    for entry in sorted(new_dir.iterdir()):
        if not entry.is_dir():
            continue
        fid = entry.name
        out = entry / f"FrustraEvo_{fid}" / "OutPutFiles"
        if out.is_dir():
            yield fid, out


def collect_timings(funfams: list, new_dir: Path, orig_dir: Path,
                    size_map: dict) -> pd.DataFrame:
    rows = []
    for funfam, new_out in funfams:
        new_time  = load_time(new_out.parent / "Time.log")
        orig_time = load_time(orig_dir / f"FrustraEvo_{funfam}" / "Time.log")
        if new_time is None and orig_time is None:
            continue
        rows.append({"funfam": funfam, "new_time": new_time, "orig_time": orig_time,
                     "n_positions": size_map.get(funfam)})
    return pd.DataFrame(rows)


# ── scatter helper ────────────────────────────────────────────────────────────

def _scatter_with_ref(ax, x: np.ndarray, y: np.ndarray,
                      hue: np.ndarray | None, xlabel: str, ylabel: str,
                      title: str, palette="tab10") -> None:
    """Scatter with y=x diagonal and R² annotation."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    hue_ok = hue[ok] if hue is not None else None

    df_plot = pd.DataFrame({"x": x, "y": y})
    if hue_ok is not None:
        df_plot["hue"] = hue_ok
        sns.scatterplot(data=df_plot, x="x", y="y", hue="hue",
                        palette=palette, alpha=0.5, s=15, ax=ax,
                        legend=False)
    else:
        sns.scatterplot(data=df_plot, x="x", y="y",
                        alpha=0.5, s=15, ax=ax)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=3)

    r2 = pearson_r2(x, y)
    cos = cosine_sim(x, y)
    ax.set_title(f"{title}\nR²={r2:.3f}  cos={cos:.3f}", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_all(scatter_df: pd.DataFrame, timing_df: pd.DataFrame,
             out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    c_new  = sns.color_palette("tab10")[0]
    c_orig = sns.color_palette("tab10")[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── panel 0: IC_SingleRes new vs orig per position ───────────────────────
    ax  = axes[0]
    sub = scatter_df[scatter_df["metric"] == "IC_SingleRes"] if not scatter_df.empty else pd.DataFrame()
    if not sub.empty:
        funfam_ids = sub["funfam"].astype("category").cat.codes.to_numpy()
        _scatter_with_ref(
            ax,
            x=sub["orig_val"].to_numpy(),
            y=sub["new_val"].to_numpy(),
            hue=funfam_ids,
            xlabel="Original ICTot",
            ylabel="Re-implementation ICTot",
            title="Frustration IC (per position)",
            palette="tab10",
        )
    else:
        ax.text(0.5, 0.5, "No IC_SingleRes data", ha="center", va="center",
                transform=ax.transAxes)
        sns.despine(ax=ax)

    # ── panel 1: new vs orig timing ───────────────────────────────────────────
    ax = axes[1]
    if not timing_df.empty:
        both = timing_df.dropna(subset=["new_time", "orig_time"])
        if not both.empty:
            n_pos = both["n_positions"].fillna(0).to_numpy(dtype=float)
            _scatter_with_ref(
                ax,
                x=both["orig_time"].to_numpy(),
                y=both["new_time"].to_numpy(),
                hue=n_pos,
                xlabel="Original time (s)",
                ylabel="Re-implementation time (s)",
                title="Timing per funfam",
                palette="viridis",
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            speedup = (both["orig_time"] / both["new_time"]).median()
            ax.set_title(ax.get_title() + f"\nmedian speedup {speedup:.1f}×", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No paired timing data", ha="center", va="center",
                    transform=ax.transAxes)
            sns.despine(ax=ax)
    else:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes)
        sns.despine(ax=ax)

    # ── panel 2: time vs MSA length ───────────────────────────────────────────
    ax = axes[2]
    if not timing_df.empty:
        has_size = timing_df.dropna(subset=["n_positions"])
        sns.scatterplot(data=has_size.dropna(subset=["new_time"]),
                        x="n_positions", y="new_time",
                        color=c_new, label="re-implementation",
                        alpha=0.6, s=30, ax=ax)
        sns.scatterplot(data=has_size.dropna(subset=["orig_time"]),
                        x="n_positions", y="orig_time",
                        color=c_orig, label="original",
                        alpha=0.6, s=30, marker="^", ax=ax)
    ax.set_xlabel("MSA length (positions)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Scaling: time vs MSA length", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    sns.despine(ax=ax)

    plt.suptitle("FrustraEvo re-implementation vs original", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "frustaevo_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new_dir",  required=True,
                   help="Dir with per-funfam subdirs from the re-implementation run")
    p.add_argument("--orig_dir", required=True,
                   help="Dir containing original FrustraEvo_* runs")
    p.add_argument("--out", default="frustaevo_ic_comparison.csv",
                   help="Output CSV (default: frustaevo_ic_comparison.csv)")
    p.add_argument("--plot_dir", default=None,
                   help="Directory for summary PNG (default: same dir as --out)")
    args = p.parse_args()

    new_dir  = Path(args.new_dir)
    orig_dir = Path(args.orig_dir)

    funfams = list(find_funfams(new_dir))
    print(f"Found {len(funfams)} completed funfams in {new_dir}\n")

    all_summary, all_scatter = [], []
    for funfam, new_out in funfams:
        orig_out = orig_dir / f"FrustraEvo_{funfam}" / "OutPutFiles"
        if not orig_out.is_dir():
            print(f"[{funfam}] original OutPutFiles not found, skipping")
            continue
        print(f"[{funfam}]")
        s, sc = compare_funfam(funfam, new_out, orig_out)
        all_summary.extend(s)
        all_scatter.extend(sc)

    if not all_summary:
        print("No results — check paths.")
        return

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(summary_df)} rows → {args.out}")

    print("\nIC summary (mean / median across funfams):")
    print(summary_df.groupby("metric")[["pearson_r2", "cosine_sim"]]
          .agg(["mean", "median"]).to_string())

    size_map = (summary_df[summary_df["metric"] == "IC_SingleRes"]
                .set_index("funfam")["n_positions"].to_dict())
    timing_df = collect_timings(funfams, new_dir, orig_dir, size_map)
    if not timing_df.empty:
        timing_out = Path(args.out).with_name(Path(args.out).stem + "_timing.csv")
        timing_df.to_csv(timing_out, index=False)
        print(f"Wrote {len(timing_df)} timing rows → {timing_out}")
        both = timing_df.dropna(subset=["new_time", "orig_time"])
        if not both.empty:
            print(f"Median speedup (orig/new): {(both['orig_time']/both['new_time']).median():.2f}×")

    plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.out).parent
    plot_all(pd.DataFrame(all_scatter), timing_df, plot_dir)


if __name__ == "__main__":
    main()
