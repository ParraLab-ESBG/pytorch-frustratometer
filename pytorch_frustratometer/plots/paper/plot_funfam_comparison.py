#!/usr/bin/env python
"""
Domain adaptation figure: per-domain cosine similarity, colored by superfamily.

Each point = one CATH domain.
X-axis     = ProteinMPNN (frozen) cosine sim between predicted and true frustration distribution.
Y-axis     = Frame2Seq cosine sim.
Color      = superfamily (C.A.T.H level) — reveals within-group structure.

Metric per domain:
  true_dist  = [n_minimal/N, n_neutral/N, n_frustrated/N]   (observed class fractions)
  pred_dist  = mean(softmax(logits)) over unmasked residues  (mean predicted probability)
  score      = cosine_similarity(true_dist, pred_dist)        in [0, 1]

CSVs from eval_per_domain.py contain (domain, true_class, pred_prob_0, pred_prob_1, pred_prob_2).

Usage:
    python frustrai_fold/plots/plot_funfam_comparison.py \
        --results_dir results/funfam_comparison \
        --test_csv    /gpfs/.../dompdb_v4.4/test.csv \
        --out         results/funfam_comparison/domain_adaptation.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "enc_frame2seq":             "Frame2Seq (finetuned)",
    "enc_protein_mpnn":          "ProteinMPNN (frozen)",
    "enc_protein_mpnn_scratch":  "ProteinMPNN (scratch)",
    "enc_protein_mpnn_finetune": "ProteinMPNN (finetune)",
}

MPNN_COLORS = {
    "enc_protein_mpnn":          "#FF7043",
    "enc_protein_mpnn_scratch":  "#FF9966",
    "enc_protein_mpnn_finetune": "#FFBB88",
}

MARKERS = {
    "enc_protein_mpnn":          "o",
    "enc_protein_mpnn_scratch":  "s",
    "enc_protein_mpnn_finetune": "^",
}

N_CLASSES = 3


def load_runs(results_dir: Path) -> dict[str, pd.DataFrame]:
    runs = {}
    for csv_path in sorted(results_dir.glob("enc_*.csv")):
        df = pd.read_csv(csv_path)
        if not {"domain", "true_class", "pred_prob_0"}.issubset(df.columns):
            print(f"  [skip] {csv_path.stem}: old format — re-run eval_per_domain.py")
            continue
        runs[csv_path.stem] = df
        print(f"  [ok] {csv_path.stem}: {len(df)} residues across {df['domain'].nunique()} domains")
    return runs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def domain_scores(df: pd.DataFrame, min_residues: int = 10) -> pd.Series:
    """Return a Series {domain -> cosine_sim} for one run's residue-level CSV."""
    prob_cols = ["pred_prob_0", "pred_prob_1", "pred_prob_2"]
    scores = {}
    for name, grp in df.groupby("domain"):
        if len(grp) < min_residues:
            continue
        # true distribution
        counts = np.array([(grp["true_class"] == c).sum() for c in range(N_CLASSES)], dtype=float)
        true_dist = counts / counts.sum()
        # predicted distribution = mean softmax over residues
        pred_dist = grp[prob_cols].values.mean(axis=0)
        scores[name] = cosine_sim(true_dist, pred_dist)
    return pd.Series(scores, name="score")


def savefig(fig, path: str, dpi: int = 300):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved → {p}  +  {p.with_suffix('.svg')}")


def sfam_mean_scores(all_scores: dict, dom2sfam: dict,
                     min_domains: int = 5) -> dict[str, dict[str, float]]:
    """Per-sfam mean cosine sim for each run. Only sfams with >= min_domains included."""
    result = {}
    for run_name, scores in all_scores.items():
        groups: dict[str, list] = {}
        for dom, score in scores.items():
            sfam = dom2sfam.get(dom, "unknown")
            if sfam != "unknown" and not np.isnan(score):
                groups.setdefault(sfam, []).append(score)
        result[run_name] = {s: float(np.mean(v))
                            for s, v in groups.items() if len(v) >= min_domains}
    return result


def domain_predicted_dists(runs: dict, min_residues: int = 10) -> dict[str, pd.DataFrame]:
    """Per-domain mean true fraction and mean predicted prob for each run.

    Returns {run_name: DataFrame indexed by domain with columns
             true_0/1/2 (true class fractions) and pred_0/1/2 (mean softmax)}
    """
    prob_cols = ["pred_prob_0", "pred_prob_1", "pred_prob_2"]
    result = {}
    for run_name, df in runs.items():
        rows = []
        for domain, grp in df.groupby("domain"):
            if len(grp) < min_residues:
                continue
            counts   = np.array([(grp["true_class"] == c).sum()
                                  for c in range(N_CLASSES)], dtype=float)
            true_dist = counts / counts.sum()
            pred_dist = grp[prob_cols].values.mean(axis=0)
            rows.append({"domain": domain,
                         "true_0": true_dist[0], "true_1": true_dist[1], "true_2": true_dist[2],
                         "pred_0": pred_dist[0], "pred_1": pred_dist[1], "pred_2": pred_dist[2],
                         "n_res":  len(grp)})
        result[run_name] = pd.DataFrame(rows).set_index("domain") if rows else pd.DataFrame()
    return result


def _draw_mechanism_row(axes_row, col, dom_dists, frame2seq_key, mpnn_key,
                        mpnn_label, mpnn_color, class_idx, class_name, global_prior):
    """Draw one mechanism scatter row: true fraction vs predicted prob for one class."""
    ax = axes_row[col]

    f2s_df  = dom_dists.get(frame2seq_key, pd.DataFrame())
    mpnn_df = dom_dists.get(mpnn_key,      pd.DataFrame())
    if f2s_df.empty or mpnn_df.empty:
        ax.set_visible(False)
        return

    common = f2s_df.index.intersection(mpnn_df.index)
    true_col = f"true_{class_idx}"
    pred_col = f"pred_{class_idx}"

    true_frac  = f2s_df.loc[common, true_col].values   # same for both (same test set)
    f2s_pred   = f2s_df.loc[common, pred_col].values
    mpnn_pred  = mpnn_df.loc[common, pred_col].values

    ax.scatter(true_frac, mpnn_pred, s=6, alpha=0.25, color=mpnn_color,
               linewidths=0, zorder=2, label=mpnn_label)
    ax.scatter(true_frac, f2s_pred,  s=6, alpha=0.25, color="#2196F3",
               linewidths=0, zorder=3, label="Frame2Seq")

    # regression lines
    for ys, color in [(f2s_pred, "#2196F3"), (mpnn_pred, mpnn_color)]:
        m, b = np.polyfit(true_frac, ys, 1)
        xs = np.array([true_frac.min(), true_frac.max()])
        ax.plot(xs, m * xs + b, color=color, lw=1.8, alpha=0.9)

    # reference lines (drawn after scatter so limits are known)
    lo_ref = min(true_frac.min(), f2s_pred.min(), mpnn_pred.min())
    hi_ref = max(true_frac.max(), f2s_pred.max(), mpnn_pred.max())
    ax.plot([lo_ref, hi_ref], [lo_ref, hi_ref], "k--", lw=1, alpha=0.35)
    ax.axhline(global_prior, color="grey", lw=1.2, ls=":", alpha=0.6,
               label=f"global prior ({global_prior:.2f})")

    r_f2s  = float(np.corrcoef(true_frac, f2s_pred)[0, 1])
    r_mpnn = float(np.corrcoef(true_frac, mpnn_pred)[0, 1])
    short  = mpnn_label.split("\n")[0]
    ax.text(0.04, 0.96,
            f"r(F2S)={r_f2s:.2f}\nr({short})={r_mpnn:.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.legend(fontsize=7, loc="lower right", markerscale=2)
    ax.set_xlabel(f"True fraction {class_name} per domain", fontsize=9)
    ax.set_ylabel(f"Mean predicted prob({class_name})\nper domain", fontsize=9)
    ax.set_title(f"Mechanism — {class_name}\nF2S vs {mpnn_label.replace(chr(10),' ')}", fontsize=9)

    # data-driven axis limits with 5% margin so diagonal is meaningful
    all_x = np.concatenate([true_frac])
    all_y = np.concatenate([f2s_pred, mpnn_pred])
    lo = min(all_x.min(), all_y.min())
    hi = max(all_x.max(), all_y.max())
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def plot_sfam_analysis(all_scores: dict, runs: dict, dom2sfam: dict,
                       frame2seq_key: str, mpnn_keys: list[str],
                       out: str, min_domains: int = 5, min_residues: int = 10):
    """Three-row figure:
      Row 0 — per-sfam cosine sim scatter (F2S vs ProteinMPNN), global aggregate marked.
      Row 1 — per-domain mechanism: true fraction class 0 (highly frustrated) vs predicted.
      Row 2 — per-domain mechanism: true fraction class 2 (minimally frustrated) vs predicted.

    Rows 1-2 use per-domain data so sfam-level aggregation does not wash out domain variance.
    A positive slope for F2S and flat ProteinMPNN confirms F2S adapts to domain frustration.
    """
    sfam_scores = sfam_mean_scores(all_scores, dom2sfam, min_domains=min_domains)
    dom_dists   = domain_predicted_dists(runs, min_residues=min_residues)

    # global prior from the frame2seq run (true labels are the same across all runs)
    tc = runs[frame2seq_key]["true_class"]
    global_prior = [(tc == c).sum() / len(tc) for c in range(N_CLASSES)]

    n_variants = len(mpnn_keys)
    fig, axes = plt.subplots(4, n_variants,
                             figsize=(5.5 * n_variants, 20),
                             squeeze=False)
    fig.suptitle(
        "F2S adapts per domain; ProteinMPNN collapses to global mean",
        fontsize=12, fontweight="bold",
    )

    f2s_sfam   = sfam_scores.get(frame2seq_key, {})
    global_agg = {r: float(all_scores[r].mean()) for r in all_scores}

    for col, mpnn_key in enumerate(mpnn_keys):
        mpnn_label = DISPLAY_NAMES.get(mpnn_key, mpnn_key)
        mpnn_color = MPNN_COLORS.get(mpnn_key, "#FF7043")
        mpnn_sfam  = sfam_scores.get(mpnn_key, {})

        # ── Row 0: per-sfam cosine sim scatter ────────────────────────────────
        ax = axes[0][col]
        common_sfams = sorted(set(f2s_sfam) & set(mpnn_sfam))
        if common_sfams:
            xs = np.array([mpnn_sfam[s] for s in common_sfams])
            ys = np.array([f2s_sfam[s]  for s in common_sfams])
            sizes = np.array([
                len([d for d, sv in dom2sfam.items()
                     if sv == s and d in all_scores[frame2seq_key].index])
                for s in common_sfams], dtype=float)
            sizes = 20 + 60 * (sizes / sizes.max())

            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
            ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="#2196F3")
            ax.scatter(xs, ys, s=sizes, alpha=0.7, color="#2196F3",
                       edgecolors="white", linewidths=0.4, zorder=2)

            gx = global_agg.get(mpnn_key, float("nan"))
            gy = global_agg.get(frame2seq_key, float("nan"))
            ax.scatter([gx], [gy], s=200, marker="*", color="#FF5722",
                       zorder=5, label="global aggregate")
            ax.legend(fontsize=7, loc="upper left")

            n_wins = int((ys > xs).sum())
            ax.text(0.04, 0.96,
                    f"F2S wins in {n_wins}/{len(common_sfams)} sfams",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_xlabel(f"{mpnn_label.replace(chr(10),' ')}\nmean cosine sim (per sfam)", fontsize=9)
        ax.set_ylabel("Frame2Seq mean cosine sim (per sfam)", fontsize=9)
        ax.set_title(f"Per-sfam: F2S vs {mpnn_label.replace(chr(10),' ')}", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # ── Row 1: per-domain mechanism — class 0 (highly frustrated) ─────────
        _draw_mechanism_row(axes[1], col, dom_dists, frame2seq_key, mpnn_key,
                            mpnn_label, mpnn_color,
                            class_idx=0, class_name="highly frustrated",
                            global_prior=global_prior[0])

        # ── Row 2: per-domain mechanism — class 1 (neutral) ──────────────────
        _draw_mechanism_row(axes[2], col, dom_dists, frame2seq_key, mpnn_key,
                            mpnn_label, mpnn_color,
                            class_idx=1, class_name="neutral",
                            global_prior=global_prior[1])

        # ── Row 3: per-domain mechanism — class 2 (minimally frustrated) ──────
        _draw_mechanism_row(axes[3], col, dom_dists, frame2seq_key, mpnn_key,
                            mpnn_label, mpnn_color,
                            class_idx=2, class_name="minimally frustrated",
                            global_prior=global_prior[2])

    plt.tight_layout()
    sfam_out = str(Path(out).with_name(Path(out).stem + "_sfam_analysis.png"))
    savefig(fig, sfam_out)


def plot(results_dir: Path, test_csv: Path, sfam_filter: list[str] | None,
         out: str, min_residues: int = 10, top_n_sfams: int = 15,
         min_sfam_domains: int = 5):

    runs = load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No enc_*.csv with prob columns in {results_dir}")

    meta = pd.read_csv(test_csv)
    meta["sfam"] = (meta["C"].astype(str) + "." + meta["A"].astype(str) + "." +
                    meta["T"].astype(str) + "." + meta["H"].astype(str))
    dom2sfam = meta.set_index("cath_domain")["sfam"].to_dict()

    frame2seq_key = next((r for r in runs if "frame2seq" in r), None)
    mpnn_keys     = [r for r in runs if "protein_mpnn" in r]

    if frame2seq_key is None:
        raise ValueError("No enc_frame2seq CSV found in results_dir")

    # compute per-domain cosine sim for every run
    all_scores = {}
    for run_name, df in runs.items():
        all_scores[run_name] = domain_scores(df, min_residues=min_residues)

    f2s_scores = all_scores[frame2seq_key]

    # build combined dataframe: one row per (domain, mpnn_variant)
    records = []
    for mpnn_key in mpnn_keys:
        mpnn_scores = all_scores[mpnn_key]
        common = f2s_scores.index.intersection(mpnn_scores.index)
        for dom in common:
            sfam = dom2sfam.get(dom, "unknown")
            if sfam_filter and sfam not in sfam_filter:
                continue
            records.append({
                "domain":  dom,
                "sfam":    sfam,
                "f2s":     f2s_scores[dom],
                "mpnn":    mpnn_scores[dom],
                "variant": mpnn_key,
            })
    df_all = pd.DataFrame(records).dropna()

    # pick top N sfams by domain count for coloring; rest = "other"
    sfam_counts = df_all.groupby("sfam")["domain"].nunique().sort_values(ascending=False)
    top_sfams   = sfam_counts.head(top_n_sfams).index.tolist()
    df_all["sfam_label"] = df_all["sfam"].where(df_all["sfam"].isin(top_sfams), "other")

    cmap   = cm.get_cmap("tab20", len(top_sfams))
    sfam_color = {s: cmap(i) for i, s in enumerate(top_sfams)}
    sfam_color["other"] = (0.7, 0.7, 0.7, 0.3)

    # ── layout ────────────────────────────────────────────────────────────────
    n_variants = len(mpnn_keys)
    fig, axes = plt.subplots(1, n_variants + 1,
                             figsize=(5 * (n_variants + 1), 5.5),
                             gridspec_kw={"width_ratios": [1.2] + [2.5] * n_variants})
    fig.suptitle(
        "Frame2Seq learns domain-specific frustration patterns\n"
        "ProteinMPNN collapses to the global mean",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 0: aggregate bar chart ─────────────────────────────────────────
    ax = axes[0]
    bar_order  = [frame2seq_key] + mpnn_keys
    bar_labels = [DISPLAY_NAMES.get(r, r).replace(" (", "\n(") for r in bar_order]
    bar_colors = ["#2196F3"] + [MPNN_COLORS.get(r, "#888") for r in mpnn_keys]

    # aggregate cosine sim = mean over all domains
    agg = {r: float(all_scores[r].mean()) for r in bar_order if r in all_scores}
    vals = [agg.get(r, float("nan")) for r in bar_order]

    bars = ax.bar(range(len(bar_order)), vals, color=bar_colors,
                  edgecolor="white", linewidth=1.2, width=0.6)
    ax.set_xticks(range(len(bar_order)))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel("Mean cosine similarity", fontsize=10)
    ax.set_title("Aggregate\n(all domains)", fontsize=10)
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panels 1+: per-domain scatter, one panel per ProteinMPNN variant ─────
    for ax_idx, mpnn_key in enumerate(mpnn_keys):
        ax   = axes[ax_idx + 1]
        sub  = df_all[df_all["variant"] == mpnn_key]

        # diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="#2196F3")

        # plot "other" first (background)
        other = sub[sub["sfam_label"] == "other"]
        ax.scatter(other["mpnn"], other["f2s"], c=[sfam_color["other"]],
                   s=18, zorder=1, label=None, linewidths=0)

        # plot top sfams with distinct colors + legend
        for sfam in top_sfams:
            grp = sub[sub["sfam_label"] == sfam]
            if grp.empty:
                continue
            ax.scatter(grp["mpnn"], grp["f2s"],
                       color=sfam_color[sfam], s=35, alpha=0.85,
                       zorder=2, label=sfam, linewidths=0.3, edgecolors="white")

        mpnn_label = DISPLAY_NAMES.get(mpnn_key, mpnn_key)
        ax.set_xlabel(f"{mpnn_label}\ncosine similarity", fontsize=10)
        ax.set_ylabel("Frame2Seq cosine similarity", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Frame2Seq vs {mpnn_label}", fontsize=10)

        # win fraction annotation
        n_f2s_wins = (sub["f2s"] > sub["mpnn"]).sum()
        frac = n_f2s_wins / len(sub) if len(sub) > 0 else 0
        ax.text(0.04, 0.96,
                f"F2S wins {frac:.0%} of domains\n(n={len(sub['domain'].unique())} domains)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.text(0.75, 0.90, "F2S\nwins", color="#2196F3",
                fontsize=8, ha="center", transform=ax.transAxes)

        if ax_idx == len(mpnn_keys) - 1:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=6, loc="lower right",
                          markerscale=1.2, title="Superfamily", title_fontsize=7,
                          ncol=1, framealpha=0.85)

    plt.tight_layout()
    savefig(fig, out)

    # ── Second figure: per-sfam domain adaptation + mechanism ─────────────────
    plot_sfam_analysis(all_scores, runs, dom2sfam, frame2seq_key, mpnn_keys,
                       out=out, min_domains=min_sfam_domains)

    # summary table
    print(f"\n{'Run':<40} {'Mean cos-sim':>12}  {'F2S win %':>10}")
    print("-" * 67)
    for r in bar_order:
        if r == frame2seq_key:
            print(f"{DISPLAY_NAMES.get(r, r):<40} {agg.get(r, float('nan')):>12.4f}  {'(ref)':>10}")
            continue
        sub = df_all[df_all["variant"] == r]
        win = (sub["f2s"] > sub["mpnn"]).mean() if len(sub) else float("nan")
        print(f"{DISPLAY_NAMES.get(r, r):<40} {agg.get(r, float('nan')):>12.4f}  {win:>10.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--test_csv",    required=True)
    parser.add_argument("--sfams",       nargs="*", default=None)
    parser.add_argument("--top_n_sfams", type=int, default=15,
                        help="Number of most-populated sfams to color distinctly")
    parser.add_argument("--min_residues", type=int, default=10,
                        help="Minimum residues per domain to include")
    parser.add_argument("--min_sfam_domains", type=int, default=5,
                        help="Minimum domains per sfam for per-sfam analysis")
    parser.add_argument("--out", default="results/funfam_comparison/domain_adaptation.png")
    args = parser.parse_args()
    plot(Path(args.results_dir), Path(args.test_csv), args.sfams,
         args.out, args.min_residues, args.top_n_sfams, args.min_sfam_domains)


if __name__ == "__main__":
    main()
