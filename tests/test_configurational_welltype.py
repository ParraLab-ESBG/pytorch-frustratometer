"""
test_configurational_welltype.py

Compares our PyTorch configurational frustration values against a
FrustratometeR reference file for a single protein, broken down by
well type (short / long / water-mediated).

Usage (on server, from repo root):
    python tests/test_configurational_welltype.py \
        --pdb      /path/to/XXXX.pdb \
        --ref      /path/to/XXXX.done/FrustrationData/XXXX.pdb_configurational \
        [--ndecoys 4000] [--use-c-rand]

The script prints:
  - Per-pair correlation (Pearson R) overall and by well type
  - Mean absolute error overall and by well type
  - A short table of the 10 pairs with the largest discrepancy
  - Any pairs whose well-type classification disagrees between ref and ours
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


WELLTYPE_COLS = ["Welltype"]
FRUST_COLS    = ["FrstIndex"]
PAIR_KEY_COLS = ["Res1", "Res2", "ChainRes1", "ChainRes2"]


def load_ref(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", skipinitialspace=True, comment="#")
    # Normalise column names (FrustratometeR uses different capitalisations)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_pytorch(pdb_path: str, ndecoys: int, use_c_rand: bool) -> dict:
    """Run our ConfigurationalFrustratometer and return a DataFrame of results."""
    from pytorch_frustratometer import Frustratometer
    from pytorch_frustratometer.data.dataset import FrustratometerDataset, collate_frustratometer
    from pytorch_frustratometer.data.io import write_pair_frustration
    from torch.utils.data import DataLoader

    coefficients_dict = {
        "mode": "configurational",
        "tert_frust_ndecoys": ndecoys,
    }
    from pytorch_frustratometer.coefficients import Coefficients
    coefficients = Coefficients.from_dict(coefficients_dict)

    model = Frustratometer("configurational", coefficients=coefficients)
    model.use_c_rand = use_c_rand
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    dataset = FrustratometerDataset([pdb_path], coefficients)
    batch   = collate_frustratometer([dataset[0]])
    if torch.cuda.is_available():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        result = model(batch)

    fi_conf     = result["configurational"][0]        # [E]
    native_e    = result["_configurational_native_energy"][0]  # [E]
    conf_stats  = result["_configurational_decoy_stats"][0]    # [2]

    edge_index  = batch["edge_index"][0]              # [2, E]
    densities   = batch["residue_density"][0]         # [L]
    aa_type     = batch["aa_type"][0]                 # [L]
    seq_mask    = batch["seq_mask"][0].bool()         # [L]
    distances   = batch["distance"][0]                # [L, L]
    res_idx     = batch.get("residue_index", [None])[0]
    chain_idx   = batch.get("chain_index", [None])[0]

    from pytorch_frustratometer.data.io import _compute_welltypes
    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    # Build valid-edge mask (same as write_pair_frustration)
    # contact_cutoff=2 matches FR's C++ parameter (includes sep >= 2)
    min_seq = int(coefficients.contact_cutoff) if hasattr(coefficients, "contact_cutoff") else 2
    ei0, ei1 = edge_index[0], edge_index[1]
    valid_e  = seq_mask[ei0] & seq_mask[ei1] & (aa_type[ei0] < 20) & (aa_type[ei1] < 20)
    valid_e  = valid_e & ((ei1 - ei0).abs() >= min_seq)
    valid_e  = valid_e.cpu()

    ei0v = ei0[valid_e].cpu()
    ei1v = ei1[valid_e].cpu()
    fi_v = fi_conf[valid_e].cpu().float().numpy()

    wt_arr = _compute_welltypes(
        torch.stack([ei0v, ei1v]),
        densities.cpu(),
        distances.cpu(),
    ).numpy()
    wt_map = {0: "short", 1: "long", 2: "water-mediated"}

    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if res_idx is None:
        res1_arr = (ei0v + 1).numpy()
        res2_arr = (ei1v + 1).numpy()
    else:
        res1_arr = res_idx[ei0v].cpu().numpy()
        res2_arr = res_idx[ei1v].cpu().numpy()

    if chain_idx is None:
        ch1 = ch2 = "A"
    else:
        ch1 = [chain_letters[int(c)] for c in chain_idx[ei0v].cpu().tolist()]
        ch2 = [chain_letters[int(c)] for c in chain_idx[ei1v].cpu().tolist()]

    aa1 = [ID_TO_AA.get(int(a), "X") for a in aa_type[ei0v].cpu().tolist()]
    aa2 = [ID_TO_AA.get(int(a), "X") for a in aa_type[ei1v].cpu().tolist()]

    frust_thr = coefficients.frustration_thresholds[1]
    frust_thr_vals = frust_thr.numpy() if hasattr(frust_thr, "numpy") else np.array(frust_thr)
    fi_class = np.searchsorted(frust_thr_vals, fi_v)
    state_map = {0: "highly", 1: "neutral", 2: "minimally"}

    dens1 = densities[ei0v].cpu().float().numpy()
    dens2 = densities[ei1v].cpu().float().numpy()

    df = pd.DataFrame({
        "Res1":        res1_arr,
        "Res2":        res2_arr,
        "ChainRes1":   ch1,
        "ChainRes2":   ch2,
        "AA1":         aa1,
        "AA2":         aa2,
        "FrstIndex":   fi_v,
        "Welltype":    [wt_map[w] for w in wt_arr],
        "FrstState":   [state_map[int(c)] for c in fi_class],
        "DecoyEnergy": float(conf_stats[0]),
        "SDEnergy":    float(conf_stats[1]),
        "NativeEnergy":  native_e[valid_e].cpu().float().numpy(),
        "DensityRes1":   dens1,
        "DensityRes2":   dens2,
    })
    return df


def compare(ref: pd.DataFrame, ours: pd.DataFrame, verbose: bool = True):
    """Merge on pair key and compare FI, broken down by well type."""
    # Build merge keys: (min_res, max_res, chain1, chain2) to be order-independent
    def make_key(df):
        r1 = df["Res1"].astype(str)
        r2 = df["Res2"].astype(str)
        c1 = df["ChainRes1"].astype(str)
        c2 = df["ChainRes2"].astype(str)
        # symmetric key
        k = pd.Series(zip(r1, r2, c1, c2))
        k_rev = pd.Series(zip(r2, r1, c2, c1))
        return k.where(r1 <= r2, k_rev)

    ref   = ref.copy();   ref["_key"]  = make_key(ref)
    ours  = ours.copy(); ours["_key"] = make_key(ours)

    ref_density_cols  = [c for c in ["DensityRes1", "DensityRes2"] if c in ref.columns]
    ours_density_cols = [c for c in ["DensityRes1", "DensityRes2"] if c in ours.columns]

    ref_cols  = ["_key", "FrstIndex", "Welltype", "NativeEnergy", "DecoyEnergy", "SDEnergy"] + ref_density_cols
    ours_cols = ["_key", "FrstIndex", "Welltype", "NativeEnergy", "DecoyEnergy", "SDEnergy", "AA1", "AA2"] + ours_density_cols

    ref_rename  = {"FrstIndex": "fi_ref",  "Welltype": "wt_ref",
                   "NativeEnergy": "nat_ref",  "DecoyEnergy": "dec_ref",  "SDEnergy": "sd_ref",
                   "DensityRes1": "dens1_ref", "DensityRes2": "dens2_ref"}
    ours_rename = {"FrstIndex": "fi_ours", "Welltype": "wt_ours",
                   "NativeEnergy": "nat_ours", "DecoyEnergy": "dec_ours", "SDEnergy": "sd_ours",
                   "DensityRes1": "dens1_ours", "DensityRes2": "dens2_ours"}

    merged = pd.merge(
        ref[ref_cols].rename(columns=ref_rename),
        ours[ours_cols].rename(columns=ours_rename),
        on="_key", how="inner",
    )

    if verbose:
        print(f"\nMatched pairs: {len(merged):,}  "
              f"(ref={len(ref):,}, ours={len(ours):,})")

    fi_ref  = merged["fi_ref"].values
    fi_ours = merged["fi_ours"].values
    diff    = np.abs(fi_ours - fi_ref)

    def stats(mask=None, label="ALL"):
        m = mask if mask is not None else np.ones(len(merged), dtype=bool)
        if m.sum() == 0:
            return
        r = np.corrcoef(fi_ref[m], fi_ours[m])[0, 1]
        mae = diff[m].mean()
        rmse = np.sqrt((diff[m] ** 2).mean())
        print(f"  {label:20s}  n={m.sum():>7,}  Pearson R={r:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    print("\n=== FI correlation by well type ===")
    stats(label="Overall")
    for wt in ("short", "long", "water-mediated"):
        mask = (merged["wt_ref"] == wt).values
        stats(mask, label=wt)

    # Well-type mismatches
    wt_mismatch = merged["wt_ref"] != merged["wt_ours"]
    print(f"\nWell-type mismatches: {wt_mismatch.sum():,} / {len(merged):,}")
    if wt_mismatch.sum() > 0 and verbose:
        print(merged[wt_mismatch][["_key", "wt_ref", "wt_ours", "fi_ref", "fi_ours",
                                    "nat_ref", "nat_ours"]].head(10).to_string(index=False))

    # Native energy comparison
    nat_ref  = merged["nat_ref"].values
    nat_ours = merged["nat_ours"].values
    finite = np.isfinite(nat_ref) & np.isfinite(nat_ours)
    if finite.any():
        r_nat = np.corrcoef(nat_ref[finite], nat_ours[finite])[0, 1]
        mae_nat = np.abs(nat_ours[finite] - nat_ref[finite]).mean()
        print(f"\nNativeEnergy:  Pearson R={r_nat:.4f}  MAE={mae_nat:.4f}")
        print(f"  DecoyEnergy ref mean={merged['dec_ref'].mean():.4f}  ours mean={merged['dec_ours'].mean():.4f}")
        print(f"  SDEnergy    ref mean={merged['sd_ref'].mean():.4f}   ours mean={merged['sd_ours'].mean():.4f}")

    # Density comparison (if both sides have it)
    if "dens1_ref" in merged.columns and "dens1_ours" in merged.columns:
        d1_ref  = merged["dens1_ref"].values
        d1_ours = merged["dens1_ours"].values
        d2_ref  = merged["dens2_ref"].values
        d2_ours = merged["dens2_ours"].values
        ok = np.isfinite(d1_ref) & np.isfinite(d1_ours) & np.isfinite(d2_ref) & np.isfinite(d2_ours)
        if ok.any():
            r1 = np.corrcoef(d1_ref[ok], d1_ours[ok])[0, 1]
            r2 = np.corrcoef(d2_ref[ok], d2_ours[ok])[0, 1]
            mae1 = np.abs(d1_ours[ok] - d1_ref[ok]).mean()
            mae2 = np.abs(d2_ours[ok] - d2_ref[ok]).mean()
            print(f"\nDensityRes1:  Pearson R={r1:.4f}  MAE={mae1:.4f}  "
                  f"ref mean={d1_ref[ok].mean():.4f}  ours mean={d1_ours[ok].mean():.4f}")
            print(f"DensityRes2:  Pearson R={r2:.4f}  MAE={mae2:.4f}  "
                  f"ref mean={d2_ref[ok].mean():.4f}  ours mean={d2_ours[ok].mean():.4f}")

    # Top discrepancies
    worst = merged.nlargest(10, "diff" if "diff" in merged.columns else "_key",
                            keep="first") if "diff" in merged.columns else \
            merged.assign(diff=diff).nlargest(10, "diff")
    print("\n=== Top 10 discrepancies ===")
    dens_show = ["dens1_ref", "dens1_ours", "dens2_ref", "dens2_ours"]
    dens_show = [c for c in dens_show if c in worst.columns]
    print(worst[["_key", "wt_ref", "wt_ours", "fi_ref", "fi_ours",
                  "nat_ref", "nat_ours", "dec_ref", "dec_ours"] + dens_show].to_string(index=False))

    return merged


def sweep_decoys(pdb: str, ref: pd.DataFrame, decoy_counts: list[int],
                 use_c_rand: bool, out_path: str | None):
    """
    Run the model with each n_decoys value and plot MAE / Pearson-R vs n_decoys.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    records = []
    for nd in decoy_counts:
        print(f"  n_decoys={nd:>6,} ...", end=" ", flush=True)
        ours = run_pytorch(pdb, nd, use_c_rand)

        # merge
        def make_key(df):
            r1, r2 = df["Res1"].astype(str), df["Res2"].astype(str)
            c1, c2 = df["ChainRes1"].astype(str), df["ChainRes2"].astype(str)
            k = pd.Series(list(zip(r1, r2, c1, c2)))
            k_rev = pd.Series(list(zip(r2, r1, c2, c1)))
            return k.where(r1 <= r2, k_rev)

        r_df  = ref.copy();  r_df["_key"]  = make_key(r_df)
        o_df  = ours.copy(); o_df["_key"]  = make_key(o_df)
        m = pd.merge(
            r_df[["_key", "FrstIndex"]].rename(columns={"FrstIndex": "fi_ref"}),
            o_df[["_key", "FrstIndex"]].rename(columns={"FrstIndex": "fi_ours"}),
            on="_key", how="inner",
        )
        if len(m) == 0:
            print("no matched pairs, skipping")
            continue
        fi_ref  = m["fi_ref"].values
        fi_ours = m["fi_ours"].values
        mae  = np.abs(fi_ours - fi_ref).mean()
        rmse = np.sqrt(((fi_ours - fi_ref) ** 2).mean())
        r, _ = pearsonr(fi_ref, fi_ours)
        print(f"n={len(m):,}  MAE={mae:.4f}  R={r:.6f}")
        records.append({"n_decoys": nd, "mae": mae, "rmse": rmse, "pearson_r": r, "n_pairs": len(m)})

    if not records:
        print("No records to plot.")
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Configurational FI accuracy vs number of decoys\n"
                 f"(single protein, n_pairs={df['n_pairs'].iloc[0]:,})")

    ax1, ax2 = axes
    ax1.plot(df["n_decoys"], df["mae"],  "o-", color="#E74C3C", label="MAE")
    ax1.plot(df["n_decoys"], df["rmse"], "s--", color="#E67E22", label="RMSE")
    # theoretical 1/√n scaling anchored at largest n_decoys
    nd_arr = np.array(df["n_decoys"])
    ref_mae  = df["mae"].iloc[-1]
    ref_nd   = df["n_decoys"].iloc[-1]
    theory   = ref_mae * np.sqrt(ref_nd / nd_arr)
    ax1.plot(nd_arr, theory, ":", color="gray", alpha=0.7, label=r"$\propto 1/\sqrt{n}$")
    ax1.set_xscale("log")
    ax1.set_xlabel("n_decoys")
    ax1.set_ylabel("Error (FI units)")
    ax1.set_title("MAE & RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["n_decoys"], 1 - df["pearson_r"], "o-", color="#2980B9")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("n_decoys")
    ax2.set_ylabel("1 − Pearson R  (log scale)")
    ax2.set_title("Pearson R deficit")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_path or "decoy_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb",       required=True, help="Input PDB file")
    ap.add_argument("--ref",       required=True, help="FrustratometeR configurational output file")
    ap.add_argument("--ndecoys",   type=int, default=4000)
    ap.add_argument("--use-c-rand", action="store_true", dest="use_c_rand",
                    help="Use C random seeding (matches FrustratometeR C++ binary exactly)")
    ap.add_argument("--verbose",   action="store_true", default=True)
    ap.add_argument("--sweep",     action="store_true",
                    help="Run n_decoys sweep and produce accuracy-vs-decoys plot")
    ap.add_argument("--sweep-values", default="100,250,500,1000,2000,4000,8000,16000,32000",
                    help="Comma-separated n_decoys values for --sweep")
    ap.add_argument("--out",       default=None, help="Output plot path for --sweep")
    args = ap.parse_args()

    print(f"Loading reference: {args.ref}")
    ref = load_ref(args.ref)
    print(f"  {len(ref):,} pairs, columns: {list(ref.columns)}")

    if args.sweep:
        counts = [int(x) for x in args.sweep_values.split(",")]
        print(f"\nDecoy sweep: {counts}")
        sweep_decoys(args.pdb, ref, counts, args.use_c_rand, args.out)
        return

    print(f"\nRunning PyTorch model on: {args.pdb}")
    ours = run_pytorch(args.pdb, args.ndecoys, args.use_c_rand)
    print(f"  {len(ours):,} pairs")

    compare(ref, ours, verbose=args.verbose)


if __name__ == "__main__":
    main()
