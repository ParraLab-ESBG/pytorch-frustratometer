"""
Detect residue modules with correlated dynamic frustration.

WARNING: This has not been run or tested.

Port of FrustratometeR's detect_dynamic_clusters() / dynamic_res() functions.

Pipeline
--------
1. Load per-residue singleresidue FrstIndex across frames (ordered structures).
2. Fit a LOWESS smoother per residue to remove noise (matches R loess()).
3. Filter residues by frustration range (> quantile) and mean (|mean| > threshold).
4. PCA on the filtered residue × frame matrix, keep n_components.
5. Compute pairwise Spearman or Pearson correlation in PCA space.
6. Build an undirected NetworkX graph: edge iff |corr| > min_corr AND p <= 0.05.
7. Run Leiden community detection (nx-cugraph GPU first, leidenalg CPU fallback).
8. Return graph, community assignments, and per-residue statistics.

All heavy compute (LOWESS, PCA, correlation) runs in PyTorch so it benefits from
GPU acceleration when available.  The LOWESS hat-matrix formulation processes all
residues simultaneously with a single batch of linear-algebra ops.

Usage
-----
    from pytorch_frustratometer.analysis import detect_dynamic_clusters

    result = detect_dynamic_clusters(
        results_dir="/path/to/results",
        order_list=["frame_001.pdb", "frame_002.pdb", ...],
    )
    for i, community in enumerate(result["communities"]):
        print(f"Community {i}: {community}")
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ---------------------------------------------------------------------------
# File discovery / loading
# ---------------------------------------------------------------------------

def _find_frustration_files(
    results_dir: Path,
    order_list: Sequence[str] | Sequence[Path] | None,
    mode: str,
) -> list[Path]:
    if order_list is not None:
        return [
            results_dir / f"{Path(item).stem}.done"
            / "FrustrationData" / f"{Path(item).stem}.pdb_{mode}"
            for item in order_list
        ]
    return sorted(
        done_dir / "FrustrationData" / f"{done_dir.name.removesuffix('.done')}.pdb_{mode}"
        for done_dir in sorted(results_dir.glob("*.done"))
        if (done_dir / "FrustrationData" /
            f"{done_dir.name.removesuffix('.done')}.pdb_{mode}").exists()
    )


def _load_frame(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    aa_col = "AA" if "AA" in df.columns else None
    labels = (
        [f"{_AA1_TO_AA3.get(str(a), 'UNK')}_{int(r)}"
         for a, r in zip(df[aa_col], df["Res"])]
        if aa_col else [f"RES_{int(r)}" for r in df["Res"]]
    )
    return pd.Series(df["FrstIndex"].values, index=labels, dtype=float)


# ---------------------------------------------------------------------------
# Vectorised LOWESS (all residues in one batch of linear-algebra ops)
# ---------------------------------------------------------------------------

def _lowess_hat_matrix(
    n_frames: int,
    frac: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute the LOWESS hat matrix H [n_frames, n_frames].

    H is the linear smoother such that fitted = H @ y for any single residue's
    frustration vector y.  All residues share the same x-grid (frame indices),
    so H is computed once and reused.

    Uses degree-1 (linear) tricube-weighted local regression, matching
    R's loess(degree=1, family="gaussian") with no robustness iterations.
    """
    x = torch.arange(n_frames, dtype=dtype, device=device)
    k = max(1, math.ceil(frac * n_frames))

    # Pairwise absolute distances [n_target, n_obs]
    abs_dx = (x.unsqueeze(0) - x.unsqueeze(1)).abs()

    # Tricube bandwidth: k-th nearest neighbour distance per target point
    h = abs_dx.sort(dim=1).values[:, k - 1].clamp(min=1e-10)  # [n_frames]

    # Tricube weights W[i, j] = (1 - (|xi-xj|/hi)^3)^3
    u = (abs_dx / h.unsqueeze(1)).clamp(max=1.0)
    W = (1.0 - u.pow(3)).pow(3)  # [n_target, n_obs]

    # Design matrix X = [1, x]  [n_obs, 2]
    X = torch.stack([torch.ones(n_frames, dtype=dtype, device=device), x], dim=1)

    # Weighted design: WX[i, j, :] = W[i, j] * X[j, :]  [n_target, n_obs, 2]
    WX = W.unsqueeze(2) * X.unsqueeze(0)

    # X^T W_i X  [n_target, 2, 2]
    XTWX = torch.bmm(WX.transpose(1, 2), X.unsqueeze(0).expand(n_frames, -1, -1))

    # Solve XTWX @ B = WX^T  →  B [n_target, 2, n_obs]
    B = torch.linalg.solve(XTWX, WX.transpose(1, 2))

    # Hat row H[i, :] = X[i] @ B[i]  →  [n_target, n_obs]
    return torch.bmm(X.unsqueeze(1), B).squeeze(1)  # [n_frames, n_frames]


def _lowess_torch(
    data: torch.Tensor,
    frac: float,
) -> torch.Tensor:
    """Vectorised LOWESS for [n_residues, n_frames] → [n_residues, n_frames].

    Precomputes the shared hat matrix H once (all residues share the same
    equally-spaced frame grid), then applies fitted = (H @ data.T).T in a
    single GEMM — no per-residue loop.

    Memory: O(n_frames² + n_frames·n_residues).  For n_frames=1000 this is
    ~8 MB for H alone; at n_frames=5000 it grows to ~200 MB — consider
    chunking if GPU VRAM is limited.
    """
    H = _lowess_hat_matrix(data.shape[1], frac, data.dtype, data.device)
    # fitted [n_res, n_frames] = (H [n_frames, n_frames] @ data.T [n_frames, n_res]).T
    return (H @ data.T).T


# ---------------------------------------------------------------------------
# PCA (torch.pca_lowrank — randomised SVD, GPU-friendly)
# ---------------------------------------------------------------------------

def _pca_torch(
    data: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA via randomised SVD.

    Parameters
    ----------
    data : [n_samples, n_features]

    Returns
    -------
    coords : [n_samples, n_components]  — individual coordinates in PCA space
    explained_ratio : [n_components]    — fraction of variance per component
    """
    n_samples = data.shape[0]
    mean = data.mean(dim=0)
    centered = data - mean
    q = min(n_components, *centered.shape)
    U, S, _Vh = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    coords = U * S.unsqueeze(0)                           # [n_samples, q]
    total_var = (centered ** 2).sum() / (n_samples - 1)
    explained = (S ** 2) / (n_samples - 1) / total_var   # [q]
    return coords, explained


# ---------------------------------------------------------------------------
# Correlation matrix (fully vectorised, matching R rcorr(t(pca_coords)))
# ---------------------------------------------------------------------------

def _rank_tensor(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Integer ranks (1-based) along *dim*.  Ties broken by position."""
    return x.argsort(dim=dim).argsort(dim=dim).to(x.dtype) + 1.0


def _pearson_from_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson correlation + two-tailed p-values between rows of x [n, m].

    Treats each row as a variable and the m columns as observations.
    Returns (corr [n, n], p_mat [n, n]).
    """
    _n, m = x.shape
    x_c = x - x.mean(dim=1, keepdim=True)
    norms = x_c.norm(dim=1, keepdim=True).clamp(min=1e-10)
    x_n = x_c / norms                                    # unit-normalised
    corr = (x_n @ x_n.T).clamp(-1.0, 1.0)
    corr.fill_diagonal_(1.0)

    # t-statistic: t = r * sqrt(df) / sqrt(1 - r²)
    df = max(m - 2, 1)
    r2 = corr.pow(2).clamp(max=1.0 - 1e-10)
    t_stat = corr * math.sqrt(df) / (1.0 - r2).sqrt()

    # Two-tailed p-values via scipy Student-t (device-agnostic; small matrix)
    from scipy.stats import t as t_dist
    p_np = 2.0 * t_dist.sf(t_stat.abs().cpu().numpy(), df=df)
    p_mat = torch.from_numpy(p_np).to(dtype=corr.dtype, device=corr.device)
    p_mat.fill_diagonal_(0.0)
    return corr, p_mat


def _correlation_matrix(
    coords: torch.Tensor,
    corr_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise correlation between residues from their PCA coordinates.

    coords: [n_residues, n_components]

    Matches R:  rcorr(t(pca$ind$coord), type=corr_type)
    which correlates the *columns* of t(coords) = the *rows* of coords,
    treating the n_components PCA coords as the n observations per pair.

    For Spearman: rank each residue's n_component values, then Pearson on ranks.
    For Pearson : directly compute Pearson on the raw PCA coordinates.

    Returns (corr [n, n], p_mat [n, n]).
    """
    if corr_type.lower() == "spearman":
        # Rank along n_components for each residue (row), then Pearson on ranks
        ranked = _rank_tensor(coords, dim=1)   # [n_res, n_comp]
        return _pearson_from_rows(ranked)
    else:
        return _pearson_from_rows(coords)


# ---------------------------------------------------------------------------
# Leiden community detection (nx-cugraph GPU → leidenalg CPU)
# ---------------------------------------------------------------------------

def _leiden_communities(
    G,       # nx.Graph
    resolution: float,
    seed: int,
) -> list[frozenset]:
    import networkx as nx
    try:
        import nx_cugraph  # noqa: F401  — registers the "cugraph" backend
    except ImportError as exc:
        raise ImportError(
            "nx-cugraph is required for Leiden clustering.\n"
            "Install it with:  pip install pytorch-frustratometer[analysis]"
        ) from exc

    communities = nx.community.leiden_communities(
        G, weight="weight", resolution=resolution, seed=seed
    )
    return [frozenset(c) for c in communities]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_dynamic_clusters(
    results_dir: str | Path,
    order_list: Sequence[str] | Sequence[Path] | None = None,
    mode: str = "singleresidue",
    loess_span: float = 0.05,
    min_frst_range: float = 0.7,
    filt_mean: float = 0.15,
    n_components: int = 10,
    min_corr: float = 0.95,
    leiden_resolution: float = 1.0,
    corr_type: str = "spearman",
    p_threshold: float = 0.05,
    device: str | torch.device | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Detect residue modules with correlated dynamic frustration.

    WARNING: This has not been run or tested.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``*.done/FrustrationData/`` sub-trees.
    order_list : sequence of str/Path, optional
        Ordered list of PDB paths or stems defining the frame order.
        Auto-discovers all ``*.done/`` directories when None.
    mode : str
        Frustration mode to read (must be ``"singleresidue"``).
    loess_span : float
        LOWESS bandwidth fraction (R ``loess(span=...)``).  Default 0.05.
    min_frst_range : float
        Quantile threshold for the per-residue frustration range filter.
        Residues below this quantile are dropped.  Default 0.70.
    filt_mean : float
        Drop residues with |mean_frustration| ≤ filt_mean.  Default 0.15.
    n_components : int
        PCA components to retain.  Default 10.
    min_corr : float
        Minimum |correlation| for a graph edge.  Default 0.95.
    leiden_resolution : float
        Leiden CPM resolution parameter.  Default 1.0.
    corr_type : str
        ``"spearman"`` (default) or ``"pearson"``.
    p_threshold : float
        Maximum p-value for a graph edge.  Default 0.05.
    device : str or torch.device, optional
        PyTorch device for heavy compute (LOWESS, PCA, correlation).
        Defaults to CUDA if available, else CPU.
    random_state : int
        Random seed.  Default 42.
    verbose : bool
        Print progress.  Default True.

    Returns
    -------
    dict
        ``graph``           — NetworkX undirected graph.
        ``communities``     — list of frozensets of node names.
        ``node_community``  — dict node_name → community index.
        ``n_communities``   — int.
        ``frustra_data``    — DataFrame [n_res × n_frames] (raw).
        ``frustra_filtered``— DataFrame [n_kept × n_frames] (after filter).
        ``fitted``          — Tensor [n_res × n_frames] (LOWESS-fitted).
        ``means``           — Tensor [n_res] (fitted means).
        ``frst_range``      — Tensor [n_res] (fitted ranges).
        ``sd``              — Tensor [n_res] (fitted stds).
        ``corr_mat``        — Tensor [n_kept × n_kept].
        ``p_mat``           — Tensor [n_kept × n_kept].
        Plus all input parameters for reproducibility.
    """
    import networkx as nx

    if mode != "singleresidue":
        raise ValueError(
            f"detect_dynamic_clusters requires mode='singleresidue', got {mode!r}"
        )

    results_dir = Path(results_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ------------------------------------------------------------------
    # 1. Discover & load frustration files
    # ------------------------------------------------------------------
    if verbose:
        print("Loading frustration data ...", flush=True)

    paths = _find_frustration_files(results_dir, order_list, mode)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frustration file(s) not found, e.g.:\n  {missing[0]}"
        )
    if not paths:
        raise FileNotFoundError(
            f"No singleresidue frustration files found under {results_dir}"
        )

    s0 = _load_frame(paths[0])
    residue_labels = s0.index.tolist()

    frames_np = [s0.values] + [
        _load_frame(p).reindex(residue_labels).values for p in paths[1:]
    ]
    # data: [n_residues, n_frames]
    data = torch.tensor(
        pd.DataFrame(frames_np).T.values,  # align via DataFrame
        dtype=torch.float32,
        device=device,
    )
    n_residues, n_frames = data.shape
    frustra_data = pd.DataFrame(
        data.cpu().numpy(),
        index=residue_labels,
        columns=[f"frame_{i+1}" for i in range(n_frames)],
    )

    if verbose:
        print(f"  {n_residues} residues × {n_frames} frames on {device}", flush=True)

    # ------------------------------------------------------------------
    # 2. Vectorised LOWESS smoothing — shared hat matrix, single GEMM
    # ------------------------------------------------------------------
    if verbose:
        print("Fitting LOWESS smoother (vectorised) ...", flush=True)

    fitted = _lowess_torch(data, frac=loess_span)  # [n_res, n_frames]

    frst_range = fitted.max(dim=1).values - fitted.min(dim=1).values  # [n_res]
    means      = fitted.mean(dim=1)                                    # [n_res]
    sds        = fitted.std(dim=1)                                     # [n_res]

    # ------------------------------------------------------------------
    # 3. Filter residues by range quantile and mean magnitude
    # ------------------------------------------------------------------
    range_threshold = torch.quantile(frst_range, min_frst_range)
    keep_mask = (frst_range > range_threshold) & (means.abs() > filt_mean)  # [n_res]

    n_kept = int(keep_mask.sum())
    if verbose:
        print(
            f"  Kept {n_kept}/{n_residues} residues "
            f"(range > {range_threshold:.3f}, |mean| > {filt_mean})",
            flush=True,
        )
    if n_kept < 2:
        raise ValueError(
            f"Only {n_kept} residue(s) passed the filter (need ≥ 2). "
            "Try lowering min_frst_range or filt_mean."
        )

    keep_idx     = keep_mask.nonzero(as_tuple=True)[0]   # [n_kept]
    kept_labels  = [residue_labels[i] for i in keep_idx.tolist()]
    filtered     = data[keep_idx]                         # [n_kept, n_frames]
    frustra_filtered = pd.DataFrame(
        filtered.cpu().numpy(),
        index=kept_labels,
        columns=frustra_data.columns,
    )

    # ------------------------------------------------------------------
    # 4. PCA (randomised SVD, GPU-friendly)
    # ------------------------------------------------------------------
    if verbose:
        print("Running PCA ...", flush=True)

    torch.manual_seed(random_state)
    coords, explained = _pca_torch(filtered, n_components)  # [n_kept, n_comp]
    n_comp = coords.shape[1]

    if verbose:
        print(
            f"  {n_comp} components → {explained.sum().item():.1%} variance explained",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 5. Pairwise correlation in PCA space (vectorised)
    # ------------------------------------------------------------------
    if verbose:
        print(f"Computing {corr_type} correlations ({n_kept}×{n_kept}) ...", flush=True)

    corr_mat, p_mat = _correlation_matrix(coords, corr_type)  # [n_kept, n_kept]

    # ------------------------------------------------------------------
    # 6. Build graph: upper-triangle edges with |corr| > min_corr AND p <= thr
    # ------------------------------------------------------------------
    if verbose:
        print("Building correlation graph ...", flush=True)

    rows_idx, cols_idx = torch.triu_indices(n_kept, n_kept, offset=1, device=device)
    c_vals = corr_mat[rows_idx, cols_idx]
    p_vals = p_mat[rows_idx, cols_idx]
    edge_mask = (c_vals.abs() > min_corr) & (p_vals <= p_threshold)

    G = nx.Graph()
    G.add_nodes_from(kept_labels)
    if edge_mask.any():
        # Move to CPU for NetworkX
        e_rows = rows_idx[edge_mask].cpu().tolist()
        e_cols = cols_idx[edge_mask].cpu().tolist()
        e_corr = c_vals[edge_mask].abs().cpu().tolist()
        e_signed = c_vals[edge_mask].cpu().tolist()
        for u_i, v_i, w, c in zip(e_rows, e_cols, e_corr, e_signed):
            G.add_edge(kept_labels[u_i], kept_labels[v_i], weight=w, corr=c)

    # Remove isolated nodes (degree 0) before Leiden
    isolated = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolated)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if verbose:
        print(
            f"  {n_nodes} nodes, {n_edges} edges "
            f"({len(isolated)} isolated removed)",
            flush=True,
        )

    if n_nodes == 0:
        if verbose:
            print("  No connected nodes — no clusters detected.", flush=True)
        return _build_result(
            G, [], {}, frustra_data, frustra_filtered, fitted,
            means, frst_range, sds, corr_mat, p_mat,
            loess_span, min_frst_range, filt_mean, n_comp,
            min_corr, leiden_resolution, corr_type, p_threshold,
        )

    # ------------------------------------------------------------------
    # 7. Leiden community detection
    # ------------------------------------------------------------------
    if verbose:
        print("Running Leiden clustering ...", flush=True)

    communities = _leiden_communities(G, resolution=leiden_resolution, seed=random_state)
    node_community = {
        node: cid
        for cid, community in enumerate(communities)
        for node in community
    }

    if verbose:
        print(f"  {len(communities)} communities found.", flush=True)

    return _build_result(
        G, communities, node_community,
        frustra_data, frustra_filtered, fitted,
        means, frst_range, sds, corr_mat, p_mat,
        loess_span, min_frst_range, filt_mean, n_comp,
        min_corr, leiden_resolution, corr_type, p_threshold,
    )


def _build_result(
    G, communities, node_community,
    frustra_data, frustra_filtered, fitted,
    means, frst_range, sds, corr_mat, p_mat,
    loess_span, min_frst_range, filt_mean, n_components,
    min_corr, leiden_resolution, corr_type, p_threshold,
) -> dict:
    return {
        "graph":            G,
        "communities":      communities,
        "node_community":   node_community,
        "n_communities":    len(communities),
        "frustra_data":     frustra_data,
        "frustra_filtered": frustra_filtered,
        "fitted":           fitted,
        "means":            means,
        "frst_range":       frst_range,
        "sd":               sds,
        "corr_mat":         corr_mat,
        "p_mat":            p_mat,
        "loess_span":        loess_span,
        "min_frst_range":    min_frst_range,
        "filt_mean":         filt_mean,
        "n_components":      n_components,
        "min_corr":          min_corr,
        "leiden_resolution": leiden_resolution,
        "corr_type":         corr_type,
        "p_threshold":       p_threshold,
    }


# ---------------------------------------------------------------------------
# Accessors (matching FrustratometeR's get_clusters() / get_frustration())
# ---------------------------------------------------------------------------

def get_clusters(result: dict, clusters: list[int] | str = "all") -> pd.DataFrame:
    """Return a DataFrame of residue cluster membership.

    WARNING: This has not been run or tested.

    Matches FrustratometeR's get_clusters().

    Parameters
    ----------
    result : dict
        Output of :func:`detect_dynamic_clusters`.
    clusters : list[int] or "all"
        Community indices to include.

    Returns
    -------
    DataFrame with columns: AA, Res, Cluster, Mean, Sd, FrstRange
    """
    if not result.get("communities"):
        return pd.DataFrame(columns=["AA", "Res", "Cluster", "Mean", "Sd", "FrstRange"])

    residue_labels = list(result["frustra_data"].index)
    label_to_idx   = {lab: i for i, lab in enumerate(residue_labels)}
    means      = result["means"]
    sds        = result["sd"]
    frst_range = result["frst_range"]

    rows = []
    for cid, community in enumerate(result["communities"]):
        if clusters != "all" and cid not in clusters:
            continue
        for node in sorted(community):
            parts = node.rsplit("_", 1)
            aa3 = parts[0] if len(parts) == 2 else node
            res = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else -1
            idx = label_to_idx.get(node)
            rows.append({
                "AA":        aa3,
                "Res":       res,
                "Cluster":   cid,
                "Mean":      means[idx].item()      if idx is not None else float("nan"),
                "Sd":        sds[idx].item()        if idx is not None else float("nan"),
                "FrstRange": frst_range[idx].item() if idx is not None else float("nan"),
            })

    return pd.DataFrame(rows)
