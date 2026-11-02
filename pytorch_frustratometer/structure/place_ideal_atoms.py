import math as _math_module
import torch
from .geometry import place_fourth_atom


def _place_fourth_atom_frame2seq(
    a_coord: torch.Tensor,
    b_coord: torch.Tensor,
    c_coord: torch.Tensor,
    length: torch.Tensor,
    planar: torch.Tensor,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Original Frame2Seq/IPA implementation.

    Builds a local backbone frame and places the 4th atom relative to c_coord.
    This is correct for IPA's local coordinate representation but should NOT be
    used to reconstruct real 3D atomic coordinates (use place_missing_cb /
    place_missing_o for that).
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).cross(bc_vec, dim=-1)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec, dim=-1), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral),
    ]
    return c_coord + sum(m * d for m, d in zip(m_vec, d_vec))


def _place_fourth_atom_batch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length: float,
    planar: float,
    dihedral: float,
) -> torch.Tensor:
    """
    Batch version of _place_fourth_atom_frame2seq.

    a, b, c : [N, 3]  — the three reference atoms (same convention as scalar version)
    length, planar, dihedral : Python floats (pre-compute sin/cos once)

    Returns d : [N, 3]  placed atom positions.
    """
    cos_p = _math_module.cos(planar)
    sin_p = _math_module.sin(planar)
    cos_d = _math_module.cos(dihedral)
    sin_d = _math_module.sin(dihedral)

    bc_vec = b - c
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n_vec  = torch.linalg.cross(b - a, bc_vec)
    n_vec  = n_vec  / n_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    m2     = torch.linalg.cross(n_vec, bc_vec)

    return (c
            + (length * cos_p)       * bc_vec
            + (length * sin_p * cos_d) * m2
            + (-length * sin_p * sin_d) * n_vec)


def _vectorized_ideal_geometry(
    pos:   torch.Tensor,   # [L, ≥3, 3]  modified in-place
    amask: torch.Tensor,   # [L, 3] bool modified in-place
    seq_mask: torch.Tensor,  # [L] bool — residues already complete (skip)
    dtype,
    device,
) -> None:
    """
    Multi-pass vectorized backbone reconstruction.

    Each pass handles all residues whose dependencies are already satisfied,
    using tensor masked-gather ops instead of a Python for-loop over residues.
    Multiple passes propagate cascade dependencies (e.g. CA-only stretches
    where C(i) placed in pass k enables N(i+1) placement in pass k+1).

    The C-placement step uses the extended-chain psi approximation instead of
    the cone-sphere method, which is slightly less accurate but fully
    vectorizable.  MODELLER or cg2all should be preferred when accuracy matters.

    Modifies *pos* and *amask* in-place.  seq_mask is read-only here.
    """
    L = pos.shape[0]
    if L == 0:
        return

    # Pre-compute geometry constants as Python floats (used in _place_fourth_atom_batch)
    cn   = float(_C_N_BOND);    nca  = float(_N_CA_BOND);  cac  = float(_CA_C_BOND)
    acn  = float(_ANG_CA_C_N);  anc  = float(_ANG_C_N_CA); anca = float(_ANG_N_CA_C)
    om   = float(_OMEGA);       ps   = float(_PSI_EXT)

    prev_c  = torch.zeros(L, dtype=torch.bool, device=device)
    prev_ca = torch.zeros(L, dtype=torch.bool, device=device)
    prev_n  = torch.zeros(L, dtype=torch.bool, device=device)
    next_n  = torch.zeros(L, dtype=torch.bool, device=device)
    next_ca = torch.zeros(L, dtype=torch.bool, device=device)

    for _ in range(L):   # worst-case: one cascade step per pass; break on convergence
        snap = amask.clone()

        n, ca, c = amask[:, 0], amask[:, 1], amask[:, 2]
        if L > 1:
            prev_c[:] = False;  prev_c[1:]  = c[:-1]
            prev_ca[:] = False; prev_ca[1:] = ca[:-1]
            prev_n[:] = False;  prev_n[1:]  = n[:-1]
            next_n[:] = False;  next_n[:-1] = n[1:]
            next_ca[:] = False; next_ca[:-1]= ca[1:]

        todo = ~seq_mask

        # ── N placement ──────────────────────────────────────────────────────
        # missing N, has CA & C & prev-C
        m = todo & ~n & ca & c & prev_c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 0] = _place_fourth_atom_batch(
                pos[idx, 2], pos[idx, 1], pos[idx - 1, 2], cn, acn, om)
            amask[idx, 0] = True

        # CA-only (no N, no C), has prev-CA & prev-C  → forward cascade step
        n = amask[:, 0]; c = amask[:, 2]
        if L > 1:
            prev_c[1:]  = amask[:-1, 2]
            prev_ca[1:] = amask[:-1, 1]
        m = todo & ~n & ca & ~c & prev_ca & prev_c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 0] = _place_fourth_atom_batch(
                pos[idx, 1], pos[idx - 1, 1], pos[idx - 1, 2], cn, acn, om)
            amask[idx, 0] = True

        # no backbone, has prev full backbone → extended-chain N
        n = amask[:, 0]; ca = amask[:, 1]; c = amask[:, 2]
        if L > 1:
            prev_n[1:]  = amask[:-1, 0]
            prev_ca[1:] = amask[:-1, 1]
            prev_c[1:]  = amask[:-1, 2]
        m = todo & ~n & ~ca & ~c & prev_n & prev_ca & prev_c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 0] = _place_fourth_atom_batch(
                pos[idx - 1, 0], pos[idx - 1, 1], pos[idx - 1, 2], cn, acn, ps)
            amask[idx, 0] = True

        # ── CA placement ─────────────────────────────────────────────────────
        n = amask[:, 0]; ca = amask[:, 1]; c = amask[:, 2]
        if L > 1:
            prev_ca[1:] = amask[:-1, 1]
            prev_c[1:]  = amask[:-1, 2]

        # only N known, has prev-CA & prev-C → omega placement
        m = todo & n & ~ca & ~c & prev_ca & prev_c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 1] = _place_fourth_atom_batch(
                pos[idx - 1, 1], pos[idx - 1, 2], pos[idx, 0], nca, anc, om)
            amask[idx, 1] = True

        # has N & C, no CA → midpoint + perpendicular
        ca = amask[:, 1]
        m = todo & n & ~ca & c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            mid  = 0.5 * (pos[idx, 0] + pos[idx, 2])
            nc   = pos[idx, 2] - pos[idx, 0]
            ref  = torch.tensor([[0., 0., 1.]], dtype=dtype, device=device).expand(len(idx), -1).clone()
            near = (nc / nc.norm(dim=-1, keepdim=True).clamp(min=1e-8) * ref).sum(-1).abs() > 0.9
            ref[near] = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
            perp = torch.linalg.cross(nc, ref)
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pos[idx, 1]  = mid + 0.9 * perp
            amask[idx, 1] = True

        # ── C placement ──────────────────────────────────────────────────────
        n = amask[:, 0]; ca = amask[:, 1]; c = amask[:, 2]
        if L > 1:
            next_n[:-1]  = amask[1:, 0]
            next_ca[:-1] = amask[1:, 1]

        # has N & CA & next-N → extended-chain psi (approximate cone-sphere)
        m = todo & n & ca & ~c & next_n
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 2] = _place_fourth_atom_batch(
                pos[idx + 1, 0], pos[idx, 0], pos[idx, 1], cac, anca, ps)
            amask[idx, 2] = True

        # has N & CA, no next-N but has next-CA → approx from next CA
        c = amask[:, 2]
        if L > 1:
            next_n[:-1] = amask[1:, 0]
        m = todo & n & ca & ~c & ~next_n & next_ca
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 2] = _place_fourth_atom_batch(
                pos[idx + 1, 1], pos[idx, 0], pos[idx, 1], cac, anca, ps)
            amask[idx, 2] = True

        # Terminal/boundary: has N & CA, no C, no next-N, no next-CA
        # (last residue of chain or residue before a gap). Place C at
        # extended-chain psi using C(i-1) as dihedral reference.
        c = amask[:, 2]
        prev_c[:] = False
        if L > 1:
            prev_c[1:] = amask[:-1, 2]
        m = todo & n & ca & ~c & ~next_n & ~next_ca & prev_c
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            pos[idx, 2] = _place_fourth_atom_batch(
                pos[idx - 1, 2], pos[idx, 0], pos[idx, 1], cac, anca, ps)
            amask[idx, 2] = True

        if amask.equal(snap):   # nothing changed → converged
            break


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def place_missing_cb(atom_positions: torch.Tensor, mode: str = "correct") -> torch.Tensor:
    """
    Place ideal Cβ for all residues (overwrites atom index 3).

    mode="correct"  (default)
        CHARMM36 IC reconstruction: place_fourth_atom(N, C, CA) with
        |CA-CB|=1.522 Å, ∠C-CA-CB=111.09°, φ(N-C-CA-CB)=123.23°.
        Use this for FrustratometeR / any real-space distance calculation.

    mode="frame2seq"
        Original Frame2Seq/IPA implementation.  Use this when running the
        Frame2Seq / IPA-based encoder so coordinates stay in the expected
        local backbone frame.
    """
    N  = atom_positions[:, 0]
    CA = atom_positions[:, 1]
    C  = atom_positions[:, 2]

    if mode == "frame2seq":
        CB = _place_fourth_atom_frame2seq(
            N, CA, C,
            torch.tensor(1.522, dtype=atom_positions.dtype, device=atom_positions.device),
            torch.tensor(1.927, dtype=atom_positions.dtype, device=atom_positions.device),
            torch.tensor(-2.143, dtype=atom_positions.dtype, device=atom_positions.device),
        )
    else:
        CB = place_fourth_atom(N, C, CA, length=1.5220, angle_deg=111.09, dihedral_deg=123.23)

    CB = torch.where(torch.isnan(CB), torch.zeros_like(CB), CB)
    atom_positions = atom_positions.clone()
    existing_cb = atom_positions[:, 3]
    is_missing = existing_cb.norm(dim=-1, keepdim=True) < 1e-6
    atom_positions[:, 3] = torch.where(is_missing, CB, existing_cb)
    return atom_positions


def place_missing_o(atom_positions: torch.Tensor, mode: str = "correct") -> torch.Tensor:
    """
    Place ideal backbone carbonyl O for all residues (overwrites atom index 4).

    mode="correct"  (default)
        O is bonded to C: C-O = 1.231 Å, ∠CA-C-O = 120.8° (2.108 rad),
        lying in the N-CA-C backbone plane (planar trans peptide bond).
        Use this for FrustratometeR / any real-space distance calculation.

    mode="frame2seq"
        Original Frame2Seq/IPA implementation.  Use this when running the
        Frame2Seq / IPA-based encoder.
    """
    N  = atom_positions[:, 0]
    CA = atom_positions[:, 1]
    C  = atom_positions[:, 2]

    if mode == "frame2seq":
        O = _place_fourth_atom_frame2seq(
            N, CA, C,
            torch.tensor(1.231, dtype=atom_positions.dtype, device=atom_positions.device),
            torch.tensor(2.108, dtype=atom_positions.dtype, device=atom_positions.device),
            torch.tensor(-3.142, dtype=atom_positions.dtype, device=atom_positions.device),
        )
    else:
        # Correct real-space placement
        # O is at C, angle CA-C-O = 120.8°, in the backbone plane
        d_vec  = C - CA
        d_unit = d_vec / (d_vec.norm(dim=-1, keepdim=True) + 1e-8)   # CA→C

        n_plane = torch.cross(d_vec, N - CA, dim=-1)
        n_plane = n_plane / (n_plane.norm(dim=-1, keepdim=True) + 1e-8)

        # In-plane direction perpendicular to CA-C at C, pointing forward along chain
        perp = torch.cross(n_plane, d_unit, dim=-1)

        angle = torch.tensor(2.108, dtype=atom_positions.dtype, device=atom_positions.device)
        o_dir = torch.cos(angle) * (-d_unit) + torch.sin(angle) * perp
        O = C + 1.231 * o_dir

    O = torch.where(torch.isnan(O), torch.zeros_like(O), O)
    atom_positions = atom_positions.clone()
    atom_positions[:, 4] = O
    return atom_positions


# ---------------------------------------------------------------------------
# Backbone reconstruction — matches FrustratometeR's MissingAtoms.py behaviour
# ---------------------------------------------------------------------------

# Ideal peptide-bond geometry (Å / radians)
_C_N_BOND   = 1.329   # C(i)–N(i+1) peptide bond
_N_CA_BOND  = 1.459   # N–Cα
_CA_C_BOND  = 1.525   # Cα–C
_ANG_CA_C_N = 2.028   # ∠Cα–C–N  (116.2°)
_ANG_C_N_CA = 2.124   # ∠C–N–Cα  (121.7°)
_ANG_N_CA_C = 1.940   # ∠N–Cα–C  (111.2°)
_OMEGA      = 3.1416  # trans peptide dihedral ≈ π
_PSI_EXT    = 3.1416  # extended-chain psi ≈ 180° (used when psi is unknown)


def _dihedral_angle(A, B, C, D):
    """Signed dihedral A-B-C-D in radians (range [-π, π])."""
    import math as _math
    b1 = B - A; b2 = C - B; b3 = D - C
    n1 = torch.cross(b1, b2, dim=0); n1 = n1 / (n1.norm() + 1e-8)
    n2 = torch.cross(b2, b3, dim=0); n2 = n2 / (n2.norm() + 1e-8)
    b2u = b2 / (b2.norm() + 1e-8)
    m = torch.cross(n1, b2u, dim=0)
    return _math.atan2(m.dot(n2).item(), n1.dot(n2).item())


def _place_C_from_N_and_Nnext(
    N_i: torch.Tensor,    # [3]  N of residue i
    CA_i: torch.Tensor,   # [3]  CA of residue i
    N_next: torch.Tensor, # [3]  N of residue i+1
    dtype, device,
    C_prev:  torch.Tensor = None,  # [3]  C(i-1) — used for phi-dihedral selection
    CA_next: torch.Tensor = None,  # [3]  CA(i+1) — used for omega-dihedral selection
) -> torch.Tensor:
    """
    Geometric placement of C(i) given N(i), CA(i), and N(i+1).

    Finds C on the N(i)-CA(i)-C(i) angle cone (|CA-C|=CA_C_BOND,
    ∠N-CA-C=ANG_N_CA_C) satisfying |C-N(i+1)| = C_N_BOND.

    Constraint:
        C(φ) = center + r*(cosφ·e1 + sinφ·e2),  center = CA + R·cos(ang)·ax
        |C-N_next|² = C_N_BOND²  →  a·cosφ + b·sinφ = −c
        where c = (|A|²+r²−C_N_BOND²)/(2r), A = center−N_next
        → two solutions φ = atan2(b,a) ± acos(−c/|(a,b)|)

    Selection priority:
      1. Omega criterion (CA_next given): dihedral(CA_i, C, N_next, CA_next) ≈ ±180°
         for trans peptide (99.7% of all bonds). Best discriminant — depends on
         which side of the cone C lies.
      2. Phi criterion (C_prev given): dihedral(C_prev, N, CA, C) is negative for
         all standard conformations. Primary fallback.
      3. Trans-peptide score (neither given): provably identical for both solutions
         when CA-N_next is unusually short, so this is effectively a tie-break.
    """
    import math as _math

    cos_a = _math.cos(_ANG_N_CA_C)
    sin_a = _math.sin(_ANG_N_CA_C)
    R     = _CA_C_BOND

    ax = N_i - CA_i
    ax = ax / (ax.norm() + 1e-8)

    ref = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
    if ax.dot(ref).abs() > 0.9:
        ref = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
    e1 = torch.cross(ax, ref, dim=0); e1 = e1 / (e1.norm() + 1e-8)
    e2 = torch.cross(ax, e1, dim=0)

    center = CA_i + R * cos_a * ax
    r      = R * sin_a

    if r < 1e-8:
        return _place_fourth_atom_frame2seq(
            N_next, N_i, CA_i,
            torch.tensor(R,           dtype=dtype, device=device),
            torch.tensor(_ANG_N_CA_C, dtype=dtype, device=device),
            torch.tensor(_OMEGA,      dtype=dtype, device=device),
        )

    A  = center - N_next
    a  = A.dot(e1).item()
    b  = A.dot(e2).item()
    c  = (A.dot(A).item() + r * r - _C_N_BOND ** 2) / (2.0 * r)

    ab   = _math.sqrt(a * a + b * b + 1e-12)
    phi0 = _math.atan2(b, a)

    def _make_C(phi_val):
        return center + r * (_math.cos(phi_val) * e1 + _math.sin(phi_val) * e2)

    if abs(c) > ab:
        return _make_C(_math.atan2(-b, -a))  # no intersection: closest to N_next

    dphi = _math.acos(max(-1.0, min(1.0, -c / ab)))
    phi1 = phi0 + dphi
    phi2 = phi0 - dphi
    C1, C2 = _make_C(phi1), _make_C(phi2)

    if CA_next is not None:
        # Omega criterion: pick C with omega closest to ±180° (trans peptide)
        def _omega_err(Ci):
            omega = _dihedral_angle(CA_i, Ci, N_next, CA_next)
            return abs(abs(omega) - _math.pi)  # distance from 180°
        return C1 if _omega_err(C1) < _omega_err(C2) else C2

    if C_prev is not None:
        # Phi criterion: backbone phi is always negative
        _PHI_IDEAL = -2.094  # −120° in radians
        def _phi_score(Ci):
            phi_d = _dihedral_angle(C_prev, N_i, CA_i, Ci)
            dist = abs(phi_d - _PHI_IDEAL)
            if dist > _math.pi:
                dist = 2 * _math.pi - dist
            return dist + (0.0 if phi_d < 0 else 6.28)
        return C1 if _phi_score(C1) < _phi_score(C2) else C2

    # Last resort: trans score (may be identical for both — defaults to phi1)
    def _trans_score(Ci):
        cn = N_next - Ci; cn = cn / (cn.norm() + 1e-8)
        return (-ax).dot(cn).item()
    return C1 if _trans_score(C1) >= _trans_score(C2) else C2


def _place_C_select_by_cb_density(
    N_i, CA_i, N_next, dtype, device,
    all_pos,          # [L, ≥4, 3] — full atom positions for density counting
    res_idx: int,     # index of the residue being reconstructed
    L: int,
    C_prev=None,
    CA_next=None,
):
    """
    Compute both geometric C solutions (cone ∩ sphere), then pick the one
    whose ideal CB has more neighbours in the [4.5, 6.5 Å] density shell.

    This breaks the reflection-symmetry degeneracy that makes all dihedral
    criteria (omega, phi, trans-score) give identical scores: the two
    solutions land on opposite sides of the cone and their CBs point in
    opposite directions, so one typically has more density neighbours.

    Falls back to `_place_C_from_N_and_Nnext` (omega/phi criterion) when:
    - the cone does not intersect the N_next sphere (unique solution), OR
    - both candidate CBs have the same neighbour count.
    """
    import math as _math

    cos_a = _math.cos(_ANG_N_CA_C)
    sin_a = _math.sin(_ANG_N_CA_C)
    R     = _CA_C_BOND

    ax = N_i - CA_i; ax = ax / (ax.norm() + 1e-8)
    ref = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
    if ax.dot(ref).abs() > 0.9:
        ref = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
    e1 = torch.cross(ax, ref, dim=0); e1 = e1 / (e1.norm() + 1e-8)
    e2 = torch.cross(ax, e1, dim=0)

    center = CA_i + R * cos_a * ax
    r      = R * sin_a

    if r < 1e-8:
        return _place_C_from_N_and_Nnext(N_i, CA_i, N_next, dtype, device,
                                          C_prev=C_prev, CA_next=CA_next)

    A = center - N_next
    a = A.dot(e1).item(); b = A.dot(e2).item()
    c = (A.dot(A).item() + r * r - _C_N_BOND ** 2) / (2.0 * r)
    ab = _math.sqrt(a * a + b * b + 1e-12)

    def _mk(phi_v):
        return center + r * (_math.cos(phi_v) * e1 + _math.sin(phi_v) * e2)

    phi0 = _math.atan2(b, a)

    if abs(c) > ab:
        return _mk(_math.atan2(-b, -a))  # unique: closest point to N_next

    dphi = _math.acos(max(-1.0, min(1.0, -c / ab)))
    C1 = _mk(phi0 + dphi)
    C2 = _mk(phi0 - dphi)

    # Ideal CB for each candidate C: place_fourth_atom(N, C, CA, ...)
    def _ideal_cb(C_cand):
        cb = place_fourth_atom(
            N_i.unsqueeze(0), C_cand.unsqueeze(0), CA_i.unsqueeze(0),
            length=1.5220, angle_deg=111.09, dihedral_deg=123.23,
        ).squeeze(0)
        return torch.where(torch.isnan(cb), torch.zeros_like(cb), cb)

    cb1 = _ideal_cb(C1)
    cb2 = _ideal_cb(C2)

    # Count [4.5, 6.5 Å] density neighbours among existing PDB CB positions
    existing_cbs = all_pos[:, 3]          # [L, 3], zero where CB is absent
    seq_idx = torch.arange(L, dtype=torch.long, device=device)
    min_sep = (seq_idx - res_idx).abs() >= 3

    def _density_count(cb_cand):
        dists  = (existing_cbs - cb_cand).norm(dim=-1)
        has_cb = existing_cbs.norm(dim=-1) > 1e-6
        return ((dists >= 4.5) & (dists <= 6.5) & has_cb & min_sep).sum().item()

    n1 = _density_count(cb1)
    n2 = _density_count(cb2)

    if n1 != n2:
        return C1 if n1 > n2 else C2

    # Tie-break: fall back to dihedral criteria
    return _place_C_from_N_and_Nnext(N_i, CA_i, N_next, dtype, device,
                                      C_prev=C_prev, CA_next=CA_next)

def contact_preservation_loss(CA, ref_CA, contact_mask, cutoff=9.5):
    """
    Penalize breaking original contact topology.
    """

    d_ref = torch.cdist(ref_CA, ref_CA)
    d_new = torch.cdist(CA, CA)

    ref_contacts = d_ref < cutoff
    new_contacts = d_new < cutoff

    return (ref_contacts ^ new_contacts).float().mean()

def backbone_energy(C, N, CA, mask):
    """
    C:  [L, 3]
    N:  [L, 3]
    CA: [L, 3]
    """

    L = C.shape[0]

    # --- 1. bond continuity (CA-C-N_next)
    C_next = torch.roll(C, shifts=-1, dims=0)
    N_next = torch.roll(N, shifts=-1, dims=0)

    bond = ((C - N_next).norm(dim=-1) - 1.33)**2
    bond = bond * mask

    # --- 2. backbone smoothness (phi proxy)
    if L > 2:
        v1 = C[1:-1] - N[1:-1]
        v2 = CA[1:-1] - C[1:-1]
        v3 = N[2:] - CA[1:-1]

        n1 = torch.cross(v1, v2, dim=-1)
        n2 = torch.cross(v2, v3, dim=-1)

        n1 = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-8)
        n2 = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-8)

        smooth = (1 - (n1 * n2).sum(-1))
        smooth = torch.nn.functional.pad(smooth, (1,1))

    else:
        smooth = torch.zeros_like(bond)

    # --- 3. steric penalty (avoid weird flips)
    dists = torch.cdist(C, C)   # [L, L]
    eye   = torch.eye(L, dtype=torch.bool, device=C.device)
    steric = ((dists < 3.5) & ~eye).float().sum(dim=-1)   # [L]

    return bond + 0.3 * smooth + 0.1 * steric

def _compute_C_candidates(N_i, CA_i, N_next, dtype, device):
    """
    Return both cone-sphere solutions (C1, C2) for placing C(i) given N(i),
    CA(i), and N(i+1).  If there is no intersection (unique solution) both
    returned tensors are the same point.
    """
    import math as _math

    cos_a = _math.cos(_ANG_N_CA_C)
    sin_a = _math.sin(_ANG_N_CA_C)
    R     = _CA_C_BOND

    ax = N_i - CA_i; ax = ax / (ax.norm() + 1e-8)
    ref = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
    if ax.dot(ref).abs() > 0.9:
        ref = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
    e1 = torch.cross(ax, ref, dim=0); e1 = e1 / (e1.norm() + 1e-8)
    e2 = torch.cross(ax, e1, dim=0)

    center = CA_i + R * cos_a * ax
    r      = R * sin_a

    def _mk(phi_v):
        return center + r * (_math.cos(phi_v) * e1 + _math.sin(phi_v) * e2)

    A  = center - N_next
    a  = A.dot(e1).item(); b = A.dot(e2).item()
    c  = (A.dot(A).item() + r * r - _C_N_BOND ** 2) / (2.0 * r)
    ab = _math.sqrt(a * a + b * b + 1e-12)
    phi0 = _math.atan2(b, a)

    if abs(c) > ab:
        C_fallback = _mk(_math.atan2(-b, -a))
        return C_fallback, C_fallback

    dphi = _math.acos(max(-1.0, min(1.0, -c / ab)))
    return _mk(phi0 + dphi), _mk(phi0 - dphi)


def _iterative_backbone_refinement(pos, amask, orig_mask_3, dtype, device, n_iters=10):
    """
    Iterative N↔C refinement for reconstructed backbone residues.

    Each iteration:
      1. Re-place N(i) from the (now-updated) C(i-1) using ideal omega geometry.
      2. Compute both cone-sphere C candidates for every reconstructed-C residue.
      3. Call select_backbone once to globally minimize backbone energy over the
         binary C1/C2 choices.

    This breaks the cascade dependency (initial N was placed from approximate C)
    and propagates geometry improvements through CA-trace stretches.
    """
    L = pos.shape[0]
    if L < 2:
        return

    n_was_missing = ~orig_mask_3[:, 0]  # [L]
    c_was_missing = ~orig_mask_3[:, 2]  # [L]
    CA_ref = pos[:, 1].clone()          # original PDB CA positions, fixed

    def _s(v):
        return torch.tensor(v, dtype=dtype, device=device)

    prev_C = pos[:, 2].clone()

    for _ in range(n_iters):
        # ── 1. Re-place N from updated C(i-1) ────────────────────────────────
        for i in range(1, L):
            if (bool(n_was_missing[i]) and bool(amask[i, 1])
                    and bool(amask[i - 1, 1]) and bool(amask[i - 1, 2])):
                pos[i, 0] = _place_fourth_atom_frame2seq(
                    pos[i, 1], pos[i - 1, 1], pos[i - 1, 2],
                    _s(_C_N_BOND), _s(_ANG_CA_C_N), _s(_OMEGA),
                )

        # ── 2. Build C1, C2 for reconstructed-C residues ─────────────────────
        C1_all = pos[:, 2].clone()
        C2_all = pos[:, 2].clone()
        has_pair = False

        for i in range(L - 1):
            if (bool(c_was_missing[i]) and bool(amask[i, 0]) and bool(amask[i, 1])
                    and bool(amask[i + 1, 0])):
                c1, c2 = _compute_C_candidates(
                    pos[i, 0], pos[i, 1], pos[i + 1, 0], dtype, device)
                if (c1 - c2).norm().item() > 0.01:
                    C1_all[i] = c1
                    C2_all[i] = c2
                    has_pair = True

        if not has_pair:
            break

        # ── 3. Global C selection via backbone energy ─────────────────────────
        valid = amask[:, 0] & amask[:, 1] & amask[:, 2]
        mask_be = valid.clone()
        mask_be[-1] = False   # last residue has no N_next bond to constrain
        if mask_be.any():
            pos[:, 2] = select_backbone(
                C1_all, C2_all, pos[:, 0], pos[:, 1],
                mask_be, CA_ref, mask_be,
            )

        if (pos[:, 2] - prev_C).norm().item() < 1e-4:
            break
        prev_C = pos[:, 2].clone()


def select_backbone_old(C1, C2, N, CA, mask, iters=3):
    """
    C1, C2: [L, 3]
    returns: C [L, 3]
    """

    L = C1.shape[0]

    choice = torch.zeros(L, dtype=torch.bool, device=C1.device)  # False = C1

    def build_C(choice):
        return torch.where(choice[:, None], C2, C1)

    for _ in range(iters):

        C = build_C(choice)

        base_E = backbone_energy(C, N, CA, mask) + 2.0 * contact_preservation_loss(CA, CA_ref, contact_mask)

        for i in range(L):
            flipped = choice.clone()
            flipped[i] = ~flipped[i]

            C_flip = build_C(flipped)
            E_flip = backbone_energy(C_flip, N, CA, mask)

            # accept flip if improves global energy
            if E_flip.sum() < base_E.sum():
                choice[i] = ~choice[i]

    return build_C(choice)

def select_backbone(C1, C2, N, CA, mask,
                    CA_ref, contact_mask,
                    iters=3, lam=2.0):

    L = C1.shape[0]
    choice = torch.zeros(L, dtype=torch.bool, device=C1.device)

    def build_C(choice):
        return torch.where(choice[:, None], C2, C1)

    def total_energy(C):
        return backbone_energy(C, N, CA, mask).sum() + lam * contact_preservation_loss(CA, CA_ref, contact_mask)

    C = build_C(choice)
    base_E = total_energy(C)

    for _ in range(iters):

        for i in range(L):

            choice[i] = ~choice[i]
            C_new = build_C(choice)

            E_new = total_energy(C_new)

            # accept if improves GLOBAL energy
            if E_new < base_E:
                base_E = E_new
            else:
                choice[i] = ~choice[i]   # revert

    return build_C(choice)


def _write_ca_only_pdb(pdb_file: str, out_path: str) -> None:
    """Extract CA ATOM lines from *pdb_file* and write a CA-only PDB to *out_path*."""
    with open(pdb_file) as f_in, open(out_path, 'w') as f_out:
        serial = 1
        for line in f_in:
            rec = line[:4]
            if rec in ('ATOM', 'HETA') and len(line) >= 16:
                if line[12:16].strip() == 'CA':
                    f_out.write(f'ATOM  {serial:5d}{line[11:].rstrip()}\n')
                    serial += 1
            elif rec in ('TER ', 'ENDM', 'END '):
                f_out.write(line)
        f_out.write('END\n')


# Module-level cache: model loaded once, reused across all proteins.
def reconstruct_missing_backbone(
    atom_positions: torch.Tensor,   # [L, ≥5, 3]  raw coords, modified in-place clone
    atom_mask_3: torch.Tensor,      # [L, 3] bool  — N, CA, C present flags
    seq_mask: torch.Tensor,         # [L] bool      — updated for repaired residues
    pdb_file: str = None,
    backbone_method: str = 'ideal',
    chain_id: str = None,
    modeller_verbose: bool = True,
) -> tuple:
    """
    Fill missing backbone atoms (N, CA, C).

    backbone_method controls which reconstruction strategy is used:

      'auto'/'ideal' — Vectorized ideal geometry only.
      'modeller'     — MODELLER via complete_pdb(); requires an input file and
                       raises if MODELLER is unavailable or reconstruction fails.

    Residues where all three atoms are successfully placed have seq_mask set
    to True.  Returns (atom_positions_clone, seq_mask_updated).
    """
    if backbone_method not in ('auto', 'ideal', 'modeller'):
        raise ValueError(f"Unsupported backbone reconstruction method: {backbone_method!r}")
    if backbone_method == 'modeller' and pdb_file is None:
        raise ValueError("pdb_file is required for backbone_method='modeller'")

    dtype  = atom_positions.dtype
    device = atom_positions.device

    # ── 1. MODELLER ──────────────────────────────────────────────────────────
    # Note: when called from get_parsed_inputs with backbone_method='modeller',
    # the caller handles the full replacement (aatype, residue_index, etc.).
    # This block is retained for callers that only need the atom coordinates.
    if backbone_method == 'modeller':
        completed, _ = _reconstruct_with_modeller(pdb_file, chain_id, verbose=modeller_verbose)
        return completed, torch.ones(completed.shape[0], dtype=torch.bool, device=device)

    # ── 2. Ideal-geometry ────────────────────────────────────────────────────
    # Vectorized multi-pass reconstruction (fast for 10k+ scale).
    if backbone_method in ('ideal', 'auto'):
        pos   = atom_positions.clone()
        amask = atom_mask_3.clone()
        _vectorized_ideal_geometry(pos, amask, seq_mask, dtype, device)
        all_ok   = amask[:, 0] & amask[:, 1] & amask[:, 2]
        seq_mask = seq_mask | all_ok
        return pos, seq_mask

    # Legacy Sequential fallback (accurate cone-sphere C placement) implementation: unreachable with the validated methods above.
    L = atom_positions.shape[0]

    def _s(v):
        return torch.tensor(v, dtype=dtype, device=device)

    pos   = atom_positions.clone()
    amask = atom_mask_3.clone()

    for i in range(L):
        if seq_mask[i]:
            continue

        N_ok  = bool(amask[i, 0])
        CA_ok = bool(amask[i, 1])
        C_ok  = bool(amask[i, 2])

        # missing N — use C(i-1), CA(i), C(i)
        if not N_ok and CA_ok and C_ok and i > 0 and amask[i-1, 2]:
            pos[i, 0] = _place_fourth_atom_frame2seq(
                pos[i, 2], pos[i, 1], pos[i-1, 2],
                _s(_C_N_BOND), _s(_ANG_CA_C_N), _s(_OMEGA),
            )
            amask[i, 0] = True
            N_ok = True

        # missing C — solve for C(i) using the N(i+1) peptide-bond constraint
        if not C_ok and N_ok and CA_ok and i < L - 1 and amask[i+1, 0]:
            C_prev_i  = pos[i-1, 2] if (i > 0 and bool(amask[i-1, 2])) else None
            CA_next_i = pos[i+1, 1] if bool(amask[i+1, 1]) else None
            pos[i, 2] = _place_C_from_N_and_Nnext(
                pos[i, 0], pos[i, 1], pos[i+1, 0], dtype, device,
                C_prev=C_prev_i, CA_next=CA_next_i,
            )
            amask[i, 2] = True
            C_ok = True

        # missing CA — midpoint of N and C + small perpendicular offset
        if not CA_ok and N_ok and C_ok:
            mid    = 0.5 * (pos[i, 0] + pos[i, 2])
            nc_vec = pos[i, 2] - pos[i, 0]
            ref    = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            if abs(torch.dot(nc_vec / (nc_vec.norm() + 1e-8), ref)) > 0.9:
                ref = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
            perp   = torch.cross(nc_vec, ref, dim=0)
            perp   = perp / (perp.norm() + 1e-8)
            pos[i, 1]   = mid + 0.9 * perp
            amask[i, 1] = True
            CA_ok = True

        # Only CA known (CA-trace region, common in low-resolution structures).
        # Cascade: place N from the left C-anchor using omega=180°, then place C
        # using N(i+1) exactly (if available) or CA(i+1) with extended psi.
        # The loop processes residues left-to-right, so each placed C enables the
        # next residue's N placement — the whole CA-trace is rebuilt iteratively.
        if not N_ok and CA_ok and not C_ok:
            if i > 0 and bool(amask[i-1, 2]) and bool(amask[i-1, 1]):
                pos[i, 0] = _place_fourth_atom_frame2seq(
                    pos[i, 1],      # a = CA(i)   [dihedral reference]
                    pos[i-1, 1],    # b = CA(i-1) [angle reference]
                    pos[i-1, 2],    # c = C(i-1)  [bond origin]
                    _s(_C_N_BOND), _s(_ANG_CA_C_N), _s(_OMEGA),
                )
                amask[i, 0] = True
                N_ok = True
            if N_ok:
                if i < L - 1 and bool(amask[i+1, 0]):
                    # Exact constraint: N(i+1) known
                    C_prev_i  = pos[i-1, 2] if (i > 0 and bool(amask[i-1, 2])) else None
                    CA_next_i = pos[i+1, 1] if bool(amask[i+1, 1]) else None
                    pos[i, 2] = _place_C_from_N_and_Nnext(
                        pos[i, 0], pos[i, 1], pos[i+1, 0], dtype, device,
                        C_prev=C_prev_i, CA_next=CA_next_i,
                    )
                    amask[i, 2] = True
                    C_ok = True
                elif i < L - 1 and bool(amask[i+1, 1]):
                    # Approximate: use extended-chain psi toward CA(i+1)
                    pos[i, 2] = _place_fourth_atom_frame2seq(
                        pos[i+1, 1],   # a = CA(i+1) [psi reference]
                        pos[i, 0],     # b = N(i)
                        pos[i, 1],     # c = CA(i)
                        _s(_CA_C_BOND), _s(_ANG_N_CA_C), _s(_PSI_EXT),
                    )
                    amask[i, 2] = True
                    C_ok = True

        # No backbone atoms at all: place N from N(i-1)-CA(i-1)-C(i-1) frame using
        # extended-chain psi (180°), then fall through to "N only" to place CA and C.
        if not N_ok and not CA_ok and not C_ok:
            if i > 0 and amask[i-1, 0] and amask[i-1, 1] and amask[i-1, 2]:
                pos[i, 0] = _place_fourth_atom_frame2seq(
                    pos[i-1, 0], pos[i-1, 1], pos[i-1, 2],
                    _s(_C_N_BOND), _s(_ANG_CA_C_N), _s(_PSI_EXT),
                )
                amask[i, 0] = True
                N_ok = True

        # Only N known: build CA from CA(i-1)-C(i-1)-N(i) frame (omega), then C from N(i+1).
        if N_ok and not CA_ok and not C_ok:
            if i > 0 and amask[i-1, 1] and amask[i-1, 2]:
                pos[i, 1] = _place_fourth_atom_frame2seq(
                    pos[i-1, 1], pos[i-1, 2], pos[i, 0],
                    _s(_N_CA_BOND), _s(_ANG_C_N_CA), _s(_OMEGA),
                )
                amask[i, 1] = True
                CA_ok = True
            if CA_ok and i < L - 1 and amask[i+1, 0]:
                C_prev_i  = pos[i-1, 2] if (i > 0 and bool(amask[i-1, 2])) else None
                CA_next_i = pos[i+1, 1] if bool(amask[i+1, 1]) else None
                pos[i, 2] = _place_C_from_N_and_Nnext(
                    pos[i, 0], pos[i, 1], pos[i+1, 0], dtype, device,
                    C_prev=C_prev_i, CA_next=CA_next_i,
                )
                amask[i, 2] = True
                C_ok = True

        # Only C known: build CA from CA(i+1)-N(i+1)-C(i) frame (omega), then N from C(i-1).
        if C_ok and not N_ok and not CA_ok:
            if i < L - 1 and amask[i+1, 0] and amask[i+1, 1]:
                pos[i, 1] = _place_fourth_atom_frame2seq(
                    pos[i+1, 1], pos[i+1, 0], pos[i, 2],
                    _s(_CA_C_BOND), _s(_ANG_CA_C_N), _s(_OMEGA),
                )
                amask[i, 1] = True
                CA_ok = True
            if CA_ok and i > 0 and amask[i-1, 2]:
                pos[i, 0] = _place_fourth_atom_frame2seq(
                    pos[i, 2], pos[i, 1], pos[i-1, 2],
                    _s(_C_N_BOND), _s(_ANG_CA_C_N), _s(_OMEGA),
                )
                amask[i, 0] = True
                N_ok = True

        # Terminal/boundary fallback: N and CA present but C still missing and
        # no N_next available (last residue of chain, or residue before a gap).
        # FrustratometeR (via MODELLER) reconstructs these; we use extended-chain
        # psi with C(i-1) as dihedral reference so the residue enters the pair set.
        if not C_ok and N_ok and CA_ok:
            if i > 0 and bool(amask[i-1, 2]):
                pos[i, 2] = _place_fourth_atom_frame2seq(
                    pos[i-1, 2],    # a = C(i-1)  [psi dihedral reference]
                    pos[i, 0],      # b = N(i)
                    pos[i, 1],      # c = CA(i)
                    _s(_CA_C_BOND), _s(_ANG_N_CA_C), _s(_PSI_EXT),
                )
                amask[i, 2] = True
                C_ok = True

        # Mark successfully reconstructed residues as valid so they participate
        # in pair calculations, matching MODELLER's behaviour in FrustratometeR.
        if N_ok and CA_ok and C_ok:
            seq_mask[i] = True

    return pos, seq_mask


def _reconstruct_with_modeller(pdb_file: str, chain_id: str = None,
                               verbose: bool = True):
    """
    Use MODELLER's complete_pdb() to reconstruct missing atoms, matching
    FrustratometeR's MissingAtoms.py exactly.

    Returns (atom_positions_tensor [L, 37, 3], prot_from_pdb) where the protein
    object carries aatype, residue_index, chain_index, hetatm etc. derived from
    the MODELLER-completed structure. Raises ImportError if MODELLER is unavailable.
    """
    import tempfile, os, sys

    # Try plain import first; fall back to MODELLER_DIR env var if set.
    try:
        from modeller import environ as modeller_environ
        from modeller.scripts import complete_pdb
    except ImportError:
        _base = os.environ.get("MODELLER_DIR", "")
        if _base:
            for _p in [
                os.path.join(_base, "bin", "lib", "x86_64-intel8", "python3.3"),
                os.path.join(_base, "modlib"),
            ]:
                if os.path.isdir(_p) and _p not in sys.path:
                    sys.path.insert(0, _p)
        try:
            from modeller import environ as modeller_environ
            from modeller.scripts import complete_pdb
        except ImportError as exc:
            raise ImportError(
                "MODELLER is not installed or not found. "
                "Set MODELLER_DIR to the MODELLER installation directory, "
                "or explicitly use backbone_method='ideal' instead."
            ) from exc

    print(f"[modeller] reconstructing missing backbone atoms for {pdb_file}")

    if not verbose:
        try:
            import modeller as _modeller_pkg
            _modeller_pkg.log.none()
        except Exception:
            pass

    import time as _time
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "completed.pdb")
        env = modeller_environ()
        env.libs.topology.read("${LIB}/top_heav.lib")
        env.libs.parameters.read("${LIB}/par.lib")
        _t0 = _time.perf_counter()
        m = complete_pdb(env, pdb_file, transfer_res_num=True)
        m.write(file=out_file)
        print(f"[modeller] done in {_time.perf_counter() - _t0:.1f}s")

        from . import protein
        prot_from_pdb = protein.from_pdb_biotite(out_file, chain_id=chain_id)
        return torch.from_numpy(prot_from_pdb.atom_positions).double(), prot_from_pdb

