"""
T1 + T2 of the exact data-consistency verification plan.

    conda activate dbt-dolce
    python tests/test_projection.py [device]      # standalone (prints PASS/FAIL)
    pytest tests/test_projection.py               # or via pytest

T0 -- LEAP output-buffer freshness. DOLCE's vendored LEAP_torch returns its
one persistent internal buffer from every call; DBTProjector clones its
outputs to compensate. T0 reports the raw behaviour and gates the wrapper.

T1 -- adjointness (adjoint-gate measurements). The dual-CG projection is only the
exact Euclidean projection if the adjoint used is the true adjoint of
`forward`. The existing smoke test (tests/test_projector.py) certifies 5%
adjointness, which is fine for SIRT and useless for CG. Here we measure it
properly; T1a/T1b/T1c failing selects the next rung of the adjoint-gate ladder
instead of failing the suite:
  T1a  randomized <Ax,y> vs <x,A^Ty> at the full 512^2 geometry, gate 1e-4
  T1b  dense ||B - A^T||_F / ||A^T||_F at a small geometry (definitive),
       plus the best-global-scale diagnostic (is B just c*A^T?)
  T1c  the autograd-VJP adjoint against the same dense A^T, and whether it
       is distinct from `backward` at all (if LEAP registers the
       backprojector as forward's gradient, VJP adds nothing and the T1 gate
       decision must say so)
  T1d  the materialized sparse A (ladder rung 3) reproduces
       the live forward on random images; its literal transpose is A^T by
       construction.
       THE ADJOINT GATE IS PER-BUILD (measured 2026-07-03): the ORIGINAL
       dbt-dolce LEAP build fails T1a/b/c (backward ~20% from A^T, VJP == backward
       bitwise) -> materialization is the operative rung; a FRESH build
       from leap/src (setup.sh) measures T1b = 0 bitwise -- an exactly
       matched adjoint. The adjoint mismatch was a build artifact, so this
       suite must be re-run per environment; the production npz stays the
       operator of record for the study either way (self-consistent, and
       the cached direct solve is ~20x faster than the live prox, T7).

T2 -- projection correctness against a dense pseudo-inverse reference.
  T2a  pure-torch fp64 problem (no LEAP needed, always runs): dual_cg_project
       vs numpy.linalg.pinv on a rank-deficient A with a consistent y,
       machine-precision gate. Also verifies, on the same reference problem,
       properties of the projection: consistency A Pi(x) = y,
       idempotence, the fixed point Pi(x*) = x*, the Pythagoras identity,
       the gamma-damping identity, and warm-start
       equivalence.
  T2b  the same reference comparison through the live LEAP pair at a small
       geometry (A materialised column-by-column, reference in fp64).
       Skips itself when the live backward fails adjointness (its
       precondition) -- T2c is the trusted-path replacement.
  T2c  the same protocol through the MATERIALIZED pair (sparse A + literal
       transpose), covering both the dual-CG solve and the cached
       Gram-eigendecomposition direct solve.

T3 -- fixed point & consistency AT SCALE, through the operator of record
     (production runs/operator/A_25deg_512.npz if present, else a freshly
     materialized 32^2 twin): Pi(x*) = x* for a consistent GT-like slice
     (real val slice when the processed dataset is reachable, else a smooth
     phantom), ||A Pi(x_hat) - y||/||y|| at solver precision for random
     x_hat, and cold-started idempotence.

T4 -- orthogonal split at scale: the correction Pi(x)-x lies in range(A^T)
     and the residual-to-GT error Pi(x)-x* in null(A); measured as the
     normalized inner product, A-annihilation of the error, and the
     Pythagoras identity.

T5 -- the rho-prox convergence numerically (LEAP-free, dense fp64): the rho-prox
     argmin ||Ax-y||^2 + rho||x-x_hat||^2 converges to Pi_S(x_hat) as
     rho -> 0 -- via the exact normal-equations solve over a rho ladder
     (monotone convergence) and via the same CG algorithm prox_cgrad
     deploys, run with the TRUE adjoint. (Through live LEAP the deployed
     prox uses `backward` != A^T -- adjoint mismatch measured -- so the limit
     statement is about the algorithm; the mismatch is the prox arm's own
     baggage.)

T6 -- ensemble structure: K projected random images have
     identical sinograms (spread annihilated on range(A^T) functionals,
     nonzero on null-space functionals), and the k-space variance across
     the ensemble concentrates in the missing wedge -- quantified as the
     z-wedge / x-wedge variance ratio and saved as a qualitative panel
     (results/verification/t6_kspace_variance.png).

T7 -- spectrum & solver economics: eigh spectrum of the dual Gram
     (sigma1^2, effective rank, condition number of the kept block)
     cross-checked by power iteration; detection of the predicted per-view
     mass near-null vectors; wall-clock of the
     cached direct projection vs dual CG (cold + warm), and vs the
     historical rho-prox when LEAP is available.

T1a/T1b/T1c/T2b skip when LEAP is not installed (same policy as
tests/test_projector.py). T2a and T5 run anywhere torch + numpy exist.
T3/T4/T6/T7 run wherever a materialized operator is obtainable (production
npz, or LEAP to build the small twin) and skip cleanly otherwise.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG = str(ROOT / "configs" / "leap_dbt_25deg.cfg")

from physics.dbt_projector import dual_cg_project  # noqa: E402  (LEAP-free)

try:
    from physics.dbt_projector import (build_projector, MaterializedOperator,
                                       _LEAP_AVAILABLE)
except Exception as e:  # pragma: no cover
    print(f"[SKIP] could not import projector: {e}")
    _LEAP_AVAILABLE = False


class SkipCheck(Exception):
    """A check whose precondition failed (not a pass, not a failure)."""


# T2a -- dense fp64 reference problem (LEAP-free)

def _dense_problem(m=36, n=144, seed=0):
    """
    Random wide A with *explicitly redundant rows*, mimicking the real
    geometry's per-view mass redundancy: AA^T is
    singular, so this exercises exactly the CG-on-a-consistent-singular-
    system regime the projection relies on. y = A x* is consistent by
    construction.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, n, generator=g, dtype=torch.float64)
    A[-1] = A[0] + 0.5 * A[1]
    A[-2] = A[2] - A[3]
    x_star = torch.rand(n, generator=g, dtype=torch.float64)
    y = A @ x_star
    x_hat = torch.rand(n, generator=g, dtype=torch.float64)
    return A, x_star, y, x_hat


def _pinv_project(A: torch.Tensor, x_hat: torch.Tensor, y: torch.Tensor):
    """Reference Pi_S(x_hat) = x_hat - A^T pinv(AA^T)(A x_hat - y), fp64."""
    An, xh, yn = A.numpy(), x_hat.numpy(), y.numpy()
    lam = np.linalg.pinv(An @ An.T) @ (An @ xh - yn)
    return torch.from_numpy(xh - An.T @ lam)


def _project(A, x_hat, y, **kw):
    kw.setdefault("num_iters", 500)
    kw.setdefault("tol", 1e-14)
    return dual_cg_project(lambda v: A @ v, lambda s: A.T @ s, x_hat, y, **kw)


def check_t2_dense_reference(tol=1e-9):
    """dual_cg_project == pinv reference at machine precision (fp64)."""
    A, x_star, y, x_hat = _dense_problem()
    x_ref = _pinv_project(A, x_hat, y)
    x_cg, _, info = _project(A, x_hat, y)
    rel = float((x_cg - x_ref).norm() / x_ref.norm())
    print(f"  T2a reference: rel_err={rel:.3e}  "
          f"(CG iters={info['n_iters']}, data_res={info['data_rel_residual']:.1e})")
    assert rel < tol, f"projection differs from pinv reference: {rel:.3e} > {tol}"
    return True


def check_t2_dense_properties(tol=1e-9):
    """Consistency, idempotence, fixed point, Pythagoras + gamma-damping identities."""
    A, x_star, y, x_hat = _dense_problem()
    x_p, lam, _ = _project(A, x_hat, y)

    # consistency: A Pi(x_hat) = y
    res = float((A @ x_p - y).norm() / y.norm())
    assert res < tol, f"consistency ||A Pi(x)-y||/||y||={res:.3e}"

    # idempotence: Pi(Pi(x_hat)) = Pi(x_hat)   (cold start)
    x_pp, _, _ = _project(A, x_p, y)
    idem = float((x_pp - x_p).norm() / x_p.norm())
    assert idem < tol, f"idempotence violated: {idem:.3e}"

    # fixed point: Pi(x*) = x*  (GT is consistent)
    x_fp, _, _ = _project(A, x_star, y)
    fp = float((x_fp - x_star).norm() / x_star.norm())
    assert fp < tol, f"GT not a fixed point: {fp:.3e}"

    # Pythagoras: ||x-x*||^2 = ||Pi(x)-x*||^2 + ||x-Pi(x)||^2
    lhs = float((x_hat - x_star).norm() ** 2)
    rhs = float((x_p - x_star).norm() ** 2 + (x_hat - x_p).norm() ** 2)
    pyth = abs(lhs - rhs) / lhs
    assert pyth < tol, f"Pythagoras identity violated: {pyth:.3e}"
    assert (x_p - x_star).norm() <= (x_hat - x_star).norm(), "no-harm violated"

    # gamma damping: residual scales by (1-gamma) and the error
    # splits as ||P_N e||^2 + (1-g)^2 ||P_R e||^2
    gamma = 0.5
    x_g, _, _ = _project(A, x_hat, y, gamma=gamma)
    r_full = float((A @ x_hat - y).norm())
    r_g = float((A @ x_g - y).norm())
    assert abs(r_g - (1 - gamma) * r_full) / r_full < tol, \
        f"gamma residual scaling violated: {r_g:.3e} vs {(1-gamma)*r_full:.3e}"
    P_R = torch.from_numpy(np.linalg.pinv(A.numpy()) @ A.numpy())
    e = x_hat - x_star
    err_pred = float((e - P_R @ e).norm() ** 2
                     + (1 - gamma) ** 2 * (P_R @ e).norm() ** 2)
    err_obs = float((x_g - x_star).norm() ** 2)
    assert abs(err_obs - err_pred) / err_pred < 1e-8, \
        f"gamma-damping error split violated: {err_obs:.6e} vs {err_pred:.6e}"

    # warm start: projecting a perturbed x_hat from the previous dual solution
    # gives the same primal as a cold start
    x_hat2 = x_hat + 0.01 * torch.randn(
        x_hat.shape, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64)
    x_cold, _, _ = _project(A, x_hat2, y)
    x_warm, _, _ = _project(A, x_hat2, y, lambda0=lam)
    warm = float((x_warm - x_cold).norm() / x_cold.norm())
    assert warm < tol, f"warm start changes the primal solution: {warm:.3e}"

    print(f"  T2a properties: consistency={res:.1e}  idempotence={idem:.1e}  "
          f"fixed_point={fp:.1e}  thm1={pyth:.1e}  warm_start={warm:.1e}")
    return True


# LEAP helpers (small geometry, dense materialisation)

def _small_cfg_file(size=32) -> str:
    """Write a scaled-down copy of the 25-degree geometry (9 views, size^2)."""
    text = "\n".join([
        f"img_dimx = {size}", f"img_dimy = {size}", "img_dimz = 1",
        "img_pwidth = 0.273", "img_pheight = 0.273",
        "img_offsetx = 0", "img_offsety = 0", "img_offsetz = 0",
        "proj_geometry = parallel", "proj_arange = 25", "proj_nangles = 9",
        "proj_nrows = 1", f"proj_ncols = {size}",
        "proj_pheight = 0.273", "proj_pwidth = 0.273",
        "proj_crow = 0", f"proj_ccol = {(size - 1) / 2}",
        "proj_phis =", "proj_sod = 0", "proj_sdd = 0", "",
    ])
    f = tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False)
    f.write(text)
    f.close()
    return f.name


def _materialize_forward(proj, size, device):
    """A as a dense (m, n) matrix, one forward call per unit image."""
    n = size * size
    cols = []
    for j in range(n):
        e = torch.zeros(size, size, device=device)
        e.view(-1)[j] = 1.0
        cols.append(proj.forward(e).reshape(-1))
    return torch.stack(cols, dim=1)          # (m, n)


def _materialize_backward(proj, size, device, adjoint_fn=None):
    """B (or any sino->image map) as a dense (n, m) matrix."""
    fn = adjoint_fn if adjoint_fn is not None else proj.backward
    m = proj.num_angles * size
    cols = []
    for i in range(m):
        e = torch.zeros(proj.num_angles, size, device=device)
        e.view(-1)[i] = 1.0
        cols.append(fn(e).reshape(-1))
    return torch.stack(cols, dim=1)          # (n, m)


def _disk(size, radius_frac=0.3):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size),
                            torch.linspace(-1, 1, size), indexing="ij")
    return ((xx ** 2 + yy ** 2).sqrt() < radius_frac).float()


# T0 -- LEAP output-buffer freshness (LEAP required)

def check_t0_buffer_freshness(device="cpu", size=32):
    """
    DOLCE's vendored LEAP_torch writes into (and returns) ONE persistent
    internal buffer per direction, so raw projector outputs are silently
    overwritten by the next call. DBTProjector.forward/backward clone their
    outputs to make that safe; this check (a) reports which behaviour the
    installed LEAP build has, and (b) gates that the wrapper's outputs really
    are independent -- CG, the dense materialisations in T1/T2b, and
    preprocess.py's sino->SIRT handoff all hold outputs across calls.
    """
    proj = build_projector(_small_cfg_file(size), device=device)
    x1 = _disk(size).to(device)
    x2 = torch.zeros(size, size, device=device)

    # (a) raw LEAP behaviour -- informational, not gated
    raw = proj._proj(x1[None, None].contiguous(), "forward")
    snap = raw.clone()
    proj._proj(x2[None, None].contiguous(), "forward")
    if torch.equal(raw, snap):
        print("  T0 raw LEAP forward returns an independent tensor per call")
    else:
        print("  T0 raw LEAP forward returns its SHARED internal buffer "
              "(later calls overwrite earlier results)")
        print("     NOTE: on this build, any dataset generated with a "
              "pre-clone DBTProjector had its sinogram zeroed inside sirt() "
              "-- re-inspect stored SIRT conditioning (scripts/inspect_h5.py).")

    # (b) wrapper guarantee -- gated
    s1 = proj.forward(x1)
    keep_s = s1.clone()
    proj.forward(x2)
    assert torch.equal(s1, keep_s), \
        "DBTProjector.forward output was mutated by a later call"
    b1 = proj.backward(torch.ones(proj.num_angles, size, device=device))
    keep_b = b1.clone()
    proj.backward(torch.zeros(proj.num_angles, size, device=device))
    assert torch.equal(b1, keep_b), \
        "DBTProjector.backward output was mutated by a later call"
    return True


# T1 -- adjointness (LEAP required)

def check_t1_adjoint_fullscale(device="cpu", n_pairs=10, tol=1e-4):
    """Randomized adjoint identity at the production 512^2 geometry."""
    proj = build_projector(CFG, device=device)
    S, A = proj.img_size, proj.num_angles
    torch.manual_seed(0)
    errs = []
    for _ in range(n_pairs):
        x = torch.randn(S, S, device=device)
        y = torch.randn(A, S, device=device)
        lhs = (proj.forward(x) * y).sum().item()
        rhs = (x * proj.backward(y)).sum().item()
        errs.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    worst = max(errs)
    print(f"  T1a adjoint @512^2: max rel_err={worst:.3e} over {n_pairs} pairs")
    assert worst < tol, (
        f"adjoint mismatch {worst:.3e} > {tol}: LEAP backward is not A^T at "
        f"CG precision. Switch project_consistent to "
        f"adjoint='vjp' and re-run this suite on that pair.")
    return True


def check_t1_dense_small(device="cpu", size=32, tol=1e-4):
    """Definitive: ||B - A^T||_F / ||A^T||_F on densely materialised operators."""
    proj = build_projector(_small_cfg_file(size), device=device)
    A = _materialize_forward(proj, size, device)
    B = _materialize_backward(proj, size, device)
    rel = float((B - A.T).norm() / A.T.norm())
    # Diagnostic: is the mismatch a pure global scale (B ~ c A^T)? If the
    # residual after optimal rescaling were tiny, c*backward would be a
    # matched adjoint and materialization would be unnecessary.
    c = float((B * A.T).sum() / (A.T * A.T).sum())
    rel_c = float((B - c * A.T).norm() / A.T.norm())
    print(f"  T1b dense adjoint @{size}^2: ||B - A^T||_F/||A^T||_F = {rel:.3e}; "
          f"best scale c*={c:.6f} leaves {rel_c:.3e}")
    assert rel < tol, (
        f"dense adjoint mismatch {rel:.3e} > {tol} (and {rel_c:.3e} after "
        f"optimal rescaling): the adjoint gate escalates past 'backward'.")
    return True


def check_t1_vjp_small(device="cpu", size=32, tol=1e-4):
    """
    The autograd-VJP adjoint against the dense A^T, and a diagnosis of
    whether it is actually distinct from `backward` (if LEAP wires the
    backprojector in as forward's gradient, VJP is not an independent
    fallback and the adjoint gate must know that).
    """
    proj = build_projector(_small_cfg_file(size), device=device)
    A = _materialize_forward(proj, size, device)
    B = _materialize_backward(proj, size, device)
    V = _materialize_backward(proj, size, device, adjoint_fn=proj.adjoint_vjp)
    rel_v = float((V - A.T).norm() / A.T.norm())
    rel_vb = float((V - B).norm() / max(float(B.norm()), 1e-30))
    same = "VJP == backward (LEAP delegates its gradient to the backprojector)" \
        if rel_vb < 1e-6 else "VJP is an independent adjoint implementation"
    print(f"  T1c vjp adjoint @{size}^2: ||V - A^T||_F/||A^T||_F = {rel_v:.3e}; "
          f"||V - B||/||B|| = {rel_vb:.3e} -> {same}")
    assert rel_v < tol, (
        f"VJP adjoint mismatch {rel_v:.3e} > {tol}. If T1b also failed, no "
        f"trusted adjoint exists and the projection must not be used until "
        f"one is implemented.")
    return True


def check_t1d_materialized(device="cpu", size=32, tol=1e-5):
    """
    Adjoint-gate fallback rung 3: the materialized sparse A
    reproduces the live LEAP forward on random images (validates linearity
    of the live forward and faithful assembly). Its adjoint is the literal
    transpose, exact by construction -- no separate adjoint gate needed.
    """
    proj = build_projector(_small_cfg_file(size), device=device)
    M = MaterializedOperator.from_projector(proj, batch_size=64, verbose=False)
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(5):
        x = torch.rand(size, size, device=device)
        ref = proj.forward(x)
        worst = max(worst, float((M.forward(x) - ref).norm() / ref.norm()))
    print(f"  T1d materialized-vs-live forward @{size}^2: max rel_err={worst:.3e}")
    assert worst < tol, (
        f"materialized A does not reproduce the live forward ({worst:.3e} > "
        f"{tol}): live forward nonlinear or assembly bug; do not use.")
    return True


# T2b -- projection vs dense reference through the real LEAP operator

def check_t2_leap_small(device="cpu", size=32, tol_ref=1e-3, tol_res=1e-4):
    proj = build_projector(_small_cfg_file(size), device=device)
    A64 = _materialize_forward(proj, size, device).to(torch.float64).cpu()

    # Precondition 1: the live (forward, backward) pair is only a valid
    # projection pair if backward passed T1b. If it did not, this check is
    # meaningless -- the trusted path is T2c.
    B = _materialize_backward(proj, size, device)
    A32 = A64.to(torch.float32).to(device)
    pair_mismatch = float((B - A32.T).norm() / A32.T.norm())
    if pair_mismatch > 1e-4:
        raise SkipCheck(
            f"live backward fails adjointness ({pair_mismatch:.3e} > 1e-4, "
            f"see T1b); projection through the live pair is disallowed -- "
            f"T2c covers the materialized pair instead.")

    # Precondition 2 (achievability): the live path runs fp32 CG, whose
    # error floor is ~cond(kept Gram) * eps_fp32. The 32^2 twin measured
    # cond ~ 1e10 on the leap/src build (vs 1.1e6 at the production 512^2),
    # where no fp32 iteration count can reach tol_ref -- a property of the
    # small geometry, not of the projection code (which T2c's direct solve
    # certifies at ~1e-9 through the same A).
    w = np.linalg.eigvalsh((A64 @ A64.T).numpy())
    kept = w > 1e-12 * max(w[-1], 1e-300)
    kappa = float(w[-1] / w[kept].min())
    floor32 = kappa * 1.2e-7
    if floor32 > tol_ref:
        raise SkipCheck(
            f"fp32 CG cannot reach tol_ref={tol_ref:.0e} at this geometry's "
            f"conditioning (cond(kept)={kappa:.2e} -> fp32 floor ~{floor32:.0e}); "
            f"correctness through this A is certified by T2c's direct solve.")

    x_star = _disk(size).to(device)
    y = proj.forward(x_star)
    torch.manual_seed(0)
    x_hat = torch.rand(size, size, device=device)

    # fp64 dense reference through the materialised operator
    x_ref = _pinv_project(
        A64, x_hat.reshape(-1).to(torch.float64).cpu(),
        y.reshape(-1).to(torch.float64).cpu()).reshape(size, size)

    # the actual implementation: live LEAP closures, fp32
    x_cg, _, info = proj.project_consistent(x_hat, y, num_iters=600, tol=1e-7)

    rel = float((x_cg.cpu().to(torch.float64) - x_ref).norm() / x_ref.norm())
    res = proj.data_residual(x_cg, y) / 100.0          # percent -> fraction
    fp, _, _ = proj.project_consistent(x_star, y, num_iters=600, tol=1e-7)
    fperr = float((fp - x_star).norm() / x_star.norm())
    print(f"  T2b LEAP @{size}^2: rel_err_vs_pinv={rel:.3e}  "
          f"data_residual={res:.3e}  fixed_point={fperr:.3e}  "
          f"(CG iters={info['n_iters']})")
    assert rel < tol_ref, f"LEAP projection differs from reference: {rel:.3e}"
    assert res < tol_res, f"projected image not consistent: residual {res:.3e}"
    assert fperr < 1e-3, f"GT not a fixed point through LEAP: {fperr:.3e}"
    return True


# T2c -- projection vs dense reference through the MATERIALIZED operator

def check_t2c_materialized(device="cpu", size=32,
                           tol_dir=1e-6, tol_cg=1e-4, tol_res=1e-7):
    """
    T2b's protocol through the materialized (A, literal A^T) pair -- the
    operator the exact arm will actually use after the adjoint gate. Covers both solve
    paths: the cached Gram-pinv direct solve (the production path) and
    dual CG (redundancy cross-check).

    Everything runs in the operator's fp64 compute dtype: the dual Gram has
    cond ~ 1e6 (measured), which amplifies fp32 roundoff to ~1e-1 -- the
    first cluster run measured exactly that (CG rel 1.7e-1, direct 5.1e-3),
    which is a precision ceiling, not an operator or formula error. In fp64
    the same amplification leaves ~1e-10 headroom, and CG on the m = 9*size
    dual system additionally has finite termination.
    """
    proj = build_projector(_small_cfg_file(size), device=device)
    M = MaterializedOperator.from_projector(proj, batch_size=64, verbose=False)
    A64 = torch.from_numpy(M.A_sp.toarray().astype(np.float64))

    x_star = _disk(size).to(device)
    y = M.forward(x_star)                        # fp64 (operator dtype)
    torch.manual_seed(0)
    x_hat = torch.rand(size, size, device=device, dtype=torch.float64)

    x_ref = _pinv_project(
        A64, x_hat.reshape(-1).cpu(),
        y.reshape(-1).cpu()).reshape(size, size)

    finfo = M.factorize(verbose=False)
    kappa = float(finfo["cond_kept"])
    x_dir, _, dinfo = M.project(x_hat, y)
    x_cg, _, cinfo = M.project_cg(x_hat, y, num_iters=3000, tol=1e-9)

    rel_dir = float((x_dir.cpu().to(torch.float64) - x_ref).norm() / x_ref.norm())
    rel_cg = float((x_cg.cpu().to(torch.float64) - x_ref).norm() / x_ref.norm())
    # stopping-rule honesty: the residual CG reports must be the residual
    # the returned image actually has
    true_cg_res = float((M.forward(x_cg) - y).norm() / y.norm())
    fp, _, _ = M.project(x_star, y)
    fperr = float((fp.to(torch.float64).cpu() - x_star.to(torch.float64).cpu()).norm()
                  / x_star.norm())
    print(f"  T2c materialized @{size}^2: rel_err direct={rel_dir:.3e} "
          f"cg={rel_cg:.3e}  residual direct={dinfo['data_rel_residual']:.3e} "
          f"cg={cinfo['data_rel_residual']:.3e} (true {true_cg_res:.3e})  "
          f"fixed_point={fperr:.3e}  (CG iters={cinfo['n_iters']}, "
          f"cond(kept)={kappa:.2e})")
    assert rel_dir < tol_dir, f"direct Gram solve differs from reference: {rel_dir:.3e}"
    assert dinfo["data_rel_residual"] < tol_res, \
        f"direct projection not consistent: {dinfo['data_rel_residual']:.3e}"
    # The stopping rule must be honest regardless of conditioning.
    assert abs(cinfo["data_rel_residual"] - true_cg_res) <= 1e-7 + 0.1 * true_cg_res, \
        (f"CG reported residual {cinfo['data_rel_residual']:.3e} disagrees with "
         f"the true residual {true_cg_res:.3e} (stopping rule dishonest)")
    # The CG cross-check is only gate-able where CG can converge in budget:
    # convergence scales with sqrt(cond of the kept dual block). At the
    # production 512^2 (cond 1.1e6) CG reaches ~1e-6; the leap/src 32^2 twin
    # measures cond ~ 1e10, where 3000 fp64 iterations legitimately stall
    # (measured 5e-4 residual) -- geometry, not code, so report it instead.
    if kappa <= 1e7:
        assert rel_cg < tol_cg, f"materialized CG differs from reference: {rel_cg:.3e}"
        assert true_cg_res < tol_res * 10, \
            f"CG projection not consistent: {true_cg_res:.3e}"
    else:
        print(f"     CG cross-check informational at cond(kept)={kappa:.1e} "
              f"(iteration-limited; the direct solve above is the production "
              f"path and is gated)")
    assert fperr < 1e-6, f"GT not a fixed point: {fperr:.3e}"
    return True


# Shared helpers for T3-T7 (operator of record + test images)

# Production operator location; override with DBT_OPERATOR_NPZ when the npz
# lives elsewhere (e.g. a network share instead of the repo's runs/).
PROD_NPZ = Path(os.environ.get("DBT_OPERATOR_NPZ",
                               ROOT / "runs" / "operator" / "A_25deg_512.npz"))
_OP_CACHE = {}


def _surrogate_sparse(size=32, n_angles=9, arc_deg=25.0):
    """
    LEAP-free numpy surrogate of the DBT geometry: parallel-beam pencil
    projector with linear detector splat, angles k*arc/n (matching LEAP's
    default 0..22.2 degrees), and a circular FOV (pixels outside it have
    zero columns) so the per-view mass redundancy of the real geometry
    holds EXACTLY. Used only as the last rung of _get_operator, to
    validate the T3-T7 code paths where neither the production npz nor
    LEAP exists -- it is never the operator of record.
    """
    import scipy.sparse as sp
    phis = [np.deg2rad(k * arc_deg / n_angles) for k in range(n_angles)]
    c = (size - 1) / 2
    fov = size / 2 - 1.0
    rows, cols, vals = [], [], []
    for j in range(size * size):
        py, px = divmod(j, size)
        if (py - c) ** 2 + (px - c) ** 2 > fov ** 2:
            continue
        for a, t in enumerate(phis):
            xr = (px - c) * np.cos(t) + (py - c) * np.sin(t) + c
            i0 = int(np.floor(xr))
            for i, w in ((i0, 1.0 - (xr - i0)), (i0 + 1, xr - i0)):
                if 0 <= i < size and w > 0:
                    rows.append(a * size + i)
                    cols.append(j)
                    vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(n_angles * size, size * size),
                         dtype=np.float32)


def _get_operator(device="cpu"):
    """
    The operator for at-scale checks, best available first:
      1. the production 512^2 materialized operator (npz; no LEAP needed),
      2. a freshly materialized 32^2 LEAP twin,
      3. the numpy surrogate geometry (code-path validation only -- loudly
         labeled; T3-T7 results on it do not certify the real operator).
    Factorized once and cached (module-level) across T3/T4/T6/T7.
    Returns (M, scale_note, spectrum_info).
    """
    key = str(device)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    if PROD_NPZ.is_file():
        M = MaterializedOperator.load(str(PROD_NPZ), device=device)
        note = "512^2 production npz"
        # Cross-build drift check (informational): the npz was materialized
        # on a specific LEAP build; if the CURRENT live build's forward has
        # drifted from it, arms that use live LEAP (none/prox y-simulation,
        # SIRT) are no longer the same operator as the exact arm's npz.
        # The study stays self-consistent per arm either way, but runs from
        # different builds must not be mixed in one comparison.
        if _LEAP_AVAILABLE:
            try:
                proj = build_projector(CFG, device=device)
                torch.manual_seed(0)
                drift = 0.0
                for _ in range(3):
                    x = torch.rand(M.img_size, M.img_size, device=device)
                    ref = proj.forward(x)
                    drift = max(drift, float(
                        (M.forward(x).to(ref.device, ref.dtype) - ref).norm()
                        / ref.norm()))
                print(f"  (npz vs current live LEAP forward: drift={drift:.2e}"
                      f" -- informational; >1e-4 means the live build changed "
                      f"since materialization)")
            except Exception as e:
                print(f"  (npz-vs-live drift check skipped: {e})")
    elif _LEAP_AVAILABLE:
        proj = build_projector(_small_cfg_file(32), device=device)
        M = MaterializedOperator.from_projector(proj, batch_size=64,
                                                verbose=False)
        note = "32^2 LEAP twin (production npz not found)"
    else:
        M = MaterializedOperator(_surrogate_sparse(32), 9, 32, device=device)
        note = ("32^2 numpy SURROGATE -- validates the test code paths only, "
                "NOT the production operator")
    spec = M.factorize(verbose=False)
    _OP_CACHE[key] = (M, note, spec)
    return _OP_CACHE[key]


def _phantom_img(S, seed=0):
    """Smooth breast-ish phantom in [0,1], fp64: disks + low-freq texture."""
    g = torch.Generator().manual_seed(seed)
    base = _disk(S, 0.40) * 0.35 + _disk(S, 0.18) * 0.35
    low = torch.rand(1, 1, 16, 16, generator=g)
    sm = torch.nn.functional.interpolate(
        low, size=(S, S), mode="bilinear", align_corners=False)[0, 0]
    return (base * (0.8 + 0.4 * sm)).clamp(0, 1).to(torch.float64)


def _gt_like_slice(S):
    """
    A consistent-GT test image at size S: a real val slice when the
    processed dataset is reachable (T3's 'with GT slices'), else the
    synthetic phantom. Returns (img fp64 [0,1], source_note).
    """
    if S == 512:
        try:
            import yaml
            with open(ROOT / "configs" / "dbt_25deg.yaml") as f:
                cfg = yaml.safe_load(f)
            fl = Path(cfg["data"]["processed_dir"]) / "val_files.txt"
            if fl.is_file():
                from data.dataset import DBTSliceDataset
                ds = DBTSliceDataset(str(fl), augment=False,
                                     deterministic=True, data_range="01")
                gt, _ = ds[0]
                return gt.squeeze().to(torch.float64), "real val slice 0"
        except Exception as e:
            print(f"  (real GT unavailable: {e}; using synthetic phantom)")
    return _phantom_img(S), "synthetic phantom"


# T3 -- fixed point & consistency at scale (operator of record)

def check_t3_scale_consistency(device="cpu",
                               tol_fp=1e-6, tol_res=1e-7, tol_idem=1e-8):
    M, note, _ = _get_operator(device)
    S = M.img_size
    x_star, src = _gt_like_slice(S)
    y = M.forward(x_star)                    # consistent by construction

    fp, _, _ = M.project(x_star, y)
    fperr = float((fp.to(torch.float64).cpu() - x_star.cpu()).norm()
                  / x_star.norm())

    torch.manual_seed(0)
    x_hat = torch.rand(S, S, dtype=torch.float64)
    x_p, _, info = M.project(x_hat, y)
    res = info["data_rel_residual"]

    # Idempotence, cold-started by construction (the direct solve keeps no
    # state between calls -- the design's warm-start-aliasing concern).
    x_pp, _, _ = M.project(x_p, y)
    idem = float((x_pp - x_p).norm() / x_p.norm())

    print(f"  T3 @{note} ({src}): fixed_point={fperr:.3e}  "
          f"residual={res:.3e}  idempotence={idem:.3e}")
    assert fperr < tol_fp, f"consistent GT not a fixed point: {fperr:.3e}"
    assert res < tol_res, f"projection residual {res:.3e} > {tol_res}"
    assert idem < tol_idem, f"idempotence violated: {idem:.3e}"
    return True


# T4 -- orthogonal split at scale

def check_t4_orthogonal_split(device="cpu", tol=1e-6):
    M, note, spec = _get_operator(device)
    S = M.img_size
    x_star = _phantom_img(S)
    y = M.forward(x_star)
    torch.manual_seed(1)
    x_hat = torch.rand(S, S, dtype=torch.float64)
    x_p = M.project(x_hat, y)[0].cpu()               # project returns on M.device

    c = (x_p - x_hat).reshape(-1)                    # in range(A^T)
    e = (x_p - x_star).reshape(-1)                   # should be in null(A)
    cosang = float((c @ e).abs() / (c.norm() * e.norm()).clamp(min=1e-30))

    # A annihilates the residual-to-GT error (normalized by sigma1 so the
    # bound is scale-free): ||A e|| <= tol * sigma1 * ||e||.
    sig1 = float(np.sqrt(spec["sigma1_sq"]))
    a_e = float(M.forward(e.reshape(S, S)).norm() / (sig1 * e.norm()))

    lhs = float((x_hat.reshape(-1) - x_star.reshape(-1)).norm() ** 2)
    rhs = float(e.norm() ** 2 + c.norm() ** 2)
    pyth = abs(lhs - rhs) / lhs

    print(f"  T4 @{note}: |cos(correction,error)|={cosang:.3e}  "
          f"||Ae||/(sigma1||e||)={a_e:.3e}  pythagoras={pyth:.3e}")
    assert cosang < tol, f"correction not orthogonal to error: {cosang:.3e}"
    assert a_e < tol, f"error not annihilated by A: {a_e:.3e}"
    assert pyth < tol, f"Pythagoras identity violated at scale: {pyth:.3e}"
    return True


# T5 -- the rho-prox converges to the projection (dense, LEAP-free)

def _dense_prox_cg(A, x_hat, y, rho, num_iters=5000, tol=1e-12):
    """prox_cgrad's algorithm (CG on (A^T A + rho I) x = A^T y + rho x_hat)
    with dense closures and the true adjoint, fp64, tighter stopping."""
    x = x_hat.clone()
    b = A.T @ y + rho * x_hat
    r = b - (A.T @ (A @ x) + rho * x)
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(num_iters):
        if rs_old.sqrt() < tol:
            break
        Ap = A.T @ (A @ p) + rho * p
        alpha = rs_old / (p * Ap).sum().clamp(min=1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def _svd_prox(A, x_hat, y, rho, U=None, s=None, Vh=None):
    """
    Exact x(rho) = (A^T A + rho I)^{-1}(A^T y + rho x_hat), computed stably
    per singular direction: coefficient (s_i (U^T y)_i + rho (V^T x)_i) /
    (s_i^2 + rho) in the row space, x_hat untouched in its complement.
    Directions with numerically zero s belong to the complement.
    """
    if U is None:
        U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    c_y = s * (U.T @ y)
    c_x = Vh @ x_hat
    zero = s < 1e-10 * s.max()
    coef = torch.where(zero, c_x, (c_y + rho * c_x) / (s ** 2 + rho))
    return Vh.T @ coef + (x_hat - Vh.T @ c_x)


def check_t5_prox_limit(tol_limit=1e-8, tol_cg=1e-6):
    """
    On the dense reference problem: x(rho) = argmin ||Ax-y||^2 + rho||x-x_hat||^2
    approaches Pi_S(x_hat) monotonically as rho -> 0 (the claim that
    the historical prox is the limiting case of the exact projection).

    Two deliberate measurements ride along:
    - the NAIVE normal-equations solve at rho=1e-8 is reported (not gated):
      cond(A^T A + rho I) ~ sigma1^2/rho ~ 1e10 amplifies fp64 roundoff to
      ~1e-6 -- the same primal ill-conditioning wall behind the historical
      "tiny rho zeroed the recon" incident, measured
      here in miniature. The stable per-direction (SVD) solve carries the
      actual limit statement.
    - prox_cgrad's own CG algorithm (with the TRUE adjoint) at rho=1e-8
      must land on the projection, pinning the deployed prox as Pi_S's
      limiting case at the algorithm level.
    """
    A, x_star, y, x_hat = _dense_problem()
    x_proj = _pinv_project(A, x_hat, y)
    n = A.shape[1]
    I = torch.eye(n, dtype=torch.float64)
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)

    rhos = (1e-2, 1e-4, 1e-6, 1e-8)
    errs = [float((_svd_prox(A, x_hat, y, r, U, s, Vh) - x_proj).norm()
                  / x_proj.norm()) for r in rhos]
    x_naive = torch.linalg.solve(A.T @ A + rhos[-1] * I,
                                 A.T @ y + rhos[-1] * x_hat)
    err_naive = float((x_naive - x_proj).norm() / x_proj.norm())
    x_cg = _dense_prox_cg(A, x_hat, y, rho=rhos[-1])
    err_cg = float((x_cg - x_proj).norm() / x_proj.norm())

    print(f"  T5 prox limit: ||x(rho)-Pi||/||Pi|| = "
          + "  ".join(f"{e:.1e}" for e in errs)
          + f"  (rho=1e-2..1e-8, stable solves)")
    print(f"     prox-CG @1e-8: {err_cg:.1e};  naive normal-eq @1e-8: "
          f"{err_naive:.1e}  <- the primal conditioning wall the dual "
          f"formulation avoids")
    assert errs[-1] < tol_limit, \
        f"prox at rho=1e-8 does not reach the projection: {errs[-1]:.3e}"
    assert all(errs[i + 1] < errs[i] or errs[i + 1] < 1e-9
               for i in range(len(errs) - 1)), \
        f"convergence not monotone in rho: {errs}"
    assert err_cg < tol_cg, \
        f"prox_cgrad's algorithm at rho=1e-8 misses the projection: {err_cg:.3e}"
    return True


# T6 -- ensemble structure

def check_t6_ensemble_structure(device="cpu", K=8,
                                tol_res=1e-7, tol_range=1e-7,
                                min_wedge_ratio=None, min_null_contrast=1e3):
    """
    Ensemble consistency on K projected random images. The SHARP gates are (i)
    identical sinograms and (ii) zero variance on range(A^T) functionals
    with a large null-space contrast. The k-space wedge ratio (iii) is
    scale-aware: white-noise inputs lose variance only on the measured
    subspace, a fraction m/n of image space (28% at 32^2 but 1.8% at the
    production 512^2), so the wedge-mean ratio approaches 1 at scale by
    construction -- strong wedge concentration is a property of
    PRIOR-driven ensembles (measured in Stage 1 on real DOLCE samples),
    not of noise projections. Default gate: ratio > 1 + m/n.
    """
    M, note, _ = _get_operator(device)
    if min_wedge_ratio is None:
        min_wedge_ratio = 1.0 + M.m / M.n
    S = M.img_size
    x_star = _phantom_img(S)
    y = M.forward(x_star)

    samples, sinos = [], []
    for k in range(K):
        g = torch.Generator().manual_seed(100 + k)
        x_hat = torch.rand(S, S, generator=g, dtype=torch.float64)
        x_p, _, _ = M.project(x_hat, y)
        samples.append(x_p.cpu())
        sinos.append(M.forward(x_p).cpu())
    X = torch.stack(samples)                       # (K, S, S)
    Y = torch.stack(sinos)                         # (K, A, S)

    # (i) all sinograms identical (== y): worst residual + cross-sample spread
    res_max = max(float((Y[k] - y.cpu()).norm() / y.norm()) for k in range(K))
    sino_spread = float(Y.std(dim=0, unbiased=True).norm() / y.norm())

    # (ii) functional test: zero variance on a range(A^T) functional,
    # nonzero on a null-space functional (contrast ratio).
    g = torch.Generator().manual_seed(7)
    w = torch.rand(M.num_angles, S, generator=g, dtype=torch.float64)
    v_range = M.adjoint(w).cpu().reshape(-1)                 # in range(A^T)
    u = torch.rand(S, S, generator=g, dtype=torch.float64)
    lam_u = M._pinv_dual(M.forward(u))
    u_null = (u - M.adjoint(
        lam_u.to(M.device, M.dtype).reshape(M.num_angles, S)).cpu()
              ).reshape(-1)                                   # in null(A)
    flat = X.reshape(K, -1)
    var_range = float((flat @ v_range).var(unbiased=True))
    var_null = float((flat @ u_null).var(unbiased=True))
    scale_r = float((v_range.norm() * flat.norm(dim=1).mean()) ** 2)
    var_range_rel = var_range / max(scale_r, 1e-300)
    contrast = var_null / max(var_range, 1e-300)

    # (iii) k-space variance concentrates in the missing (z) wedge
    F = torch.fft.fftshift(torch.fft.fft2(X), dim=(-2, -1))
    Fc = F - F.mean(dim=0, keepdim=True)
    V = (Fc.abs() ** 2).mean(dim=0)                # complex variance per freq
    yy, xx = np.indices((S, S))
    dy, dx = yy - S // 2, xx - S // 2
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.abs(np.degrees(np.arctan2(dy, dx)))
    ring = (r >= 4) & (r < S // 2)                 # skip DC, stay in band
    x_wedge = ring & ((theta < 30) | (theta > 150))          # near kx
    z_wedge = ring & (np.abs(theta - 90) < 30)               # near ky (depth)
    Vn = V.numpy()
    ratio = float(Vn[z_wedge].mean() / max(Vn[x_wedge].mean(), 1e-300))

    print(f"  T6 @{note} (K={K}): res_max={res_max:.3e}  "
          f"sino_spread={sino_spread:.3e}  var(range-functional)={var_range_rel:.3e}  "
          f"null/range variance contrast={contrast:.1e}  "
          f"kspace z/x variance ratio={ratio:.3f} (white-noise gate "
          f"{min_wedge_ratio:.3f}; strong concentration needs prior-driven "
          f"samples -- Stage 1)")

    # qualitative panel (design: saved artifact)
    out_dir = ROOT / "results" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        ax[0].imshow(X.mean(0).numpy(), cmap="gray")
        ax[0].set_title("ensemble mean"); ax[0].axis("off")
        ax[1].imshow(X.std(0, unbiased=True).numpy(), cmap="magma")
        ax[1].set_title("per-pixel std (image)"); ax[1].axis("off")
        im = ax[2].imshow(np.log10(Vn + 1e-300), cmap="viridis")
        ax[2].contour(x_wedge, colors="w", linewidths=0.5)
        ax[2].contour(z_wedge, colors="r", linewidths=0.5)
        ax[2].set_title(f"log10 k-space ensemble variance (z/x={ratio:.1f})")
        ax[2].axis("off")
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "t6_kspace_variance.png", dpi=120)
        plt.close(fig)
        print(f"     panel saved: {out_dir / 't6_kspace_variance.png'}")
    except Exception as e:
        print(f"     (panel skipped: {e})")

    assert res_max < tol_res, f"a sample is not consistent: {res_max:.3e}"
    assert sino_spread < tol_res, f"sinograms differ across samples: {sino_spread:.3e}"
    assert var_range_rel < tol_range, \
        f"spread leaks into a range(A^T) functional: {var_range_rel:.3e}"
    assert contrast > min_null_contrast, \
        f"null-vs-range variance contrast too small: {contrast:.1e}"
    assert ratio > min_wedge_ratio, (
        f"k-space ensemble variance shows no measured-wedge depletion at all "
        f"(ratio {ratio:.3f} <= white-noise gate {min_wedge_ratio:.3f}) -- "
        f"check geometry axes")
    return True


# T7 -- spectrum & solver economics

def check_t7_spectrum_economics(device="cpu", reps=5):
    import time

    M, note, spec = _get_operator(device)
    S = M.img_size
    A_ang = M.num_angles
    sig1_sq = spec["sigma1_sq"]

    # (a) spectrum: eff. rank / near-null count vs the per-view-mass prediction
    n_null = M.m - spec["eff_rank"]
    lam_min_kept = sig1_sq / spec["cond_kept"]

    # (b) predicted near-null vectors: per-view mass differences
    quots = []
    for i in range(1, A_ang):
        u = torch.zeros(A_ang, S, dtype=torch.float64)
        u[0], u[i] = 1.0, -1.0
        u = u / u.norm()
        quots.append(float(M.adjoint(u).norm() ** 2))    # u^T AA^T u
    mass_rayleigh = max(quots) / sig1_sq

    # (c) power-iteration cross-check of sigma1^2 (through the same pair)
    g = torch.Generator().manual_seed(3)
    v = torch.rand(A_ang, S, generator=g, dtype=torch.float64)
    v = v / v.norm()
    for _ in range(100):
        w = M.forward(M.adjoint(v))
        v = w / w.norm()
    power_sig1 = float((v * M.forward(M.adjoint(v))).sum())
    power_rel = abs(power_sig1 - sig1_sq) / sig1_sq

    # (d) economics: cached direct solve vs dual CG (cold/warm) per projection
    x_star = _phantom_img(S)
    y = M.forward(x_star)
    torch.manual_seed(4)
    x_hat = torch.rand(S, S, dtype=torch.float64)

    def timeit(fn, n=reps):
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn()
        return (time.perf_counter() - t0) / n, out

    t_dir, _ = timeit(lambda: M.project(x_hat, y))
    # CG runs are deterministic and long; one rep each is representative.
    # 5000 iterations: at the production cond (1.1e6) cold CG needs ~2500
    # to pass 1e-6 (2000 measured just short at 7.1e-6).
    t_cg_cold, (xc, lam, cinfo) = timeit(
        lambda: M.project_cg(x_hat, y, num_iters=5000, tol=1e-7), n=1)
    x_hat2 = x_hat + 0.01 * torch.rand(S, S, dtype=torch.float64)
    t_cg_warm, (_, _, winfo) = timeit(
        lambda: M.project_cg(x_hat2, y, num_iters=5000, tol=1e-7,
                             lambda0=lam), n=1)

    prox_line = "prox timing skipped (LEAP not available)"
    if _LEAP_AVAILABLE:
        try:
            cfg = CFG if S == 512 else _small_cfg_file(S)
            proj = build_projector(cfg, device=device)
            xh32 = x_hat.to(torch.float32).to(device)
            y32 = y.to(torch.float32).to(device)
            t_prox, _ = timeit(
                lambda: proj.prox_cgrad(xh32, y32, rho=0.3, num_iters=100))
            prox_line = f"prox_cgrad(100 it) {t_prox * 1e3:8.2f} ms"
        except Exception as e:
            prox_line = f"prox timing failed: {e}"

    print(f"  T7 @{note}: sigma1^2={sig1_sq:.4e} (power-iter rel diff "
          f"{power_rel:.2e})  eff.rank={spec['eff_rank']}/{M.m} "
          f"(near-null={n_null}, predicted >= {A_ang - 1})  "
          f"lambda_min_kept={lam_min_kept:.3e}")
    print(f"     per-view-mass Rayleigh (max over view pairs) = "
          f"{mass_rayleigh:.3e} * sigma1^2")
    print(f"     economics/projection: direct {t_dir * 1e3:8.2f} ms | "
          f"CG cold {t_cg_cold * 1e3:8.2f} ms ({cinfo['n_iters']} it) | "
          f"CG warm {t_cg_warm * 1e3:8.2f} ms ({winfo['n_iters']} it) | "
          f"{prox_line}")

    assert n_null >= A_ang - 1, (
        f"near-null count {n_null} < {A_ang - 1}: the per-view mass "
        f"redundancy predicted by the design is not in the spectrum")
    assert mass_rayleigh < 1e-4, (
        f"per-view mass differences are not near-null "
        f"(Rayleigh {mass_rayleigh:.3e} rel. to sigma1^2)")
    assert power_rel < 0.05, (
        f"power iteration disagrees with eigh sigma1^2 by {power_rel:.2e}")
    # CG convergence scales with sqrt(cond of the kept dual block); the
    # production 512^2 geometry (cond 1.1e6) reaches 1e-6, while the 32^2
    # LEAP twin (cond ~ 1e10, measured on the leap/src build) and the numpy
    # surrogate (~1e8) legitimately stall in a 2000-iteration budget --
    # there CG timing stays informational and only the direct path is gated.
    kappa = float(spec["cond_kept"])
    if kappa <= 1e7:
        assert cinfo["data_rel_residual"] < 1e-6, \
            f"CG cross-solve not consistent: {cinfo['data_rel_residual']:.3e}"
    else:
        print(f"     (CG residual {cinfo['data_rel_residual']:.1e} informational "
              f"at cond(kept)={kappa:.1e}; direct path gated by T3)")
    return True


# T9 -- noise-aware modes: eps-MAP + discrepancy ball

def check_t9_noise_modes(device="cpu", tol_ref=1e-8, tol_delta=1e-9):
    """
    The eps-solve matches the spectral formula; eps -> 0 recovers
    Pi_S; the discrepancy-ball modes land the residual exactly on delta
    (gamma surrogate by the damping identity; exact mode by the eps root-find) and
    no-op inside the ball. These are the modes that make the pipeline
    correct OUTSIDE the noiseless inverse-crime regime.
    """
    M, note, spec = _get_operator(device)
    S = M.img_size
    x_star = _phantom_img(S)
    y = M.forward(x_star)
    torch.manual_seed(5)
    x_hat = torch.rand(S, S, dtype=torch.float64)

    # (a) eps-solve vs the dense spectral reference x_eps = x - A^T(AA^T+eI)^{-1} r
    A64 = torch.from_numpy(M.A_sp.toarray().astype(np.float64))
    r0 = (A64 @ x_hat.reshape(-1).cpu()) - y.reshape(-1).cpu()
    eps = 1e-2 * float(spec["sigma1_sq"])
    G = A64 @ A64.T
    lam_ref = torch.linalg.solve(G + eps * torch.eye(M.m, dtype=torch.float64),
                                 r0)
    x_ref = x_hat.reshape(-1).cpu() - A64.T @ lam_ref
    x_eps, _, _ = M.project(x_hat, y, eps=eps)
    rel = float((x_eps.cpu().reshape(-1) - x_ref).norm() / x_ref.norm())

    # (b) eps -> 0 recovers the exact projection
    x_p, _, _ = M.project(x_hat, y)
    errs = [float((M.project(x_hat, y, eps=e)[0] - x_p).norm() / x_p.norm())
            for e in (1e-2, 1e-6, 1e-10)]

    # (c) discrepancy ball: residual lands ON delta; no-op inside the ball
    r_norm = float((M.forward(x_hat) - y).norm())
    delta = 0.3 * r_norm
    outs = {}
    for mode in ("gamma", "exact"):
        xb, _, info = M.project_ball(x_hat, y, delta=delta, mode=mode)
        outs[mode] = abs(float((M.forward(xb) - y).norm()) - delta) / delta
    x_in, _, _ = M.project_ball(x_p, y, delta=max(delta, 1e-6), mode="gamma")
    noop = float((x_in - x_p.cpu()).norm() / x_p.norm())

    print(f"  T9 @{note}: eps-solve vs dense ref={rel:.3e}  "
          f"eps->0 errs={errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}  "
          f"ball residual-on-delta: gamma={outs['gamma']:.2e} "
          f"exact={outs['exact']:.2e}  inside-ball noop={noop:.1e}")
    assert rel < tol_ref, f"eps-solve differs from spectral reference: {rel:.3e}"
    # Convergence at tiny eps is limited by the kept block's smallest
    # eigenvalue (bound ~ eps/lam_min): spectrum-aware gate, as
    # for T2c/T7 (surrogate lam_min ~ 1e-6 vs production 3e-4).
    lam_min = float(spec["sigma1_sq"] / spec["cond_kept"])
    tol_limit = max(1e-6, 10.0 * 1e-10 / lam_min)
    assert errs[0] > errs[1] > errs[2], f"eps->0 not monotone: {errs}"
    assert errs[-1] < tol_limit, \
        f"eps->0 does not recover the projection: {errs[-1]:.3e} > {tol_limit:.1e}"
    assert outs["gamma"] < tol_delta, \
        f"gamma ball mode misses delta: {outs['gamma']:.3e}"
    assert outs["exact"] < 1e-6, \
        f"exact ball mode misses delta: {outs['exact']:.3e}"
    assert noop < 1e-12, f"inside-ball input was modified: {noop:.3e}"
    return True


def check_t10_prox_equivalence(device="cpu", tol=1e-6):
    """
    The rho-prox baseline solved through the trusted operator equals the
    primal prox it is defined as. evaluate.py routes the prox arm through
    MaterializedOperator.project(eps=rho), i.e. the dual step
        x = x_hat - A^T (A A^T + rho I)^{-1} (A x_hat - y),
    which by the push-through identity (A^T A + rho I)^{-1} A^T =
    A^T (A A^T + rho I)^{-1} is exactly the primal prox
        argmin_x ||A x - y||^2 + rho ||x - x_hat||^2 = (A^T A + rho I)^{-1}(A^T y + rho x_hat).
    This test certifies that the prox arm's stable/trusted solver computes the
    SAME operator the historical primal-CG prox targeted -- so the fix (which
    removes the mismatched-adjoint divergence to a black image) does not change
    the baseline's definition, only how it is computed. As rho -> 0 the dual
    step must also approach exact consistency ||A x - y|| -> 0.
    """
    M, note, spec = _get_operator(device)
    S = M.img_size
    x_star = _phantom_img(S)
    y = M.forward(x_star)
    torch.manual_seed(7)
    x_hat = torch.rand(S, S, dtype=torch.float64)

    A64 = torch.from_numpy(M.A_sp.toarray().astype(np.float64))
    xh = x_hat.reshape(-1).cpu()
    yv = y.reshape(-1).cpu()
    n = A64.shape[1]
    AtA = A64.T @ A64
    Aty = A64.T @ yv

    rels, resids = [], []
    rhos = (10.0, 1.0, 0.3, 1e-2, 1e-4)
    for rho in rhos:
        x_primal = torch.linalg.solve(
            AtA + rho * torch.eye(n, dtype=torch.float64), Aty + rho * xh)
        x_dual, _, _ = M.project(x_hat, y, gamma=1.0, eps=rho)
        rel = float((x_dual.cpu().reshape(-1) - x_primal).norm()
                    / x_primal.norm().clamp(min=1e-30))
        res = float((A64 @ x_dual.cpu().reshape(-1) - yv).norm()
                    / yv.norm().clamp(min=1e-30))
        rels.append(rel)
        resids.append(res)

    print(f"  T10 @{note}: prox(dual)=prox(primal) rel over rho{list(rhos)} = "
          + "/".join(f"{r:.1e}" for r in rels)
          + f"  ; data-residual rho->0 = {resids[0]:.2e}->{resids[-1]:.2e}")
    # Equivalence holds at every rho (conditioning of the primal reference
    # degrades as rho->0, so gate the well-conditioned settings tightly and
    # allow the tiny-rho reference its own round-off headroom).
    assert max(rels[:3]) < tol, f"prox dual != primal at usable rho: {rels[:3]}"
    assert max(rels) < 1e-3, f"prox dual != primal even loosely: {rels}"
    # rho -> 0 approaches exact consistency (monotone decreasing residual).
    assert resids[-1] < resids[0], f"residual not shrinking with rho: {resids}"
    return True


# pytest entry points

def test_t2_dense_reference():
    assert check_t2_dense_reference()


def test_t2_dense_properties():
    assert check_t2_dense_properties()


def test_t5_prox_limit():
    assert check_t5_prox_limit()


try:
    import pytest
    leap_required = pytest.mark.skipif(
        not _LEAP_AVAILABLE, reason="LEAP_torch not installed")

    @leap_required
    def test_t0_buffer_freshness():
        assert check_t0_buffer_freshness()

    @leap_required
    def test_t1_adjoint_fullscale():
        assert check_t1_adjoint_fullscale()

    @leap_required
    def test_t1_dense_small():
        assert check_t1_dense_small()

    @leap_required
    def test_t1_vjp_small():
        assert check_t1_vjp_small()

    @leap_required
    def test_t1d_materialized():
        assert check_t1d_materialized()

    @leap_required
    def test_t2_leap_small():
        try:
            assert check_t2_leap_small()
        except SkipCheck as e:
            pytest.skip(str(e))

    @leap_required
    def test_t2c_materialized():
        assert check_t2c_materialized()

    def _run_or_skip(fn, **kw):
        try:
            assert fn(**kw)
        except SkipCheck as e:
            pytest.skip(str(e))

    # T3/T4/T6/T7 need a materialized operator (production npz OR LEAP for
    # the 32^2 twin); they skip themselves when neither is available.
    def test_t3_scale_consistency():
        _run_or_skip(check_t3_scale_consistency)

    def test_t4_orthogonal_split():
        _run_or_skip(check_t4_orthogonal_split)

    def test_t6_ensemble_structure():
        _run_or_skip(check_t6_ensemble_structure)

    def test_t7_spectrum_economics():
        _run_or_skip(check_t7_spectrum_economics)

    def test_t9_noise_modes():
        _run_or_skip(check_t9_noise_modes)

    def test_t10_prox_equivalence():
        _run_or_skip(check_t10_prox_equivalence)
except ImportError:
    pass


# Standalone runner

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    checks = [("T2a dense reference", check_t2_dense_reference, {}),
              ("T2a dense properties", check_t2_dense_properties, {}),
              ("T5 prox limit", check_t5_prox_limit, {})]
    if _LEAP_AVAILABLE:
        checks += [
            ("T0 buffer freshness", check_t0_buffer_freshness, {"device": device}),
            ("T1a adjoint 512^2", check_t1_adjoint_fullscale, {"device": device}),
            ("T1b dense adjoint", check_t1_dense_small, {"device": device}),
            ("T1c vjp adjoint",   check_t1_vjp_small, {"device": device}),
            ("T1d materialized",  check_t1d_materialized, {"device": device}),
            ("T2b LEAP small",    check_t2_leap_small, {"device": device}),
            ("T2c materialized",  check_t2c_materialized, {"device": device}),
        ]
    else:
        print("[SKIP] LEAP_torch not installed -- running the LEAP-free "
              "checks only (T0/T1a/T1b/T1c/T2b need scripts/setup.sh).")
    # T3/T4/T6/T7 run off the operator of record: the production npz if it
    # exists (LEAP not needed), else a fresh 32^2 twin (LEAP), else SKIP.
    checks += [
        ("T3 fixed point/consistency @scale", check_t3_scale_consistency,
         {"device": device}),
        ("T4 orthogonal split @scale", check_t4_orthogonal_split,
         {"device": device}),
        ("T6 ensemble structure", check_t6_ensemble_structure,
         {"device": device}),
        ("T7 spectrum & economics", check_t7_spectrum_economics,
         {"device": device}),
        ("T9 noise modes (eps-MAP + ball)", check_t9_noise_modes,
         {"device": device}),
        ("T10 prox = trusted-dual equivalence", check_t10_prox_equivalence,
         {"device": device}),
    ]
    # T1a/T1b/T1c are adjoint-gate *measurements*: their failure selects the
    # next rung of the adjoint ladder rather than failing the
    # suite, as long as some rung ends up trusted.
    adjoint_gate_measurements = {"T1a adjoint 512^2", "T1b dense adjoint", "T1c vjp adjoint"}
    ok = True
    status = {}
    for name, fn, kw in checks:
        try:
            fn(**kw)
            status[name] = True
            print(f"[PASS] {name}")
        except SkipCheck as e:
            status[name] = None
            print(f"[SKIP] {name}: {e}")
        except AssertionError as e:
            status[name] = False
            if name in adjoint_gate_measurements:
                print(f"[MEASURED-FAIL] {name}: {e}")
            else:
                ok = False
                print(f"[FAIL] {name}: {e}")

    if _LEAP_AVAILABLE:
        if status.get("T1b dense adjoint"):
            verdict = "backward (T1b passed)"
        elif status.get("T1c vjp adjoint"):
            verdict = "autograd vjp (T1c passed independently of backward)"
        elif status.get("T1d materialized") and status.get("T2c materialized"):
            verdict = ("materialized sparse A + literal transpose; "
                       "run scripts/materialize_operator.py "
                       "once for the production geometry")
        else:
            verdict = "NONE -- no trusted operator pair; projection must not be used"
            ok = False
        print(f"[adjoint gate] adjoint source: {verdict}")

    print("All projection checks PASSED." if ok else "Some checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
