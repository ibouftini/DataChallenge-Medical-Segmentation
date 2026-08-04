"""
DBT forward/backward projector wrapper around LEAP (LEAP_torch backend).

2D parallel-beam geometry: each sagittal slice is treated as a 3D volume with
a single Z slice (Z=1).  LEAP therefore operates on:
  - Volume tensor:   (B, Z=1, Y=H, X=W)  ->  shape (B, 1, H, W)
  - Sinogram tensor: (B, numAngles, rows=1, W)  ->  shape (B, numAngles, 1, W)

All public methods expose simplified 2D-friendly shapes:
  forward(image)  :  (H,W) or (B,H,W)  ->  (numAngles, W) or (B, numAngles, W)
  backward(sino)  :  (numAngles,W) or (B,numAngles,W)  ->  (H,W) or (B,H,W)

LEAP API used (verified against wustl-cig/DOLCE leap/src/LEAP_torch.py):
  proj = Projector(forward_project=None, use_static=False,
                   use_gpu=<bool>, gpu_device=<torch.device|None>, batch_size=1)
  proj.load_param(cfg_path)     # reads img_*/proj_* keys from the .cfg
  proj.set_projector(1)         # 1 = Separable-Footprint projector
  sino  = proj(volume, "forward")    # (B,1,H,W)   -> (B,A,1,W)
  image = proj(sino,   "backward")   # (B,A,1,W)   -> (B,1,H,W)
"""

import contextlib
import io
import math
import os
import tempfile

import numpy as np
import torch

# Some LEAP builds (e.g. the leap/src build, 2026-07) print tensor shapes
# from inside every projector call -- thousands of lines per reverse
# diffusion. Suppress Python-level stdout around LEAP calls unless the user
# asks for it (exceptions still propagate normally).
_LEAP_QUIET = os.environ.get("DBT_LEAP_VERBOSE", "0") != "1"


def _quiet_leap(fn, *args, **kw):
    if _LEAP_QUIET:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kw)
    return fn(*args, **kw)

try:
    from LEAP_torch import Projector as _LeapProjector
    _LEAP_AVAILABLE = True
except ImportError:
    _LeapProjector = None
    _LEAP_AVAILABLE = False


# Config parser

def _parse_leap_cfg(cfg_path: str) -> dict:
    params = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                params[key.strip()] = val.strip()
    return params


def _resolve_device(device):
    """
    Normalise a device spec ('cuda:0', 0, 'cpu', torch.device) into a
    (use_gpu, torch_device) pair for the LEAP constructor.
    """
    if isinstance(device, int):
        torch_device = torch.device(f"cuda:{device}") if device >= 0 else torch.device("cpu")
    else:
        torch_device = torch.device(device)
    use_gpu = (torch_device.type == "cuda")
    return use_gpu, torch_device


# Exact data-consistency projection

def dual_cg_project(
    forward_fn,
    adjoint_fn,
    x_hat: torch.Tensor,
    sino: torch.Tensor,
    eps: float = 0.0,
    gamma: float = 1.0,
    num_iters: int = 30,
    tol: float = 1e-6,
    lambda0: torch.Tensor = None,
):
    """
    Project x_hat onto {x : Ax = y} (damped by gamma) via the DUAL system:

        x = x_hat - gamma * A^T lambda,   (A A^T + eps I) lambda = A x_hat - y.

    CG runs in sinogram space (m = A*W unknowns, vs n = H*W for the primal
    prox), and each iteration costs exactly one forward + one adjoint call.
    With lambda0=None CG starts from 0, which on a consistent singular SPD
    system converges to the minimum-norm dual solution (Kaasschieter 1988);
    the primal projection x is unique regardless.

    IMPORTANT: the result is the exact Euclidean projection only if
    adjoint_fn is the true adjoint of forward_fn. Verify with
    tests/test_projection.py (T1) before trusting it on a new operator pair.

    Args:
        forward_fn / adjoint_fn : callables implementing A and A^T. Shapes are
            opaque to this routine (image/sino tensors of any layout).
        eps    : Tikhonov safeguard on the dual (0 = exact projection).
        gamma  : damping in [0, 2]; 1 = full projection. Any value in [0, 2]
                 is no-harm w.r.t. a consistent GT.
        lambda0: warm start for the dual variable (from the previous
                 diffusion step); pass None for a cold start.

    Returns:
        (x, lam, info) -- projected image, dual solution (reusable as the next
        warm start), and dict with n_iters and the achieved relative data
        residual ||A x - y|| / ||y||.

    Stopping: for eps=0 the CG residual r = r0 - AA^T lam equals the data
    residual A(x_hat - A^T lam) - y of the current primal iterate, so CG
    stops when ||r|| <= tol * ||y|| -- i.e. tol IS the delivered relative
    data residual. A curvature guard (p^T AA^T p ~ 0) stops iteration once
    roundoff leaves only null(AA^T) components in r; without it CG diverges
    when asked to grind below machine precision on the singular system
    (AA^T is provably near-singular for this geometry).
    """
    scale = sino.norm().clamp(min=1e-30)
    r0 = forward_fn(x_hat) - sino
    threshold = tol * scale
    if r0.norm() <= threshold:               # already consistent
        return x_hat.clone(), torch.zeros_like(sino), {
            "n_iters": 0, "data_rel_residual": float(r0.norm() / scale)}

    def op(l):
        out = forward_fn(adjoint_fn(l))
        return out + eps * l if eps > 0 else out

    if lambda0 is None:
        lam = torch.zeros_like(sino)
        r = r0.clone()                       # op(0) = 0, save one operator call
    else:
        lam = lambda0.clone()
        r = r0 - op(lam)

    p = r.clone()
    rs_old = (r * r).sum()
    n_iters = 0
    curv_ratio_max = 0.0
    for _ in range(num_iters):
        if rs_old.sqrt() <= threshold:
            break
        Ap = op(p)
        curv = (p * Ap).sum()
        # Rayleigh quotient p^T AA^T p / p^T p along the current direction;
        # once it collapses relative to the largest one seen, r consists of
        # null(AA^T) roundoff and further steps would blow lam up.
        ratio = float(curv / (p * p).sum().clamp(min=1e-30))
        if curv <= 0 or ratio < 1e-14 * curv_ratio_max:
            break
        curv_ratio_max = max(curv_ratio_max, ratio)
        alpha = rs_old / curv
        lam = lam + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        n_iters += 1

    x = x_hat - gamma * adjoint_fn(lam)
    return x, lam, {
        "n_iters": n_iters,
        "data_rel_residual": float(rs_old.sqrt() / scale),
    }


# DBTProjector

class DBTProjector:
    """
    LEAP wrapper for 2D parallel-beam DBT slice reconstruction.

    Conventions:
      - "image" means a 2D attenuation slice:  shape (H, W)  or  (B, H, W)
      - "sino"  means a 2D sinogram:           shape (A, W)  or  (B, A, W)
        where A = num_angles, W = detector columns

    Internally LEAP sees 4D tensors (batch, Z|angles, Y|rows, X|cols); the
    extra singleton dimensions are hidden from callers.
    """

    def __init__(self, cfg_path: str, device=0):
        if not _LEAP_AVAILABLE:
            raise RuntimeError(
                "LEAP_torch not found. Run scripts/setup.sh to build and "
                "install LEAP into the dbt-dolce environment."
            )
        self.cfg_path = cfg_path
        self._use_gpu, self.device = _resolve_device(device)
        self._params = _parse_leap_cfg(cfg_path)

        self.num_angles  = int(self._params["proj_nangles"])
        self.angle_range = float(self._params["proj_arange"])
        self.img_size    = int(self._params["img_dimx"])
        self.pixel_size  = float(self._params["img_pwidth"])

        self._proj = self._build_projector()

        # SIRT geometry normalisation vectors (lazy, cached per device/dtype)
        self._col_norm = None
        self._row_norm = None
        self._norm_key = None

    # LEAP projector construction

    def _make_leap(self, batch_size: int = 1):
        proj = _LeapProjector(
            forward_project=None,
            use_static=False,
            use_gpu=self._use_gpu,
            gpu_device=self.device if self._use_gpu else None,
            batch_size=batch_size,
        )
        # DOLCE's vendored LEAP_torch.load_param splits every line on '=',
        # so comment or blank lines in the .cfg raise IndexError. Feed it a
        # sanitized `key = value` copy instead of restricting the on-disk file.
        fd, tmp_cfg = tempfile.mkstemp(suffix=".cfg", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                for key, val in self._params.items():
                    f.write(f"{key} = {val}\n")
            _quiet_leap(proj.load_param, tmp_cfg)  # geometry (img_*/proj_*)
        finally:
            os.unlink(tmp_cfg)
        _quiet_leap(proj.set_projector, 1)  # 1 = Separable-Footprint projector
        return proj

    def _build_projector(self):
        return self._make_leap(batch_size=1)

    def _proj_call(self, x4d: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Call the vendored LEAP projector, batch-safely. self._proj is built
        with batch_size=1, and LEAP's forward indexes its persistent buffer
        as proj_data[batch] over the INPUT batch -- on current builds a
        batched input therefore raises IndexError (seen in the setup smoke
        test: 'index 1 is out of bounds for dimension 0 with size 1').
        Chunk to single-item calls and clone each output (the buffer is
        reused between calls) -- correct on every build, and batch>1 only
        occurs in tests/SIRT warm-up, never in the sampling hot path.
        """
        B = x4d.shape[0]
        if B == 1:
            return _quiet_leap(self._proj, x4d, mode)
        return torch.cat([_quiet_leap(self._proj, x4d[b:b + 1], mode).clone()
                          for b in range(B)], dim=0)

    # Internal shape helpers

    def _img_to_leap(self, image: torch.Tensor):
        """(H,W) or (B,H,W) -> (B,1,H,W) [LEAP volume]. Returns (t, was_unbatched)."""
        if image.dim() == 2:
            return image[None, None].contiguous(), True
        elif image.dim() == 3:
            return image[:, None].contiguous(), False
        elif image.dim() == 4:
            return image.contiguous(), False
        raise ValueError(f"Unexpected image ndim={image.dim()}")

    def _sino_to_leap(self, sino: torch.Tensor):
        """(A,W) or (B,A,W) -> (B,A,1,W) [LEAP sino]. Returns (t, was_unbatched)."""
        if sino.dim() == 2:
            return sino[None, :, None].contiguous(), True
        elif sino.dim() == 3:
            return sino[:, :, None].contiguous(), False
        elif sino.dim() == 4:
            return sino.contiguous(), False
        raise ValueError(f"Unexpected sino ndim={sino.dim()}")

    def _leap_to_img(self, vol: torch.Tensor, unbatch: bool) -> torch.Tensor:
        """(B,1,H,W) -> (H,W) if unbatch else (B,H,W)"""
        out = vol.squeeze(1)
        return out.squeeze(0) if unbatch else out

    def _leap_to_sino(self, sino: torch.Tensor, unbatch: bool) -> torch.Tensor:
        """(B,A,1,W) -> (A,W) if unbatch else (B,A,W)"""
        out = sino.squeeze(2)
        return out.squeeze(0) if unbatch else out

    # Public forward / backward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image (H,W)|(B,H,W) -> sino (A,W)|(B,A,W)."""
        vol_4d, unbatch = self._img_to_leap(image)
        sino_4d = self._proj_call(vol_4d, "forward")
        # DOLCE's vendored LEAP writes into (and returns) one persistent
        # sinogram buffer, so the next projector call would overwrite this
        # result; clone so callers can hold outputs across calls (CG, tests).
        return self._leap_to_sino(sino_4d, unbatch).clone()

    def backward(self, sino: torch.Tensor) -> torch.Tensor:
        """sino (A,W)|(B,A,W) -> image (H,W)|(B,H,W)."""
        sino_4d, unbatch = self._sino_to_leap(sino)
        vol_4d = self._proj_call(sino_4d, "backward")
        # Same persistent-buffer hazard as forward, on the volume buffer.
        return self._leap_to_img(vol_4d, unbatch).clone()

    # SIRT

    def _init_sirt_norms(self, device, dtype):
        """Cache column (A*1) and row (A^T*1) normalisation vectors for SIRT."""
        H = W = self.img_size
        A = self.num_angles
        ones_vol  = torch.ones(1, H, W, device=device, dtype=dtype)
        ones_sino = torch.ones(1, A, W, device=device, dtype=dtype)
        self._col_norm = torch.clamp(self.forward(ones_vol),  min=1e-8)   # (1,A,W)
        self._row_norm = torch.clamp(self.backward(ones_sino), min=1e-8)  # (1,H,W)
        self._norm_key = (str(device), dtype)

    def sirt(
        self,
        sino: torch.Tensor,
        num_iters: int = 200,
        positivity: bool = True,
    ) -> torch.Tensor:
        """
        SIRT reconstruction.
        sino : (A, W) or (B, A, W)
        x    : (H, W) or (B, H, W)  [same batch dim as input]
        """
        unbatch = sino.dim() == 2
        if unbatch:
            sino = sino[None]

        B = sino.shape[0]
        H = W = self.img_size

        key = (str(sino.device), sino.dtype)
        if self._col_norm is None or self._norm_key != key:
            self._init_sirt_norms(sino.device, sino.dtype)

        col_norm = self._col_norm.expand(B, -1, -1)
        row_norm = self._row_norm.expand(B, -1, -1)

        x = torch.zeros(B, H, W, dtype=sino.dtype, device=sino.device)
        for _ in range(num_iters):
            residual = (sino - self.forward(x)) / col_norm
            x = x + self.backward(residual) / row_norm
            if positivity:
                x = torch.clamp(x, min=0.0)

        return x.squeeze(0) if unbatch else x

    # Proximal data-consistency (APGM)

    def _power_iteration(self, n_iter: int = 20) -> float:
        """
        Estimate the spectral norm sigma_max(A)^2 of the normal operator A^T A
        via power iteration.  Used to set a stable APGM step size.
        """
        H = W = self.img_size
        device = self.device
        dtype  = torch.float32
        v = torch.randn(H, W, device=device, dtype=dtype)
        v = v / (v.norm() + 1e-12)
        sigma_sq = 1.0
        for _ in range(n_iter):
            Av  = self.forward(v)
            AtAv = self.backward(Av)
            sigma_sq = AtAv.norm().item() / (v.norm().item() + 1e-12)
            v = AtAv / (AtAv.norm() + 1e-12)
        return float(sigma_sq)

    def prox_apgm(
        self,
        x_hat: torch.Tensor,
        sino: torch.Tensor,
        rho: float,
        num_iters: int = 30,
    ) -> torch.Tensor:
        """
        argmin_x  ||Ax - b||^2 + rho * ||x - x_hat||^2   via APGM.

        Step size = 1 / (sigma_max(A)^2 + rho), with sigma_max^2 estimated
        once by power iteration and cached.  Inputs/outputs are 2D (H, W).
        """
        if not hasattr(self, "_sigma_sq") or self._sigma_sq is None:
            self._sigma_sq = self._power_iteration()
        step = 1.0 / (self._sigma_sq + rho)

        x = x_hat.clone()
        z = x_hat.clone()
        t = 1.0
        for _ in range(num_iters):
            grad = self.backward(self.forward(z) - sino) + rho * (z - x_hat)
            x_new = z - step * grad
            t_new = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
            z = x_new + ((t - 1) / t_new) * (x_new - x)
            x, t = x_new, t_new
        return x

    # Proximal data-consistency (CG)

    def prox_cgrad(
        self,
        x_hat: torch.Tensor,
        sino: torch.Tensor,
        rho: float,
        num_iters: int = 100,
    ) -> torch.Tensor:
        """
        Conjugate gradient solver for argmin_x ||Ax-b||^2 + rho*||x-x_hat||^2.
        Solves (A^T A + rho I) x = A^T b + rho x_hat.  Inputs/outputs 2D (H, W).
        """
        x = x_hat.clone()
        b = self.backward(sino) + rho * x_hat
        r = b - (self.backward(self.forward(x)) + rho * x)
        p = r.clone()
        rs_old = (r * r).sum()

        for _ in range(num_iters):
            Ap = self.backward(self.forward(p)) + rho * p
            curv = (p * Ap).sum()
            # (A^T A + rho I) is SPD only when `backward` is the true adjoint of
            # `forward`. On a mismatched-adjoint LEAP build it is not, and curv
            # can be <= 0; the old code clamped the denominator to 1e-12, which
            # turned a negative curvature into a huge positive step and blew the
            # iterate up to NaN (sanitised downstream to a black image). Stop
            # cleanly instead and return the last good iterate -- the trusted
            # dual route (MaterializedOperator.project) is the correct solver;
            # this path is the diagnostic "naive prox" ablation.
            if not torch.isfinite(curv) or curv <= 1e-12:
                break
            alpha = rs_old / curv
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() < 1e-6:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def prox_solver(
        self,
        x_hat: torch.Tensor,
        sino: torch.Tensor,
        rho: float,
        method: str = "cgrad",
    ) -> torch.Tensor:
        if method == "apgm":
            return self.prox_apgm(x_hat, sino, rho)
        elif method == "cgrad":
            return self.prox_cgrad(x_hat, sino, rho)
        raise ValueError(f"Unknown prox solver: {method!r}")

    # Exact data-consistency projection

    def adjoint_vjp(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of `forward` computed by autograd VJP: A^T y = grad_x <Ax, y>.

        This equals the true adjoint of whatever `forward` implements *if*
        LEAP's autograd graph is exact. If LEAP registers the backprojector
        as the gradient of the forward projector, this returns `backward`
        under another name and adds nothing -- T1 in tests/test_projection.py
        measures both against the densely materialised A^T so the fallback
        decision rests on data, not assumption.
        sino: (A, W) -> image (H, W).
        """
        with torch.enable_grad():
            x = torch.zeros(self.img_size, self.img_size,
                            device=sino.device, dtype=sino.dtype,
                            requires_grad=True)
            s = self.forward(x)
            (g,) = torch.autograd.grad(s, x, grad_outputs=sino)
        # g may alias LEAP's persistent volume buffer (same hazard as
        # backward); clone so the adjoint is safe to hold across calls.
        return g.detach().clone()

    def project_consistent(
        self,
        x_hat: torch.Tensor,
        sino: torch.Tensor,
        eps: float = 0.0,
        gamma: float = 1.0,
        num_iters: int = 30,
        tol: float = 1e-6,
        warm_state: torch.Tensor = None,
        adjoint: str = "backward",
    ):
        """
        Exact (gamma=1) or damped projection of x_hat onto {x : Ax = sino},
        the rho->0 limit of prox_cgrad/prox_apgm computed stably through the
        m x m dual system (see dual_cg_project). Inputs/outputs 2D (H, W).

        adjoint: "backward" uses the LEAP backprojector as A^T; "vjp" uses
        the autograd adjoint (T1 gate decides which).

        Returns (x, warm_state, info); pass warm_state back in on the next
        diffusion step to warm-start CG.
        """
        adj = self.backward if adjoint == "backward" else self.adjoint_vjp
        return dual_cg_project(self.forward, adj, x_hat, sino,
                               eps=eps, gamma=gamma, num_iters=num_iters,
                               tol=tol, lambda0=warm_state)

    # Metrics

    def data_residual(self, x: torch.Tensor, sino: torch.Tensor) -> float:
        """||y - A(x)|| / ||y|| * 100  (percent)."""
        y_pred = self.forward(x)
        res = torch.norm(sino - y_pred) / torch.norm(sino).clamp(min=1e-8)
        return res.item() * 100.0

    # Operator materialization

    def materialize_sparse(self, batch_size: int = 256, verbose: bool = True):
        """
        Materialize A as a scipy CSR matrix (m, n) by forward-projecting
        batched pixel-basis images. One-time cost per geometry; each ray
        footprint touches O(width) pixels so nnz ~ 3e6 at the production
        512^2 size. This is the adjoint-fallback operator source: its literal
        transpose is A^T by construction (see MaterializedOperator).
        """
        import scipy.sparse as sp

        H = W = self.img_size
        n, m = H * W, self.num_angles * W
        leap_b = self._make_leap(batch_size=batch_size)
        basis = torch.zeros(batch_size, 1, H, W, device=self.device)
        flat = basis.view(batch_size, -1)

        starts = range(0, n, batch_size)
        if verbose:
            try:
                from tqdm import tqdm
                starts = tqdm(starts, desc=f"materializing A ({H}^2)")
            except ImportError:
                pass

        rows, cols, vals = [], [], []
        for start in starts:
            b = min(batch_size, n - start)
            basis.zero_()
            flat[torch.arange(b), torch.arange(start, start + b)] = 1.0
            sino = _quiet_leap(leap_b, basis[:b], "forward")  # (>=b, A, 1, W)
            s = sino[:b].reshape(b, -1).cpu().numpy()      # host copy
            kk, ii = np.nonzero(s)
            rows.append(ii)
            cols.append(start + kk)
            vals.append(s[kk, ii])

        A = sp.csr_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols).astype(np.int64))),
            shape=(m, n), dtype=np.float32)
        if verbose:
            print(f"  materialized A: shape=({m}, {n})  nnz={A.nnz}  "
                  f"({A.nnz / m:.1f} per row)")
        return A


# Materialized operator (the adjoint hard-failure fix)

class MaterializedOperator:
    """
    Explicit sparse A with a literal-transpose adjoint and a one-time
    eigendecomposition of the dual Gram AA^T.

    This is the designated permanent fix when the adjoint gate fails hard -- i.e.
    T1 measures that neither LEAP's `backward` nor the autograd VJP is the
    true adjoint of `forward`. With the literal transpose, every statement
    about the dual formulation applies verbatim; and since
    the Gram is only m x m (4608^2 at production size) it is factorized
    ONCE and reused across every diffusion step / slice / ensemble sample,
    replacing per-step CG by two sparse matvecs plus one dense solve.

    The Gram solve is an eigendecomposition pseudo-inverse with a relative
    eigenvalue threshold rather than a Cholesky eps-shift: AA^T is provably
    near-singular here, the thresholded pinv is the
    exact minimum-norm dual solve -- identical in form to the
    numpy.linalg.pinv reference that T2a validates at machine precision --
    and the threshold plays the eps role.
    """

    def __init__(self, A_csr, num_angles: int, img_size: int, device="cpu",
                 dtype=torch.float64):
        self.A_sp = A_csr.tocsr()
        self.num_angles = int(num_angles)
        self.img_size = int(img_size)
        self.m, self.n = self.A_sp.shape
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # Computation dtype. The stored entries (fp32, from LEAP) DEFINE the
        # operator; the solves must nevertheless run in fp64: the dual Gram
        # has cond ~ 1e6 (measured at 512^2), which amplifies fp32 roundoff
        # (1e-7) to ~1e-1 -- measured as exactly that failure in T2c before
        # this default. In fp64 the same amplification leaves 1e-10 headroom.
        # Cost is negligible (~3.5M nnz).
        self.dtype = dtype
        coo = self.A_sp.tocoo()
        idx = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        val = torch.from_numpy(coo.data).to(dtype)
        self._A = torch.sparse_coo_tensor(
            idx, val, (self.m, self.n)).coalesce().to(self.device)
        self._At = torch.sparse_coo_tensor(
            idx.flip(0), val, (self.n, self.m)).coalesce().to(self.device)
        self._eig = None                     # (U fp64 cpu, eigvals fp64, rcond)

    @classmethod
    def from_projector(cls, projector: "DBTProjector",
                       batch_size: int = 256, verbose: bool = True):
        A = projector.materialize_sparse(batch_size=batch_size, verbose=verbose)
        return cls(A, projector.num_angles, projector.img_size,
                   device=projector.device)

    def save(self, path: str):
        np.savez_compressed(
            path, data=self.A_sp.data, indices=self.A_sp.indices,
            indptr=self.A_sp.indptr, shape=np.asarray(self.A_sp.shape),
            num_angles=self.num_angles, img_size=self.img_size)

    @classmethod
    def load(cls, path: str, device="cpu"):
        import scipy.sparse as sp
        z = np.load(path)
        A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"]))
        return cls(A, int(z["num_angles"]), int(z["img_size"]), device=device)

    # Operator pair (literal adjoint)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image (H,W) -> sino (A,W); sparse matvec with the explicit A."""
        x = image.reshape(-1, 1).to(self.device, self.dtype)
        return torch.sparse.mm(self._A, x).reshape(self.num_angles, self.img_size)

    def adjoint(self, sino: torch.Tensor) -> torch.Tensor:
        """sino (A,W) -> image (H,W); the literal transpose A^T (exact)."""
        s = sino.reshape(-1, 1).to(self.device, self.dtype)
        return torch.sparse.mm(self._At, s).reshape(self.img_size, self.img_size)

    # Dual-Gram factorization and projection

    def factorize(self, rcond: float = 1e-12, verbose: bool = True):
        """
        One-time dense eigendecomposition of AA^T (m x m, fp64 on CPU).
        rcond is the relative eigenvalue cutoff of the pseudo-inverse; the
        near-null cluster from per-view mass redundancy sits at ~1e-14 of
        sigma1^2 with fp32 operator entries, so any cutoff in [1e-11, 1e-8]
        lands in the spectral gap. Returns spectrum info (feeds T7).
        """
        # The Gram must be ASSEMBLED in fp64, not assembled in fp32 and cast:
        # fp32 assembly perturbs its entries by ~1e-7 relative, which the
        # ~1e6 conditioning turns into ~1e-3 solve error. Measured (T2c,
        # 2026-07-03): direct solve 4.5e-3 with fp32 assembly while CG
        # through the identical fp64 operator reached 2.3e-6 -- the
        # discrepancy isolated fp32 Gram assembly as the only error source.
        A64 = self.A_sp.astype(np.float64)
        G = (A64 @ A64.T).toarray()
        w, U = np.linalg.eigh(G)                       # ascending
        sig1 = float(w[-1])
        keep = w > rcond * max(sig1, 1e-300)
        info = {"sigma1_sq": sig1, "eff_rank": int(keep.sum()), "m": self.m,
                "cond_kept": float(sig1 / w[keep].min()) if keep.any() else float("inf")}
        self._eig = (torch.from_numpy(U), torch.from_numpy(w), float(rcond))
        if verbose:
            print(f"  Gram spectrum: sigma1^2={sig1:.4e}  "
                  f"eff.rank={info['eff_rank']}/{self.m}  "
                  f"cond(kept)={info['cond_kept']:.3e}")
        return info

    def _pinv_dual(self, r: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """
        Dual solve via the cached eigh. eps = 0: minimum-norm solution of
        (AA^T) lam = r (thresholded pseudo-inverse). eps > 0: the regularized
        solve (AA^T + eps I)^{-1} r -- SPD, so no threshold is needed (the
        eps-MAP mode; eigenvalues clamped at 0
        against eigh roundoff).
        """
        U, w, rcond = self._eig
        r64 = r.detach().reshape(-1).to("cpu", torch.float64)
        c = U.T @ r64
        if eps > 0:
            c = c / (w.clamp(min=0.0) + eps)
        else:
            keep = w > rcond * max(float(w[-1]), 1e-300)
            c = torch.where(keep, c / w, torch.zeros_like(c))
        return U @ c

    def project(self, x_hat: torch.Tensor, sino: torch.Tensor,
                gamma: float = 1.0, eps: float = 0.0):
        """
        Exact (gamma-damped) projection onto {x : Ax = sino} via the cached
        Gram factorization -- no CG, cost = 2 sparse matvecs + 2 dense
        (m x m) matvecs. eps > 0 switches to the noise-aware MAP step
        x = x_hat - A^T (AA^T + eps I)^{-1} (A x_hat - y), the rho-prox /
        eps-solve with eps = sigma^2/tau^2
        -- NOT an exact projection.
        Returns (x, lam, info) like dual_cg_project.
        """
        if self._eig is None:
            self.factorize(verbose=False)
        out_dtype = x_hat.dtype
        xh = x_hat.to(self.device, self.dtype)
        y = sino.to(self.device, self.dtype)
        r = self.forward(xh) - y
        lam = self._pinv_dual(r, eps=eps).to(self.device, self.dtype) \
                                         .reshape(self.num_angles, self.img_size)
        x = xh - gamma * self.adjoint(lam)
        res = float((self.forward(x) - y).norm() / y.norm().clamp(min=1e-30))
        return x.to(out_dtype), lam, {"n_iters": 0, "data_rel_residual": res}

    def project_ball(self, x_hat: torch.Tensor, sino: torch.Tensor,
                     delta: float, mode: str = "gamma"):
        """
        Relaxation onto the discrepancy ball {x : ||Ax - y|| <= delta},
        delta = sigma*sqrt(m) by the discrepancy principle.
        If ||A x_hat - y|| <= delta already: no-op.

        mode="gamma" (cheap surrogate): damped step with
        gamma = 1 - delta/||A x_hat - y||; the residual
        lands EXACTLY on delta along the full-projection direction.
        mode="exact": the true ball projection -- the eps-solve with the
        unique eps making the constraint active, found by bisection on the
        closed-form residual eps -> ||sum_i (eps/(w_i+eps)) c_i u_i||
        (monotone increasing in eps; costs one eigh-basis transform total).
        Returns (x, lam_or_None, info).
        """
        if self._eig is None:
            self.factorize(verbose=False)
        out_dtype = x_hat.dtype
        xh = x_hat.to(self.device, self.dtype)
        y = sino.to(self.device, self.dtype)
        r = self.forward(xh) - y
        r_norm = float(r.norm())
        if r_norm <= delta:
            return x_hat.clone(), None, {"n_iters": 0, "eps": 0.0,
                                         "data_rel_residual": float(
                                             r_norm / y.norm().clamp(min=1e-30))}
        if mode == "gamma":
            g = 1.0 - delta / r_norm
            x, lam, info = self.project(xh, y, gamma=g, eps=0.0)
            info["gamma"] = g
            return x.to(out_dtype), lam, info
        if mode != "exact":
            raise ValueError(f"unknown ball-projection mode {mode!r}")
        # Closed-form residual in the eigenbasis: bisect log-eps.
        U, w, _ = self._eig
        c = (U.T @ r.detach().reshape(-1).to("cpu", torch.float64))
        wc = w.clamp(min=0.0)

        def resid(eps):
            return float(((eps / (wc + eps)) * c).norm())

        lo, hi = 1e-16, float(wc[-1]) * 1e4 + 1.0
        for _ in range(200):
            mid = (lo * hi) ** 0.5
            if resid(mid) < delta:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-12:
                break
        eps_star = (lo * hi) ** 0.5
        x, lam, info = self.project(xh, y, gamma=1.0, eps=eps_star)
        info["eps"] = eps_star
        return x.to(out_dtype), lam, info

    def project_cg(self, x_hat: torch.Tensor, sino: torch.Tensor, **kw):
        """dual_cg_project through the (A, literal A^T) pair."""
        out_dtype = x_hat.dtype
        x, lam, info = dual_cg_project(self.forward, self.adjoint,
                                       x_hat.to(self.device, self.dtype),
                                       sino.to(self.device, self.dtype), **kw)
        return x.to(out_dtype), lam, info

    # Faithful DOLCE data-consistency (dataFidelities/CTClass.py:apgm)

    def dolce_apgm(self, x_iter: torch.Tensor, target_sino: torch.Tensor,
                   rho: float, step: float = 5e-6, n_iters: int = 30):
        """
        Faithful port of DOLCE's DEPLOYED data-consistency step (the `apgm`
        method of dataFidelities/CTClass.py, DOLCE's default solver), run
        through the trusted materialized (A, A^T) pair.

        This is DELIBERATELY NOT an exact projection. DOLCE applies a GENTLE,
        early-stopped accelerated-gradient correction to the post-update
        diffusion iterate, anchored back to that iterate:

            grad   = A^T (A s_k - b)
            x_{k+1} = s_k - step * grad - rho * (s_k - z),     z = x_iter (anchor)
            s_{k+1} = Nesterov-extrapolate(x_{k+1})

        for `n_iters` (DOLCE: 30) with a small FIXED `step` (DOLCE: 5e-6 in
        raw-LEAP units). The per-step data residual contraction on the top
        singular direction is ~= step * sigma1^2 (sigma1^2 from `factorize`);
        with our operator (sigma1^2 ~ 327) DOLCE's step 5e-6 removes ~0.16% of
        the residual per outer diffusion step -- a whisper that accumulates
        over the trajectory, versus `project`/`project_cg` which remove ~all of
        it in one step. `step` is exposed because it lives in the operator's
        (geometry-dependent) units and should be swept on a new geometry.

        `rho` is DOLCE's trust-region weight pulling s back toward the iterate z
        (DOLCE schedules it as 1 - exp(-(i+1)/(8T)); the caller supplies the
        value). `target_sino` is the measurement, which the caller anneals to
        the iterate's noise level (DOLCE: q_sample(y, t)).

        Inputs/outputs are 2D (H, W) in physics ([0,1]) space.
        """
        out_dtype = x_iter.dtype
        z = x_iter.to(self.device, self.dtype)
        b = target_sino.to(self.device, self.dtype)
        x = z.clone()
        s = z.clone()
        t = 1.0
        for _ in range(n_iters):
            grad = self.adjoint(self.forward(s) - b)
            x_next = s - step * grad - rho * (s - z)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            s = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x, t = x_next, t_next
        return x.to(out_dtype)


# Factory

def build_projector(cfg_path: str, device=0) -> DBTProjector:
    return DBTProjector(cfg_path, device=device)
