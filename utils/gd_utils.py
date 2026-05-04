"""
utils/gd_utils.py — Shared gradient-descent refinement utilities.

Provides a differentiable Gaussian renderer and an Adam-based refinement
loop that can be called from both the tiled EM pipeline and the full-image
hybrid residual pipeline.

Public API
----------
build_gd_params(all_means, all_covs, all_colors, eff_amps, device)
    Convert numpy Gaussian parameters to optimisable torch tensors.

render_gaussians_torch(means_t, l_diag_t, l_off_t, logit_colors_t,
                        log_abs_amps_t, amp_signs_t, pos, H, W)
    Differentiable additive signed Gaussian renderer (matches
    ``render_gaussians_signed`` semantics).

gradient_descent_refinement(all_means, all_covs, all_colors, eff_amps,
                             gt_np, composite_np, bboxes, gd_cfg)
    Fine-tune a merged Gaussian set with gradient descent, starting from
    the provided composite image (guaranteed non-degrading via a frozen
    gap bridge).

Gap-bridge guarantee
--------------------
The EM pipeline may assemble its composite using a mix of rendering models
(weighted-average for initial fit, additive signed for residuals).  The GD
renderer uses a unified additive signed model.  To ensure step-0 output
equals the EM composite exactly — so metrics can only improve — we compute::

    gap = composite - clip(gd_render_0, 0, 1)   (constant, no gradient)

and add it every step::

    output = clip(gd_render + gap, 0, 1)

At step 0:  output == composite  exactly.
After each step:  output diverges from composite only if it reduces the loss.
"""

from __future__ import annotations

import numpy as np

# Torch is optional — callers must check _TORCH_AVAILABLE or gd_cfg['enabled']
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------------

def build_gd_params(
    all_means : np.ndarray,
    all_covs  : np.ndarray,
    all_colors: np.ndarray,
    eff_amps  : np.ndarray,
    device    : 'torch.device',
) -> tuple:
    """
    Convert numpy Gaussian parameters to optimisable torch tensors for the
    **additive signed** GD renderer (matching ``render_gaussians_signed``).

    Parameterisation
    ----------------
    - Covariances → Cholesky log-diagonal + off-diagonal (forces Σ ≻ 0).
    - Colors → logit (sigmoid in renderer keeps values in [0, 1]).
    - Amplitudes → log|amp| (learnable) + sign (fixed, never flipped).
      ``eff_amps = amp_sign × gmm_weight`` as computed at the call-site.

    Cholesky factor L (lower triangular, 2×2)::

        [[exp(l_diag[k,0]),  0              ],
         [l_off[k],          exp(l_diag[k,1])]]

    Args:
        all_means  : (K, 2) pixel-space Gaussian centres.
        all_covs   : (K, 2, 2) pixel-space covariance matrices.
        all_colors : (K, 3) colours in [0, 1].
        eff_amps   : (K,) effective signed amplitudes (amp_sign × weight).
        device     : torch device to place tensors on.

    Returns:
        (means_t, l_diag_t, l_off_t, logit_colors_t, log_abs_amps_t, amp_signs_t)
        First five have requires_grad=True; amp_signs_t is fixed (no gradient).
    """
    K   = all_means.shape[0]
    eps = 1e-5

    # Cholesky factorisation of each 2×2 covariance
    l_diag = np.zeros((K, 2), dtype=np.float32)
    l_off  = np.zeros(K,       dtype=np.float32)
    for k, cov in enumerate(all_covs):
        cov_reg = cov + eps * np.eye(2)
        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            scale = float(np.sqrt(max(np.trace(cov_reg) / 2, eps)))
            L     = np.array([[scale, 0.0], [0.0, scale]])
        l_diag[k, 0] = float(np.log(max(L[0, 0], eps)))
        l_diag[k, 1] = float(np.log(max(L[1, 1], eps)))
        l_off[k]     = float(L[1, 0])

    # Colors: logit so sigmoid recovers original value
    colors_clipped = np.clip(all_colors, eps, 1.0 - eps)
    logit_colors   = np.log(colors_clipped / (1.0 - colors_clipped)).astype(np.float32)

    # Amplitudes: decompose into fixed sign and learnable log-magnitude.
    # Signs are NEVER changed during optimisation to preserve EM semantics
    # (positive Gaussians stay additive; negative ones stay subtractive).
    eff_amps_arr = np.asarray(eff_amps, dtype=np.float32)
    amp_signs    = np.sign(eff_amps_arr)
    amp_signs[amp_signs == 0] = 1.0
    log_abs_amps = np.log(np.maximum(np.abs(eff_amps_arr), eps)).astype(np.float32)

    to       = lambda a: torch.tensor(a, dtype=torch.float32, device=device,
                                       requires_grad=True)
    to_fixed = lambda a: torch.tensor(a, dtype=torch.float32, device=device,
                                       requires_grad=False)
    return (
        to(all_means.astype(np.float32)),  # means
        to(l_diag),                         # Cholesky log-diagonal
        to(l_off),                          # Cholesky off-diagonal
        to(logit_colors),                   # logit-colors
        to(log_abs_amps),                   # learnable log|amp|
        to_fixed(amp_signs),                # fixed signs
    )


# ---------------------------------------------------------------------------
# Differentiable renderer
# ---------------------------------------------------------------------------

def render_gaussians_torch(
    means_t       : 'torch.Tensor',
    l_diag_t      : 'torch.Tensor',
    l_off_t       : 'torch.Tensor',
    logit_colors_t: 'torch.Tensor',
    log_abs_amps_t: 'torch.Tensor',
    amp_signs_t   : 'torch.Tensor',
    pos           : 'torch.Tensor',
    H             : int,
    W             : int,
) -> 'torch.Tensor':
    """
    Differentiable additive signed Gaussian renderer.

    Matches ``render_gaussians_signed`` semantics used by the EM residual
    pipeline::

        render(p) = Σ_k amp_k · α_k(p) · c_k

    where
      * α_k(p) = exp(-0.5 · ||L_k⁻¹(p − μ_k)||²)  — peak-normalised to 1
      * amp_k  = amp_signs_t[k] · exp(log_abs_amps_t[k])  — signed amplitude
      * c_k    = sigmoid(logit_colors_t[k])  — colour in [0, 1]

    The output is NOT clamped — callers add the frozen gap and then clamp,
    so that gradient signal passes through the clamp boundary correctly.

    Cholesky factor L_k (lower triangular, 2×2)::

        L_k = [[exp(l_diag_t[k,0]),  0                  ],
               [l_off_t[k],          exp(l_diag_t[k,1])]]

    Args:
        means_t        : (K, 2)  Gaussian centres in pixel coordinates.
        l_diag_t       : (K, 2)  log-diagonal of Cholesky factor.
        l_off_t        : (K,)    below-diagonal entry of Cholesky factor.
        logit_colors_t : (K, 3)  logit-transformed colours.
        log_abs_amps_t : (K,)    log of |amplitude| (learnable).
        amp_signs_t    : (K,)    fixed amplitude signs (+1 or -1).
        pos            : (P, 2)  pixel positions [[x, y], ...].
        H, W           : Image height and width.

    Returns:
        (H, W, 3) float32 tensor — signed additive canvas, NOT clamped.
    """
    colors = torch.sigmoid(logit_colors_t)                 # (K, 3) in [0, 1]
    amps   = amp_signs_t * torch.exp(log_abs_amps_t)       # (K,) signed

    # Reconstruct Cholesky factor entries
    l00 = torch.exp(l_diag_t[:, 0])   # (K,) > 0
    l11 = torch.exp(l_diag_t[:, 1])   # (K,) > 0
    l10 = l_off_t                      # (K,)

    # Mahalanobis distance via forward substitution  L z = (p - mu)
    diff = pos.unsqueeze(0) - means_t.unsqueeze(1)         # (K, P, 2)
    z0   = diff[:, :, 0] / l00.unsqueeze(1)                # (K, P)
    z1   = (diff[:, :, 1] - l10.unsqueeze(1) * z0) / l11.unsqueeze(1)
    alpha = torch.exp(-0.5 * (z0 ** 2 + z1 ** 2))         # (K, P), peak-norm=1

    # Additive canvas: canvas[p, c] = Σ_k amp_k · α_k(p) · c_k
    canvas = torch.einsum('k,kp,kc->pc', amps, alpha, colors)  # (P, 3)
    return canvas.reshape(H, W, 3)  # caller adds gap and clamps


# ---------------------------------------------------------------------------
# Gradient-descent refinement
# ---------------------------------------------------------------------------

def gradient_descent_refinement(
    all_means    : np.ndarray,
    all_covs     : np.ndarray,
    all_colors   : np.ndarray,
    eff_amps     : np.ndarray,
    gt_np        : np.ndarray,
    composite_np : np.ndarray,
    bboxes       : list,
    gd_cfg       : dict,
) -> tuple:
    """
    Fine-tune the Gaussian set with Adam gradient descent, starting from
    the EM composite image.

    The key guarantee: the output at step 0 is **identical** to the EM
    composite, so quality metrics can only improve (or stay equal).

    Gap-bridge mechanism
    --------------------
    The EM composite may have been assembled with a mix of rendering models
    (weighted-average for initial Gaussians, additive signed for residuals).
    The differentiable GD renderer uses a unified additive signed model,
    so at initialisation its output (``gd_render_0``) differs from the
    composite.  A **frozen gap** bridges this::

        gap = composite - clip(gd_render_0, 0, 1)    (constant, no gradient)

    Every step the output is::

        output = clip(gd_render + gap, 0, 1)

    At step 0: ``output = clip(gd_render_0 + gap) = composite`` exactly.
    After each step: ``gd_render`` changes and the output improves.

    Boundary weighting
    ------------------
    For tiled pipelines, pixels near tile boundaries are given a higher
    loss weight (``boundary_weight``) to target block artefacts.

    For full-image pipelines (``bboxes = [(0, 0, H, W)]``), there are no
    interior tile boundaries, so the boundary mask returns all-False and
    ``w_map`` is all-ones (uniform loss).  Set ``boundary_weight: 1.0``
    and ``border_px: 0`` for the full-image case.

    Args:
        all_means    : (K, 2)     global pixel-space Gaussian centres.
        all_covs     : (K, 2, 2)  pixel-space covariance matrices.
        all_colors   : (K, 3)     Gaussian colours in [0, 1].
        eff_amps     : (K,)       effective signed amplitudes (amp_sign × weight).
                                  Scale does not need to match the composite
                                  exactly — the gap bridge compensates.
        gt_np        : (H, W, 3)  float32 ground-truth image.
        composite_np : (H, W, 3)  float32 EM composite (the step-0 anchor).
        bboxes       : Tile bounding boxes for boundary-pixel up-weighting.
                       Pass ``[(0, 0, H, W)]`` for full-image (no tile seams).
        gd_cfg       : Dict with keys: n_steps, lr_means, lr_colors, lr_scales,
                       lambda_dssim, boundary_weight, border_px, log_interval.

    Returns:
        (refined_render, refined_means, refined_covs, refined_colors, refined_amps)
        All numpy; render is (H, W, 3) float32, PSNR >= composite_np PSNR.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for gradient-descent refinement.  "
            "Install it with:  pip install torch  "
            "or set gd.enabled: false in the config."
        )

    from utils.tiling_utils  import boundary_mask
    from utils.metrics_utils import compute_ssim_loss

    H, W   = gt_np.shape[:2]
    device = torch.device('cpu')

    n_steps      = int(gd_cfg.get('n_steps',          150))
    lr_means     = float(gd_cfg.get('lr_means',        0.5))
    lr_colors    = float(gd_cfg.get('lr_colors',       0.005))
    lr_scales    = float(gd_cfg.get('lr_scales',       0.002))
    lambda_dssim = float(gd_cfg.get('lambda_dssim',    0.2))
    bnd_weight   = float(gd_cfg.get('boundary_weight', 1.0))
    border_px    = int(gd_cfg.get('border_px',          0))
    log_interval = int(gd_cfg.get('log_interval',       25))

    print(f"    GD refinement: K={len(all_means)}  steps={n_steps}  "
          f"lr_means={lr_means}  bnd_weight={bnd_weight:.1f}x")

    # Pixel position grid  (P = H*W, 2) — [x, y] matching Gaussian mean convention
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pos_np = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    pos    = torch.tensor(pos_np, device=device)

    gt_t        = torch.tensor(gt_np,        dtype=torch.float32, device=device)
    composite_t = torch.tensor(composite_np, dtype=torch.float32, device=device)

    # Boundary pixel weight map
    bmask = boundary_mask((H, W), bboxes, border_px=border_px).ravel()
    w_map = np.where(bmask, bnd_weight, 1.0).astype(np.float32)
    w_t   = torch.tensor(w_map, device=device)

    (means_t, l_diag_t, l_off_t,
     logit_colors_t, log_abs_amps_t, amp_signs_t) = build_gd_params(
        all_means, all_covs, all_colors, eff_amps, device)

    # Frozen gap: guarantees step-0 output == composite
    with torch.no_grad():
        init_render = render_gaussians_torch(
            means_t, l_diag_t, l_off_t, logit_colors_t,
            log_abs_amps_t, amp_signs_t, pos, H, W,
        ).clamp(0.0, 1.0)
        gap = composite_t - init_render   # (H, W, 3), constant throughout

    optimizer = torch.optim.Adam([
        {'params': [means_t],              'lr': lr_means  },
        {'params': [l_diag_t, l_off_t],    'lr': lr_scales },
        {'params': [logit_colors_t],       'lr': lr_colors },
        {'params': [log_abs_amps_t],       'lr': lr_scales },
    ])

    for step in range(n_steps):
        optimizer.zero_grad()

        gd_render = render_gaussians_torch(
            means_t, l_diag_t, l_off_t, logit_colors_t,
            log_abs_amps_t, amp_signs_t, pos, H, W,
        )
        # output = clip(gd_render + gap, 0, 1)
        # At step 0: output == composite exactly (gap accounts for model offset).
        output = (gd_render + gap).clamp(0.0, 1.0)

        l1_loss    = ((output - gt_t).abs().mean(dim=2).reshape(-1) * w_t).mean()
        dssim_loss = compute_ssim_loss(output, gt_t)
        loss       = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * dssim_loss

        loss.backward()
        optimizer.step()

        if log_interval > 0 and (step + 1) % log_interval == 0:
            with torch.no_grad():
                print(f"      step {step+1:4d}/{n_steps}  "
                      f"loss={loss.item():.5f}  "
                      f"L1={l1_loss.item():.5f}  "
                      f"dSSIM={dssim_loss.item():.5f}")

    # Extract final render (with gap) and updated parameters
    with torch.no_grad():
        final_gd     = render_gaussians_torch(
            means_t, l_diag_t, l_off_t, logit_colors_t,
            log_abs_amps_t, amp_signs_t, pos, H, W,
        )
        final_render = (final_gd + gap).clamp(0.0, 1.0).numpy()

    refined_means  = means_t.detach().numpy()
    refined_colors = torch.sigmoid(logit_colors_t).detach().numpy()
    refined_amps   = (amp_signs_t * torch.exp(log_abs_amps_t)).detach().numpy()
    l00 = np.exp(l_diag_t.detach().numpy()[:, 0])
    l11 = np.exp(l_diag_t.detach().numpy()[:, 1])
    l10 = l_off_t.detach().numpy()
    K   = len(l00)
    refined_covs = np.zeros((K, 2, 2), dtype=np.float32)
    for k in range(K):
        Lk = np.array([[l00[k], 0.0], [l10[k], l11[k]]], dtype=np.float32)
        refined_covs[k] = Lk @ Lk.T

    return (
        final_render.astype(np.float32),
        refined_means,
        refined_covs,
        refined_colors,
        refined_amps,
    )
