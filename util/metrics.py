"""
Evaluation metrics for font generation, following FontDiffuser
(Yang et al., 2023, arXiv:2312.12142).

Metrics:
  - SSIM  (Structural Similarity Index Measure)  -- higher is better
  - LPIPS (Learned Perceptual Image Patch Similarity) -- lower is better
  - L1    (Mean Absolute Error) -- lower is better
  - FID   (Fréchet Inception Distance) -- lower is better (batch/folder-level only)

Input: 256x512 image where the left 256x256 half is the target and the right
256x256 half is the generated image.
"""

import os
import glob
import shutil
import tempfile

import torch
import numpy as np
import cv2
from util.image_io import imwrite_unicode
from pytorch_msssim import ssim as _ssim_fn
import lpips
import torch_fidelity


_lpips_fn = None


def _get_lpips_fn(device='cpu'):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    return _lpips_fn


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def _save_tensors_to_dir(images, out_dir):
    """Save a (N, 3, H, W) float [0,1] tensor as PNGs in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    images_np = (images * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    for i, img in enumerate(images_np):
        # RGB -> BGR for cv2
        imwrite_unicode(os.path.join(out_dir, f"{i:06d}.png"), img[:, :, ::-1])


def compute_fid(images_real, images_gen, device="cpu", batch_size=64):
    """Compute FID between two sets of images using torch-fidelity.

    Args:
        images_real: (N, 3, H, W) float tensor in [0, 1] — reference set.
        images_gen: (M, 3, H, W) float tensor in [0, 1] — generated set.
        device: torch device string (used to set cuda flag).
        batch_size: batch size for Inception feature extraction.

    Returns:
        FID score (float).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        real_dir = os.path.join(tmpdir, "real")
        gen_dir = os.path.join(tmpdir, "gen")
        _save_tensors_to_dir(images_real, real_dir)
        _save_tensors_to_dir(images_gen, gen_dir)

        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            input2=real_dir,
            cuda=("cuda" in device),
            fid=True,
            isc=False,
            kid=False,
            prc=False,
            verbose=False,
            batch_size=batch_size,
        )
        return metrics["frechet_inception_distance"]
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(pair_image, device='cpu'):
    """Compute SSIM, LPIPS, and L1 from a single 256x512 paired image.

    Args:
        pair_image: numpy array (H, W*2, C) uint8 BGR or RGB, or a file path.
                    Left half = target, right half = generated.
        device: torch device for LPIPS network.

    Returns:
        dict with keys 'ssim', 'lpips', 'l1'.
    """
    if isinstance(pair_image, str):
        pair_image = cv2.imread(pair_image)

    h, w = pair_image.shape[:2]
    assert w == 2 * h, f"Expected width=2*height, got {w}x{h}"

    target_bgr = pair_image[:, :h, :]
    generated_bgr = pair_image[:, h:, :]

    # BGR -> RGB, uint8 -> float [0,1], (H,W,3) -> (1,3,H,W)
    target = torch.from_numpy(target_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    generated = torch.from_numpy(generated_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    ssim_val = _ssim_fn(generated, target, data_range=1.0, size_average=True).item()

    lpips_fn = _get_lpips_fn(device)
    gen_lpips = (generated.to(device) * 2.0 - 1.0)
    tgt_lpips = (target.to(device) * 2.0 - 1.0)
    lpips_val = lpips_fn(gen_lpips, tgt_lpips).item()

    l1_val = torch.abs(generated - target).mean().item()

    return {'ssim': ssim_val, 'lpips': lpips_val, 'l1': l1_val}


@torch.no_grad()
def compute_metrics_batch(pair_images, device='cpu'):
    """Compute SSIM, LPIPS, and L1 from a batch of 256x512 paired images.

    Args:
        pair_images: numpy array (B, H, W*2, C) uint8 BGR, or list of arrays.
                     Left half = target, right half = generated.
        device: torch device for LPIPS network.

    Returns:
        dict with keys 'ssim', 'lpips', 'l1' (averaged over batch).
    """
    if isinstance(pair_images, list):
        pair_images = np.stack(pair_images)

    B, h, w, c = pair_images.shape
    assert w == 2 * h, f"Expected width=2*height, got {w}x{h}"

    targets_bgr = pair_images[:, :, :h, :]
    generateds_bgr = pair_images[:, :, h:, :]

    # (B,H,W,3) BGR uint8 -> (B,3,H,W) RGB float [0,1]
    targets = torch.from_numpy(targets_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
    generateds = torch.from_numpy(generateds_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0

    ssim_val = _ssim_fn(generateds, targets, data_range=1.0, size_average=True).item()

    lpips_fn = _get_lpips_fn(device)
    gen_lpips = (generateds.to(device) * 2.0 - 1.0)
    tgt_lpips = (targets.to(device) * 2.0 - 1.0)
    lpips_val = lpips_fn(gen_lpips, tgt_lpips).mean().item()

    l1_val = torch.abs(generateds - targets).mean().item()

    return {'ssim': ssim_val, 'lpips': lpips_val, 'l1': l1_val}


@torch.no_grad()
def compute_metrics_folder(folder, device='cpu', batch_size=32):
    """Compute averaged SSIM, LPIPS, L1, and FID over all 256x512 paired images in a folder.

    Args:
        folder: path to directory containing paired PNG images.
        device: torch device for LPIPS network.
        batch_size: number of images per batch for LPIPS.

    Returns:
        dict with keys 'ssim', 'lpips', 'l1', 'fid' (averaged over all images).
    """
    paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    assert len(paths) > 0, f"No PNG files found in {folder}"

    lpips_fn = _get_lpips_fn(device)
    ssim_sum, lpips_sum, l1_sum, count = 0.0, 0.0, 0.0, 0

    all_targets_rgb = []
    all_generateds_rgb = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = [cv2.imread(p) for p in batch_paths]

        h = imgs[0].shape[0]
        targets_bgr = np.stack([img[:, :h, :] for img in imgs])
        generateds_bgr = np.stack([img[:, h:, :] for img in imgs])

        targets = torch.from_numpy(targets_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
        generateds = torch.from_numpy(generateds_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0

        B = targets.shape[0]

        ssim_vals = _ssim_fn(generateds, targets, data_range=1.0, size_average=False)
        ssim_sum += ssim_vals.sum().item()

        gen_lpips = (generateds.to(device) * 2.0 - 1.0)
        tgt_lpips = (targets.to(device) * 2.0 - 1.0)
        lpips_vals = lpips_fn(gen_lpips, tgt_lpips)
        lpips_sum += lpips_vals.sum().item()

        l1_vals = torch.abs(generateds - targets).mean(dim=(1, 2, 3))
        l1_sum += l1_vals.sum().item()

        all_targets_rgb.append(targets)
        all_generateds_rgb.append(generateds)

        count += B

    all_targets_rgb = torch.cat(all_targets_rgb, dim=0)
    all_generateds_rgb = torch.cat(all_generateds_rgb, dim=0)
    fid_val = compute_fid(all_targets_rgb, all_generateds_rgb, device=device, batch_size=batch_size)

    return {
        'ssim': ssim_sum / count,
        'lpips': lpips_sum / count,
        'l1': l1_sum / count,
        'fid': fid_val,
        'num_images': count,
    }
