"""
Attribution Faithfulness Evaluation Metrics.
Matches the published Intelligence-Based Medicine (Elsevier 2026) paper protocol.
"""

from typing import Sequence
import numpy as np
from scipy.integrate import trapezoid


def top_mass_mask(attr_map: np.ndarray, mass: float = 0.50) -> np.ndarray:
    """
    Finds the brightest pixels that hold a given percentage of total glow.
    """
    flat = attr_map.astype(np.float32).ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    total = csum[-1]

    mask = np.zeros_like(flat, dtype=np.uint8)
    if total <= 0:
        return mask.reshape(attr_map.shape)

    cutoff = int(np.searchsorted(csum, mass * total, side="left")) + 1
    mask[order[:cutoff]] = 1
    return mask.reshape(attr_map.shape)


def rasterize_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    shape: tuple[int, int] = (224, 224),
    margin_frac: float = 0.0,
) -> np.ndarray:
    """
    Turns doctor's bounding box coordinates (xmin, ymin, xmax, ymax) 
    into a binary mask (1 inside box, 0 outside).
    
    Uses integer truncation when margin_frac == 0.0 to preserve <1e-4 
    parity with paper CSVs.
    """
    h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)

    for xmin, ymin, xmax, ymax in boxes:
        if margin_frac > 0.0:
            mrg_w = margin_frac * w
            mrg_h = margin_frac * h
            x1 = max(0, int(np.floor(xmin - mrg_w)))
            y1 = max(0, int(np.floor(ymin - mrg_h)))
            x2 = min(w, int(np.ceil(xmax + mrg_w)))
            y2 = min(h, int(np.ceil(ymax + mrg_h)))
        else:
            x1 = max(0, int(xmin))
            y1 = max(0, int(ymin))
            x2 = min(w, int(xmax))
            y2 = min(h, int(ymax))

        mask[y1:y2, x1:x2] = 1

    return mask


def precision_mass(attr_map: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    """
    Measures: What fraction of the AI's total attention landed inside the ground-truth box?
    Returns (precision_mass, null_precision, delta_precision).
    """
    ig_raw = np.abs(attr_map.astype(np.float64))
    if ig_raw.ndim == 3:
        ig_raw = ig_raw.sum(axis=0)

    total_sum = ig_raw.sum()
    b_mask = gt_mask.astype(bool)
    null_prec = float(b_mask.mean())

    if total_sum <= 1e-12:
        return 0.0, null_prec, -null_prec

    ig_norm = ig_raw / total_sum
    prec_mass = float(np.clip(ig_norm[b_mask].sum(), 0.0, 1.0))
    delta_prec = prec_mass - null_prec

    return prec_mass, null_prec, delta_prec


def recall_top50(
    attr_map: np.ndarray,
    gt_mask: np.ndarray,
    top50_mask: np.ndarray | None = None,
) -> float:
    """
    Measures: How much of the doctor's sick box was highlighted by the top-50% most energetic pixels?
    """
    b_mask = gt_mask.astype(bool)
    if b_mask.sum() == 0:
        return 0.0

    if top50_mask is not None:
        pred_mask = top50_mask.astype(bool)
    else:
        abs_map = np.abs(attr_map)
        if abs_map.ndim == 3:
            abs_map = abs_map.sum(axis=0)
        pred_mask = top_mass_mask(abs_map, 0.50).astype(bool)

    intersection = (pred_mask & b_mask).sum()
    return float(np.clip(intersection / b_mask.sum(), 0.0, 1.0))


def auc_miou(attr_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Sweeps 100 mass thresholds (1% to 100%) and integrates Area Under Mean IoU Curve.
    """
    b_mask = gt_mask.astype(bool)
    abs_map = np.abs(attr_map)
    if abs_map.ndim == 3:
        abs_map = abs_map.sum(axis=0)

    mass_fracs = np.linspace(0.01, 1.00, 100)
    miou_vals = np.zeros_like(mass_fracs)

    for i, f in enumerate(mass_fracs):
        pred_mask = top_mass_mask(abs_map, float(f)).astype(bool)
        intersection = (pred_mask & b_mask).sum()
        union = (pred_mask | b_mask).sum()
        miou_vals[i] = (intersection / union) if union > 0 else 0.0

    return float(np.clip(trapezoid(miou_vals, dx=0.01) / 0.99, 0.0, 1.0))
