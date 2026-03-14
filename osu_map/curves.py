"""Curve sampling for osu! slider types: Linear, Bezier, Perfect Circle, Catmull-Rom."""

from __future__ import annotations

import numpy as np

EVAL_STEP = 1.0   # osu!px between raw curve evaluation points (accuracy)
MESH_STEP = 3.0   # osu!px between output path vertices (lower = smoother SDF contours)


def sample_curve(curve_type: str, points: np.ndarray, pixel_length: float) -> np.ndarray:
    """Dispatch to the correct sampler and return evenly-spaced path points.

    Parameters
    ----------
    curve_type : str
        One of "Linear", "Bezier", "Pass-Through", "Catmull".
    points : ndarray (K, 2)
        All control points **including** the slider start position.
    pixel_length : float
        Total slider length in osu! pixels.

    Returns
    -------
    ndarray (M, 2) float32
        Evenly spaced points along the path, clipped to *pixel_length*.
    """
    if len(points) < 2:
        return points.astype(np.float32)

    dispatch = {
        "Linear": _sample_linear,
        "Bezier": _sample_bezier,
        "Pass-Through": _sample_perfect_circle,
        "Catmull": _sample_catmull,
    }
    fn = dispatch.get(curve_type, _sample_bezier)
    raw = fn(points.astype(np.float64))
    return _clip_and_resample(raw, pixel_length)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def _sample_linear(pts: np.ndarray) -> np.ndarray:
    """Walk along straight-line segments between successive points."""
    out: list[np.ndarray] = []
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        n_samples = max(int(seg_len / EVAL_STEP), 1)
        for j in range(n_samples):
            t = j / n_samples
            out.append(pts[i] + seg * t)
    out.append(pts[-1].copy())
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bezier (multi-segment, split on duplicate control points)
# ---------------------------------------------------------------------------

def _sample_bezier(pts: np.ndarray) -> np.ndarray:
    segments = _split_bezier_segments(pts)
    out: list[np.ndarray] = []
    for seg in segments:
        out.extend(_evaluate_bezier_segment(seg))
    if len(out) == 0:
        return pts.copy()
    return np.array(out, dtype=np.float64)


def _split_bezier_segments(pts: np.ndarray) -> list[np.ndarray]:
    """Split control points at duplicates into sub-curves."""
    segments: list[list[np.ndarray]] = []
    current: list[np.ndarray] = [pts[0]]
    for i in range(1, len(pts)):
        if np.allclose(pts[i], pts[i - 1], atol=0.1):
            if len(current) >= 2:
                segments.append(np.array(current))
            current = [pts[i]]
        else:
            current.append(pts[i])
    if len(current) >= 2:
        segments.append(np.array(current))
    return segments


def _evaluate_bezier_segment(ctrl: np.ndarray) -> list[np.ndarray]:
    """De Casteljau evaluation of a single bezier segment."""
    approx_len = float(np.sum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1)))
    n_samples = max(int(approx_len / EVAL_STEP), 2)
    result = []
    for i in range(n_samples + 1):
        t = i / n_samples
        result.append(_de_casteljau(ctrl, t))
    return result


def _de_casteljau(pts: np.ndarray, t: float) -> np.ndarray:
    work = pts.copy()
    n = len(work)
    for r in range(1, n):
        for i in range(n - r):
            work[i] = (1 - t) * work[i] + t * work[i + 1]
    return work[0].copy()


# ---------------------------------------------------------------------------
# Perfect Circle (pass-through 3 points)
# ---------------------------------------------------------------------------

def _sample_perfect_circle(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 3:
        return _sample_bezier(pts)

    p0, p1, p2 = pts[0], pts[1], pts[2]
    center = _circumcenter(p0, p1, p2)
    if center is None:
        return _sample_bezier(pts)

    radius = np.linalg.norm(p0 - center)
    a0 = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    a2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])

    if _angle_between(a0, a1, a2):
        direction = 1
    else:
        direction = -1

    total_angle = _sweep_angle(a0, a2, direction)
    arc_length = abs(total_angle) * radius
    n_samples = max(int(arc_length / EVAL_STEP), 2)

    out = []
    for i in range(n_samples + 1):
        frac = i / n_samples
        angle = a0 + total_angle * frac
        out.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(out, dtype=np.float64)


def _circumcenter(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) +
          (bx * bx + by * by) * (cy - ay) +
          (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) +
          (bx * bx + by * by) * (ax - cx) +
          (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array([ux, uy])


def _angle_between(a0: float, a1: float, a2: float) -> bool:
    """Check if a1 lies in the counter-clockwise arc from a0 to a2."""
    d1 = (a1 - a0) % (2 * np.pi)
    d2 = (a2 - a0) % (2 * np.pi)
    return d1 <= d2


def _sweep_angle(a0: float, a2: float, direction: int) -> float:
    diff = (a2 - a0) % (2 * np.pi)
    if direction == 1:
        return diff if diff > 0 else diff + 2 * np.pi
    else:
        return -(2 * np.pi - diff) if diff > 0 else diff


# ---------------------------------------------------------------------------
# Catmull-Rom
# ---------------------------------------------------------------------------

def _sample_catmull(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return pts.copy()

    out: list[np.ndarray] = []
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[min(i + 1, n - 1)]
        p3 = pts[min(i + 2, n - 1)]

        seg_len = float(np.linalg.norm(p2 - p1))
        n_samples = max(int(seg_len / EVAL_STEP), 2)
        for j in range(n_samples):
            t = j / n_samples
            out.append(_catmull_point(p0, p1, p2, p3, t))
    out.append(pts[-1].copy())
    return np.array(out, dtype=np.float64)


def _catmull_point(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


# ---------------------------------------------------------------------------
# Clip to pixel_length and resample to MESH_STEP spacing
# ---------------------------------------------------------------------------

def _clip_and_resample(raw: np.ndarray, pixel_length: float) -> np.ndarray:
    """Clip path to pixel_length and resample to even MESH_STEP spacing."""
    if len(raw) < 2:
        return raw.astype(np.float32)

    diffs = np.diff(raw, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total < 1e-6:
        return raw[:1].astype(np.float32)

    target_len = min(pixel_length, total)
    n_out = max(int(target_len / MESH_STEP), 2)
    even_dists = np.linspace(0, target_len, n_out)

    out = np.empty((n_out, 2), dtype=np.float64)
    idx = 0
    for i, d in enumerate(even_dists):
        while idx < len(cum_lengths) - 2 and cum_lengths[idx + 1] < d:
            idx += 1
        seg_d = cum_lengths[idx + 1] - cum_lengths[idx]
        if seg_d < 1e-9:
            out[i] = raw[idx]
        else:
            t = (d - cum_lengths[idx]) / seg_d
            out[i] = raw[idx] * (1 - t) + raw[idx + 1] * t
    return out.astype(np.float32)
