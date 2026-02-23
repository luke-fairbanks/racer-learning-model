"""Track generation, save/load, and data structures."""

import json
import math
import os
from dataclasses import dataclass

import numpy as np


# -------------------------
# Track data
# -------------------------
@dataclass
class Track:
    pts: np.ndarray          # (N,2) float
    seg_len: np.ndarray      # (N,) length of each segment i->i+1 (wrap)
    cum_len: np.ndarray      # (N+1,) cumulative lengths, cum_len[0]=0, cum_len[N]=total
    total: float
    half_width: float


# -------------------------
# Catmull-Rom spline
# -------------------------
def _catmull_rom(p0, p1, p2, p3, n_pts, alpha=0.5):
    """Centripetal Catmull-Rom spline between p1 and p2."""
    def tj(ti, pi, pj):
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        d = math.sqrt(dx * dx + dy * dy)
        return ti + max(d, 1e-6) ** alpha

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    pts = []
    for i in range(n_pts):
        t = t1 + (t2 - t1) * i / n_pts

        a1x = (t1 - t) / (t1 - t0 + 1e-8) * p0[0] + (t - t0) / (t1 - t0 + 1e-8) * p1[0]
        a1y = (t1 - t) / (t1 - t0 + 1e-8) * p0[1] + (t - t0) / (t1 - t0 + 1e-8) * p1[1]
        a2x = (t2 - t) / (t2 - t1 + 1e-8) * p1[0] + (t - t1) / (t2 - t1 + 1e-8) * p2[0]
        a2y = (t2 - t) / (t2 - t1 + 1e-8) * p1[1] + (t - t1) / (t2 - t1 + 1e-8) * p2[1]
        a3x = (t3 - t) / (t3 - t2 + 1e-8) * p2[0] + (t - t2) / (t3 - t2 + 1e-8) * p3[0]
        a3y = (t3 - t) / (t3 - t2 + 1e-8) * p2[1] + (t - t2) / (t3 - t2 + 1e-8) * p3[1]

        b1x = (t2 - t) / (t2 - t0 + 1e-8) * a1x + (t - t0) / (t2 - t0 + 1e-8) * a2x
        b1y = (t2 - t) / (t2 - t0 + 1e-8) * a1y + (t - t0) / (t2 - t0 + 1e-8) * a2y
        b2x = (t3 - t) / (t3 - t1 + 1e-8) * a2x + (t - t1) / (t3 - t1 + 1e-8) * a3x
        b2y = (t3 - t) / (t3 - t1 + 1e-8) * a2y + (t - t1) / (t3 - t1 + 1e-8) * a3y

        cx = (t2 - t) / (t2 - t1 + 1e-8) * b1x + (t - t1) / (t2 - t1 + 1e-8) * b2x
        cy = (t2 - t) / (t2 - t1 + 1e-8) * b1y + (t - t1) / (t2 - t1 + 1e-8) * b2y
        pts.append((cx, cy))
    return pts


# -------------------------
# Default track layout
# -------------------------
DEFAULT_CONTROL_PTS = [
    (400, 0),       # start/finish straight
    (550, -10),     # approach turn 1
    (600, -80),     # turn 1 entry
    (580, -180),    # turn 1 apex
    (480, -260),    # exit turn 1
    (300, -310),    # back straight entry
    (50, -330),     # back straight mid
    (-150, -300),   # back straight end
    (-300, -220),   # turn 3 entry
    (-380, -100),   # turn 3 apex
    (-360, 40),     # turn 3 exit
    (-280, 140),    # flowing left
    (-160, 190),    # wide apex
    (-20, 170),     # exit
    (100, 100),     # kink entry
    (180, 30),      # kink
    (260, -10),     # acceleration zone
]


# -------------------------
# Track builder
# -------------------------
def make_track(control_pts=None, half_width=42.0, **_kwargs):
    """
    Generates a closed racing circuit using Catmull-Rom splines.
    If control_pts is None, uses the default circuit layout.
    """
    if control_pts is None:
        control_pts = DEFAULT_CONTROL_PTS

    if len(control_pts) < 3:
        raise ValueError("Need at least 3 control points to make a track")

    N = len(control_pts)
    pts_per_segment = 12
    all_pts = []

    for i in range(N):
        p0 = control_pts[(i - 1) % N]
        p1 = control_pts[i]
        p2 = control_pts[(i + 1) % N]
        p3 = control_pts[(i + 2) % N]
        segment_pts = _catmull_rom(p0, p1, p2, p3, pts_per_segment)
        all_pts.extend(segment_pts)

    pts = np.array(all_pts, dtype=np.float32)

    nxt = np.roll(pts, -1, axis=0)
    seg = np.linalg.norm(nxt - pts, axis=1).astype(np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(cum[-1])
    return Track(pts=pts, seg_len=seg, cum_len=cum, total=total, half_width=float(half_width))


# -------------------------
# Save / Load track JSON
# -------------------------
def save_track_json(control_pts, filepath, half_width=42.0):
    """Save track control points to a JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    data = {
        "control_pts": [list(p) for p in control_pts],
        "half_width": half_width,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_track_json(filepath):
    """Load track control points from a JSON file. Returns (control_pts, half_width)."""
    with open(filepath) as f:
        data = json.load(f)
    pts = [tuple(p) for p in data["control_pts"]]
    hw = data.get("half_width", 42.0)
    return pts, hw


def list_tracks(tracks_dir="tracks"):
    """List available track files. Always includes 'default' as option 0."""
    tracks = [{"name": "default", "path": None}]
    if os.path.isdir(tracks_dir):
        for f in sorted(os.listdir(tracks_dir)):
            if f.endswith(".json"):
                name = os.path.splitext(f)[0]
                tracks.append({"name": name, "path": os.path.join(tracks_dir, f)})
    return tracks


def generate_random_track(rng=None, half_width=42.0):
    """
    Generate a random closed racing circuit.
    Places 8-16 control points on a randomly-shaped ellipse
    with radial and angular perturbations for variety.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_pts = rng.integers(8, 17)  # 8-16 control points
    base_radius_x = rng.uniform(200, 500)
    base_radius_y = rng.uniform(200, 500)

    angles = np.sort(rng.uniform(0, 2 * math.pi, n_pts))
    # Ensure minimum angular spacing
    for _ in range(20):
        diffs = np.diff(np.append(angles, angles[0] + 2 * math.pi))
        if np.min(diffs) > 0.3:
            break
        # Re-jitter tight points
        tight = np.where(diffs < 0.3)[0]
        angles[tight] += rng.uniform(-0.2, 0.2, len(tight))
        angles = np.sort(angles % (2 * math.pi))

    control_pts = []
    for angle in angles:
        # Radial perturbation (30-130% of base radius)
        r_scale = rng.uniform(0.3, 1.3)
        rx = base_radius_x * r_scale
        ry = base_radius_y * r_scale
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        # Additional jitter
        x += rng.uniform(-40, 40)
        y += rng.uniform(-40, 40)
        control_pts.append((float(x), float(y)))

    return make_track(control_pts=control_pts, half_width=half_width)

