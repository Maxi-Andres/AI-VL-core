#!/usr/bin/env python3
"""
perception3d.py — where a thing the camera sees IS, in metres, in the G1's torso frame.

The grasp so far replays poses recorded by hand, and the first time the robot stood 10 cm
further from the table it closed its hand on the table (2026-11-03). To compute a pose instead,
the arm needs the bottle's position in the frame its own chain hangs from — torso_link. This
module is that math, and only that: no camera, no network, no model, so every step is tested
with numbers.

    a box in the COLOUR picture (YOLO, or the VLM)
      + the DEPTH snapshot in millimetres (robot's GET /camera/depth)
      + the RealSense calibration (depth and colour intrinsics, depth->colour extrinsics)
      -> the depth pixels whose 3D point projects inside the box
      -> the object's front surface, robustly (the box also holds table and wall)
      -> depth optical frame -> d435_link -> torso_link (the URDF mount)

WHY PROJECT DEPTH INTO COLOUR, AND NOT THE OTHER WAY. The two sensors are ~15 mm apart and see
different pictures; a colour pixel has no depth of its own. Every depth pixel, on the other hand,
IS a 3D point, and projecting it into the colour picture asks a question with one answer: does
it land inside the box. The points stay in the depth camera's frame, which is where d435_link
starts.

THE MOUNT is Unitree's own URDF (g1_29dof, d435_joint): parent torso_link, xyz (0.0576235,
0.01753, 0.42987), rpy (0, 0.8307767, 0) — 47.6 degrees pitched down. Both arms hang from
torso_link too, so a point in this frame is what an arm solver needs, whatever the waist does.

THE CALIBRATION comes from the camera itself (robot-video-pipeline/robot/realsense_calib.py,
run once on PC2). Until that file exists, NOMINAL is used: the D435's data-sheet geometry, good
to roughly a centimetre at a metre. A result says which one it used, so nothing downstream can
mistake a nominal number for a measured one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Camera models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera: pixel (u, v) = (fx * x / z + ppx, fy * y / z + ppy), optical frame
    (x right, y down, z forward). The D435 streams used here have no distortion model (depth)
    or a negligible one (colour, Brown-Conrady with ~0 coefficients), so none is applied."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    def scaled(self, width, height):
        """The same lens at another resolution of the SAME aspect: the colour picture can reach
        the PC resized (the RTSP is pinned to WIDTH x HEIGHT on the robot)."""
        if abs(width / height - self.width / self.height) > 0.01:
            raise ValueError(f"{width}x{height} is not the aspect of the calibrated "
                             f"{self.width}x{self.height}: a crop, not a resize")
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.ppx * sx,
                          self.ppy * sy)


@dataclass(frozen=True)
class Calibration:
    depth: Intrinsics
    color: Intrinsics
    rotation: tuple       # depth -> colour, 3x3, row-major
    translation: tuple    # depth -> colour, metres, in the depth optical frame
    nominal: bool = False
    source: str = ""

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        ext = d["depth_to_color"]
        return cls(Intrinsics(**d["depth"]), Intrinsics(**d["color"]),
                   tuple(ext["rotation"]), tuple(ext["translation"]),
                   nominal=False, source=path)


# The D435's data-sheet geometry: depth 87 x 58 degrees at 848x480, colour 69 x 42 at 1280x720,
# the colour imager 15 mm to the right of the depth origin (the left IR imager), same axes. Only
# until the camera's own calibration has been read (see the module docstring).
NOMINAL = Calibration(
    depth=Intrinsics(848, 480, 424.0 / math.tan(math.radians(87 / 2)),
                     240.0 / math.tan(math.radians(58 / 2)), 424.0, 240.0),
    color=Intrinsics(1280, 720, 640.0 / math.tan(math.radians(69 / 2)),
                     360.0 / math.tan(math.radians(42 / 2)), 640.0, 360.0),
    rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    translation=(0.015, 0.0, 0.0),
    nominal=True, source="D435 data sheet",
)


def load_calibration(path=None):
    """The camera's own calibration if `path` exists, else NOMINAL."""
    if path:
        try:
            return Calibration.from_json(path)
        except FileNotFoundError:
            pass
    return NOMINAL


# --------------------------------------------------------------------------- #
# Depth -> points -> the box
# --------------------------------------------------------------------------- #
def depth_points(z, intr, scale_m=0.001, step=1):
    """uint16 depth (H x W) -> (N x 3 points in metres, depth optical frame). 0 = no reading,
    dropped. `step` subsamples the grid (2 = every other pixel): a box needs hundreds of points,
    not a full frame's 400 thousand."""
    if z.shape != (intr.height, intr.width):
        raise ValueError(f"depth is {z.shape[1]}x{z.shape[0]}, the calibration is "
                         f"{intr.width}x{intr.height}")
    v, u = np.mgrid[0:z.shape[0]:step, 0:z.shape[1]:step]
    d = z[::step, ::step].astype(np.float64) * scale_m
    keep = d > 0
    u, v, d = u[keep], v[keep], d[keep]
    return np.column_stack(((u - intr.ppx) / intr.fx * d, (v - intr.ppy) / intr.fy * d, d))


def project(points, intr):
    """N x 3 points in a camera's optical frame -> N x 2 pixels (u, v) in its picture."""
    z = points[:, 2]
    return np.column_stack((points[:, 0] / z * intr.fx + intr.ppx,
                            points[:, 1] / z * intr.fy + intr.ppy))


def to_color_frame(points, calib):
    r = np.asarray(calib.rotation, dtype=np.float64).reshape(3, 3)
    return points @ r.T + np.asarray(calib.translation, dtype=np.float64)


def shrink_box(box, keep=0.5):
    """The centre `keep` of a box (each side). The edges of a detection hold the background —
    a bottle's box is half table and wall at its borders — and the centre holds the object."""
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * keep / 2, (y2 - y1) * keep / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def points_in_box(points, calib, box, color_size=None, keep=0.5):
    """The depth points (depth optical frame) that land inside `box`, a colour-picture box in
    pixels of a picture `color_size` = (w, h) (the calibrated size if None)."""
    color = calib.color if color_size is None else calib.color.scaled(*color_size)
    uv = project(to_color_frame(points, calib), color)
    x1, y1, x2, y2 = shrink_box(box, keep)
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    return points[inside]


# The object is the NEAREST surface in its box, and this thick: a bottle seen from the front is
# a few centimetres of curved surface; anything further back in the box is the wall behind it.
SURFACE_BAND_M = 0.04
# Fewer points than this and the answer is noise, not a surface.
MIN_POINTS = 30


def front_surface(points, band=SURFACE_BAND_M, near_pct=10):
    """(median point of the object's front surface, points used, spread in metres), or None.

    The front is found from the NEAR end of the range, not its median: in a box that is 60%
    wall the median is the wall. near_pct (the 10th percentile), not the minimum, so a few
    flying pixels in front of the object do not become the object."""
    if len(points) < MIN_POINTS:
        return None
    rng = np.linalg.norm(points, axis=1)
    front = np.percentile(rng, near_pct)
    sel = points[(rng >= front - band / 2) & (rng <= front + band)]
    if len(sel) < MIN_POINTS:
        return None
    centre = np.median(sel, axis=0)
    spread = float(np.median(np.linalg.norm(sel - centre, axis=1)))
    return centre, len(sel), spread


# --------------------------------------------------------------------------- #
# Camera -> torso
# --------------------------------------------------------------------------- #
# Unitree's g1_29dof URDF, d435_joint (parent torso_link, child d435_link).
D435_XYZ = (0.0576235, 0.01753, 0.42987)
D435_RPY = (0.0, 0.8307767239493009, 0.0)


def rpy_matrix(roll, pitch, yaw):
    """URDF rpy -> rotation matrix (fixed axes X, then Y, then Z: R = Rz @ Ry @ Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


# Optical (x right, y down, z forward) -> link (x forward, y left, z up), the ROS convention.
OPTICAL_TO_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def camera_to_torso(p_optical, xyz=D435_XYZ, rpy=D435_RPY):
    """A point in the depth optical frame -> torso_link (x forward, y left, z up), metres."""
    p_link = OPTICAL_TO_LINK @ np.asarray(p_optical, dtype=np.float64)
    return rpy_matrix(*rpy) @ p_link + np.asarray(xyz, dtype=np.float64)


def locate(z, box, calib, color_size=None, scale_m=0.001, keep=0.5):
    """The whole chain: depth snapshot + colour box -> a dict with the torso-frame position.

    {"ok": True, "torso": [x, y, z], "camera": [x, y, z], "range_m", "points", "spread_m",
     "calibration": "measured"|"nominal"} — or {"ok": False, "error": why}."""
    pts = depth_points(z, calib.depth, scale_m, step=2)
    inside = points_in_box(pts, calib, box, color_size, keep)
    found = front_surface(inside)
    if found is None:
        return {"ok": False, "points": len(inside),
                "error": f"only {len(inside)} depth readings inside the box (glass, black "
                         f"cloth, or closer than the D435's ~0.2 m minimum)"
                if len(inside) < MIN_POINTS else "no surface stands out inside the box"}
    centre, n, spread = found
    torso = camera_to_torso(centre)
    return {"ok": True,
            "torso": [round(float(v), 4) for v in torso],
            "camera": [round(float(v), 4) for v in centre],
            "range_m": round(float(np.linalg.norm(centre)), 4),
            "points": n, "spread_m": round(spread, 4),
            "calibration": "nominal" if calib.nominal else "measured"}
