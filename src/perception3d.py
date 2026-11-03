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


def color_ray_in_torso(u, v, calib, color_size=None):
    """(origin, unit direction) in torso_link of the colour camera's ray through pixel (u, v)."""
    color = calib.color if color_size is None else calib.color.scaled(*color_size)
    r = np.asarray(calib.rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(calib.translation, dtype=np.float64)
    d_col = np.array([(u - color.ppx) / color.fx, (v - color.ppy) / color.fy, 1.0])
    # colour optical -> depth optical is the inverse of depth -> colour: p_d = R^T (p_c - t)
    origin_d, dir_d = r.T @ (-t), r.T @ d_col
    origin = camera_to_torso(origin_d)
    direction = camera_to_torso(origin_d + dir_d) - origin
    return origin, direction / np.linalg.norm(direction)


def fit_plane(points, tol=0.015, rounds=4):
    """(centroid, unit normal pointing up in +z, inliers) of the dominant plane, or None."""
    if len(points) < MIN_POINTS:
        return None
    inl = np.ones(len(points), bool)
    for _ in range(rounds):
        c = points[inl].mean(axis=0)
        n = np.linalg.svd(points[inl] - c)[2][2]
        n = n * (1 if n[2] >= 0 else -1)
        inl = np.abs((points - c) @ n) < tol
        if inl.sum() < MIN_POINTS:
            return None
    return c, n, inl


# A support is a TABLE only if it is near level in the torso frame. The torso itself leans a few
# degrees (4.7 on the G1 2026-11-03); a wall behind the object is ~90 off.
TABLE_MAX_TILT_DEG = 25.0
# The table is sampled in two bands beside the object's base: as wide as the box on each side,
# from 35% of the box's height above its bottom edge to 10% below it.
TABLE_BAND = (0.35, 0.10)


def torso_to_color_pixels(points, calib, color):
    """N x 3 torso-frame points -> N x 2 pixels of the colour picture `color`."""
    r = rpy_matrix(*D435_RPY)
    p_link = (np.asarray(points) - np.asarray(D435_XYZ)) @ r     # r.T @ p, row-wise
    p_opt = p_link @ OPTICAL_TO_LINK                              # OPTICAL_TO_LINK.T @ p
    return project(to_color_frame(p_opt, calib), color)


_RING = np.linspace(0, 2 * np.pi, 36, endpoint=False)


def cylinder_box(base, up, height, radius, calib, color):
    """The colour-picture box of an upright cylinder standing at `base` along `up`."""
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ring = radius * (np.outer(np.cos(_RING), e1) + np.outer(np.sin(_RING), e2))
    pts = np.concatenate((base + ring, base + height * up + ring))
    uv = torso_to_color_pixels(pts, calib, color)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def fit_cylinder(box, plane_c, up, base0, height0, radius0, calib, color, iters=30):
    """(base, height, radius): the upright cylinder on the plane (plane_c, up) whose picture's
    box is `box` — Gauss-Newton on the 4 box edges, 4 unknowns (2 on the plane, height,
    radius). None if it does not converge to within ~2 px."""
    e1 = np.cross(up, [0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    base0 = base0 - ((base0 - plane_c) @ up) * up                 # onto the plane
    x = np.array([0.0, 0.0, height0, radius0])
    target = np.asarray(box, dtype=np.float64)

    def model(x):
        b = base0 + x[0] * e1 + x[1] * e2
        return cylinder_box(b, up, max(x[2], 0.01), max(x[3], 0.003), calib, color)

    for _ in range(iters):
        r = model(x) - target
        jac = np.empty((4, 4))
        for k, step in enumerate((1e-3, 1e-3, 1e-3, 1e-4)):
            dx = np.zeros(4)
            dx[k] = step
            jac[:, k] = (model(x + dx) - model(x)) / step
        delta = np.linalg.lstsq(jac, -r, rcond=None)[0]
        x = x + np.clip(delta, -0.05, 0.05)
        if np.abs(delta).max() < 1e-5:
            break
    if np.abs(model(x) - target).max() > 2.0 or x[2] <= 0.01 or x[3] <= 0.003:
        return None
    return base0 + x[0] * e1 + x[1] * e2, float(x[2]), float(x[3])


def standing_on_table(points, box, calib, color_size=None):
    """Where an object STANDING on a table is, from the table alone — or None.

    For what the depth camera cannot see: clear plastic, glass, black. MEASURED on the G1
    2026-11-03 with a 23 cm clear water bottle: 62 readings inside its box, nearly all of the
    table seen through it. But the table around its base reads perfectly, so: fit the table
    plane beside the base, cross it with the colour ray through the box's bottom-centre (the
    front edge of the base), step back half the box's width (the base is round, the edge is its
    front), and cross the ray through the top-centre with the vertical above that point for the
    height. That run gave 0.238 m for the 0.23 m bottle."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    color = calib.color if color_size is None else calib.color.scaled(*color_size)
    uv = project(to_color_frame(points, calib), color)
    up, down = TABLE_BAND
    rows = (uv[:, 1] >= y2 - up * h) & (uv[:, 1] <= y2 + down * h)
    left = (uv[:, 0] >= x1 - w) & (uv[:, 0] < x1 - 0.05 * w)
    right = (uv[:, 0] > x2 + 0.05 * w) & (uv[:, 0] <= x2 + w)
    ring = points[rows & (left | right)]
    torso = np.array([camera_to_torso(p) for p in ring]) if len(ring) else ring
    plane = fit_plane(torso)
    if plane is None:
        return None
    c, n, inl = plane
    if np.degrees(np.arccos(min(1.0, n[2]))) > TABLE_MAX_TILT_DEG:
        return None                          # a wall or a slope, not something to stand on
    o, e = color_ray_in_torso((x1 + x2) / 2, y2, calib, color_size)
    denom = e @ n
    if abs(denom) < 1e-3:
        return None                          # looking along the table: no crossing
    # A first guess from the box's bottom-centre and width, then the real fit: the upright
    # cylinder ON this plane whose picture has exactly this box. The guess alone is biased —
    # perspective leans a vertical object off-centre, so the box is wider than the object and
    # its centre is not the base's (synthetic bottle: radius 6.9 cm for 3.2, 6 cm sideways).
    front = o + ((c - o) @ n) / denom * e
    rng = float(np.linalg.norm(front - o))
    o2, e2 = color_ray_in_torso((x1 + x2) / 2, y1, calib, color_size)
    _a, h0 = np.linalg.lstsq(np.column_stack((e2, -n)), front - o2, rcond=None)[0]
    fit = fit_cylinder(box, c, n, front, max(float(h0), 0.05), w / color.fx * rng / 4,
                       calib, color)
    if fit is None:
        return None
    base, height, radius = fit
    return {"base": base, "height": float(height), "radius": radius, "up": n, "table_c": c,
            "table_points": int(inl.sum()),
            "table_tilt_deg": float(np.degrees(np.arccos(min(1.0, n[2]))))}





def locate(z, box, calib, color_size=None, scale_m=0.001, keep=0.5):
    """The whole chain: depth snapshot + colour box -> a dict with the torso-frame position.

    "method": "table" when the object stands on a table: "torso" is the centre of its base,
    "grasp" the middle of its axis, "top", "height_m", "radius_m", and "surface_check" (its own
    readings, with how far off the axis they are). "method": "surface" otherwise: "torso" is
    its front surface, plus "camera", "range_m", "points", "spread_m". Both carry
    "calibration": "measured"|"nominal". Or {"ok": False, "error": why}."""
    pts = depth_points(z, calib.depth, scale_m, step=2)
    inside = points_in_box(pts, calib, box, color_size, keep)
    found = front_surface(inside)
    stand = standing_on_table(pts, box, calib, color_size)
    cal_name = "nominal" if calib.nominal else "measured"
    surface = None
    if found is not None:
        centre, n, spread = found
        surface = {"torso": [round(float(v), 4) for v in camera_to_torso(centre)],
                   "camera": [round(float(v), 4) for v in centre],
                   "range_m": round(float(np.linalg.norm(centre)), 4),
                   "points": n, "spread_m": round(spread, 4)}
    # An object standing on a table is located BY the table whenever there is one: the table
    # reads cleanly, and the fit gives what an arm needs (the axis and the height). The object's
    # own readings come second because they lie on anything clear — MEASURED on the G1
    # 2026-11-03, clear 23 cm bottle: table fit 0.231 m tall, while its 58 "surface" readings
    # put its front 2-3 cm BEHIND its axis (refraction). They are kept as a cross-check.
    if stand is not None and 0.02 < stand["height"] < 0.6:
        base, up, tall = stand["base"], stand["up"], stand["height"]
        out = {"ok": True, "method": "table",
               "torso": [round(float(v), 4) for v in base],
               "grasp": [round(float(v), 4) for v in base + tall / 2 * up],
               "top": [round(float(v), 4) for v in base + tall * up],
               "height_m": round(tall, 4), "radius_m": round(float(stand["radius"]), 4),
               "table_points": stand["table_points"],
               "table_tilt_deg": round(stand["table_tilt_deg"], 1), "calibration": cal_name}
        if surface is not None:
            p = np.asarray(surface["torso"]) - base
            off_axis = float(np.linalg.norm(p - (p @ up) * up))
            out["surface_check"] = {**surface, "off_axis_m": round(off_axis, 4)}
        return out
    if surface is not None:
        return {"ok": True, "method": "surface", **surface, "calibration": cal_name}
    return {"ok": False, "points": len(inside),
            "error": f"only {len(inside)} depth readings inside the box (glass, black cloth, "
                     f"or closer than the D435's ~0.2 m minimum), and no table around its base"
            if len(inside) < MIN_POINTS else
            "no surface stands out inside the box, and no table around its base"}
