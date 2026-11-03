"""perception3d — a colour box + the depth snapshot -> a point in the G1's torso frame.

Every test names the defect it catches. Synthetic depth only: no camera, no robot. The scenes
are built in the depth camera's own geometry, so the right answer is known exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import perception3d as p3

CAL = p3.NOMINAL
W, H = CAL.depth.width, CAL.depth.height


def wall(mm):
    return np.full((H, W), mm, np.uint16)


def color_box_around(point, half_px=40):
    """The colour-picture box centred on where a depth-frame point appears."""
    uv = p3.project(p3.to_color_frame(np.array([point]), CAL), CAL.color)[0]
    return (uv[0] - half_px, uv[1] - half_px, uv[0] + half_px, uv[1] + half_px)


def test_a_wall_straight_ahead_is_at_its_distance():
    """The defect: a units or axis slip (mm vs m, x vs z) — the simplest scene must be exact."""
    out = p3.locate(wall(1000), color_box_around((0, 0, 1.0)), CAL)
    assert out["ok"]
    assert out["camera"][2] == pytest.approx(1.0, abs=0.005)
    assert abs(out["camera"][0]) < 0.02 and abs(out["camera"][1]) < 0.02
    assert out["calibration"] == "nominal"


def test_the_object_is_the_near_surface_not_the_wall_behind_it():
    """The defect: a median over the box — a bottle's box is mostly wall at its borders, and the
    median lands on the wall, 1 m behind the bottle."""
    z = wall(1500)
    cu, cv = int(CAL.depth.ppx), int(CAL.depth.ppy)
    z[cv - 25:cv + 25, cu - 12:cu + 12] = 600            # a bottle, 0.6 m, ~5 x 7 cm
    out = p3.locate(z, color_box_around((0, 0, 0.6), half_px=60), CAL)
    assert out["ok"]
    assert out["camera"][2] == pytest.approx(0.6, abs=0.01)


def test_a_few_flying_pixels_in_front_do_not_become_the_object():
    """The defect: taking the MINIMUM range — D435 edges throw single pixels towards the camera."""
    z = wall(900)
    rng = np.random.default_rng(0)
    cu, cv = int(CAL.depth.ppx), int(CAL.depth.ppy)
    for _ in range(15):
        z[cv + rng.integers(-20, 20), cu + rng.integers(-20, 20)] = 300
    out = p3.locate(z, color_box_around((0, 0, 0.9)), CAL)
    assert out["camera"][2] == pytest.approx(0.9, abs=0.01)


def test_no_readings_in_the_box_is_an_error_with_a_reason():
    """The defect: glass (or a black bottle) giving no depth, and a grasp reaching for (0,0,0)."""
    z = wall(1000)
    cu, cv = int(CAL.depth.ppx), int(CAL.depth.ppy)
    z[cv - 60:cv + 60, cu - 60:cu + 60] = 0
    out = p3.locate(z, color_box_around((0, 0, 1.0), half_px=30), CAL)
    assert not out["ok"] and "readings" in out["error"]


def test_a_resized_colour_picture_scales_the_box_with_it():
    """The defect: a box from a 640x360 picture applied to 1280x720 intrinsics — it would point
    at the top-left quarter of the scene."""
    box = color_box_around((0, 0, 1.0))
    half = tuple(v / 2 for v in box)
    a = p3.locate(wall(1000), box, CAL)
    b = p3.locate(wall(1000), half, CAL, color_size=(640, 360))
    assert a["camera"] == pytest.approx(b["camera"], abs=0.005)


def test_a_crop_is_refused_not_scaled():
    with pytest.raises(ValueError, match="aspect"):
        CAL.color.scaled(640, 480)


def test_depth_of_another_size_than_the_calibration_is_refused():
    """The defect: a 424x240 depth (the streamed size) read with 848x480 intrinsics."""
    with pytest.raises(ValueError, match="calibration"):
        p3.depth_points(np.zeros((240, 424), np.uint16), CAL.depth)


def test_straight_ahead_of_the_camera_is_forward_and_down_in_the_torso():
    """The defect: the pitch sign. The head camera looks 47.6 degrees DOWN: a point 1 m along
    its axis is in front of the torso and below the camera, never above it."""
    pitch = p3.D435_RPY[1]
    t = p3.camera_to_torso((0.0, 0.0, 1.0))
    x0, y0, z0 = p3.D435_XYZ
    assert t[0] == pytest.approx(x0 + math.cos(pitch), abs=1e-9)
    assert t[1] == pytest.approx(y0, abs=1e-9)
    assert t[2] == pytest.approx(z0 - math.sin(pitch), abs=1e-9)


def test_right_in_the_picture_is_right_of_the_robot():
    """The defect: a mirrored axis — optical x is the picture's right, which is the robot's -y."""
    t = p3.camera_to_torso((0.2, 0.0, 1.0))
    assert t[1] < p3.D435_XYZ[1] - 0.15


def test_a_measured_calibration_file_is_used_and_said_so(tmp_path):
    f = tmp_path / "cal.json"
    f.write_text('{"depth": {"width": 848, "height": 480, "fx": 420, "fy": 420, "ppx": 424, '
                 '"ppy": 240}, "color": {"width": 1280, "height": 720, "fx": 910, "fy": 910, '
                 '"ppx": 640, "ppy": 360}, "depth_to_color": {"rotation": [1,0,0,0,1,0,0,0,1], '
                 '"translation": [0.015, 0, 0]}}')
    cal = p3.load_calibration(str(f))
    assert not cal.nominal and cal.depth.fx == 420
    assert p3.load_calibration(str(tmp_path / "missing.json")).nominal


# --------------------------------------------------------------------------- #
# Objects the camera cannot see: standing on a table
# --------------------------------------------------------------------------- #
TABLE_Z = -0.10          # the G1's desk, 2026-11-03: 0.098 m below the torso origin


def torso_to_optical(p):
    r = p3.rpy_matrix(*p3.D435_RPY)
    return p3.OPTICAL_TO_LINK.T @ (r.T @ (np.asarray(p) - np.asarray(p3.D435_XYZ)))


def table_scene(table_z=TABLE_Z):
    """Depth of a level table: each pixel's ray crossed with the plane z = table_z."""
    i = CAL.depth
    v, u = np.mgrid[0:i.height, 0:i.width]
    rays = np.stack(((u - i.ppx) / i.fx, (v - i.ppy) / i.fy, np.ones_like(u, float)), -1)
    o = p3.camera_to_torso((0, 0, 0))
    d = np.einsum("ij,hwj->hwi", p3.rpy_matrix(*p3.D435_RPY) @ p3.OPTICAL_TO_LINK, rays)
    s = (table_z - o[2]) / d[..., 2]
    z = np.where(s > 0, s * 1000, 0)               # optical z = s (rays have z_opt = 1)
    return np.clip(z, 0, 65000).astype(np.uint16)


def bottle_box(base, height=0.23, radius=0.032):
    """The colour-picture box of an upright cylinder standing at `base` (torso frame)."""
    a = np.linspace(0, 2 * np.pi, 48)
    ring = np.stack((np.cos(a) * radius, np.sin(a) * radius, np.zeros_like(a)), -1)
    pts = np.concatenate((ring + base, ring + base + (0, 0, height)))
    opt = np.array([torso_to_optical(p) for p in pts])
    uv = p3.project(p3.to_color_frame(opt, CAL), CAL.color)
    return (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())


def test_a_clear_bottle_is_found_by_the_table_it_stands_on():
    """The defect (G1, 2026-11-03): a clear water bottle — the camera reads the table THROUGH
    it, and the bottle was 'not found'. Its base and height come from the table around it."""
    base = np.array([0.32, -0.09, TABLE_Z])
    out = p3.locate(table_scene(), bottle_box(base), CAL)
    assert out["ok"] and out["method"] == "table"
    assert out["torso"] == pytest.approx(list(base), abs=0.015)
    assert out["height_m"] == pytest.approx(0.23, abs=0.02)


def test_the_table_seen_through_a_clear_object_is_not_taken_for_the_object():
    """The defect: the table behind a clear bottle passing as the bottle's front surface — a
    grasp aimed 10+ cm behind it. Same scene, so the answer must be the table method's."""
    base = np.array([0.30, 0.05, TABLE_Z])
    out = p3.locate(table_scene(), bottle_box(base), CAL)
    assert out["method"] == "table"


def test_an_object_on_a_table_reports_where_to_grasp_and_checks_its_own_surface():
    """The table answers for anything standing on it, clear or not; the object's own readings
    stay as a cross-check, so a disagreement is visible instead of silently picked from."""
    base = np.array([0.32, -0.09, TABLE_Z])
    box = bottle_box(base)
    z = table_scene()
    i = CAL.depth
    v, u = np.mgrid[0:i.height, 0:i.width]
    # The bottle made opaque: its front surface, a plane facing the camera at its axis.
    front = torso_to_optical(base + np.array([0, 0, 0.12]))
    du = p3.project(np.array([front]), i)[0]
    patch = (np.abs(u - du[0]) < 10) & (np.abs(v - du[1]) < 40)
    z[patch] = int(front[2] * 1000)
    out = p3.locate(z, box, CAL)
    assert out["method"] == "table"
    assert out["grasp"] == pytest.approx(list(base + np.array([0, 0, 0.115])), abs=0.015)
    assert out["surface_check"]["off_axis_m"] < 0.05


def test_with_no_table_the_object_is_measured_by_its_own_surface():
    """A wall behind, nothing to stand on (the first tests' scenes): the surface method."""
    out = p3.locate(wall(800), color_box_around((0, 0, 0.8)), CAL)
    assert out["method"] == "surface"
