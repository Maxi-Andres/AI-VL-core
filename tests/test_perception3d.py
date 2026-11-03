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
