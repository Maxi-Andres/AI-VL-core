"""The drive pad groups the G1's arm actions by the locomotion mode each one needs.

Every test here names the defect it catches. Imports only command_common: no model, no robot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import command_common as cc


def _grouped():
    return [v for vs in cc.ARM_ACTION_GROUPS.values() for v in vs]


def test_every_arm_action_sits_in_exactly_one_group():
    """Catches: an action added to ARM_ACTION_IDS and not to a group — the pad would draw it
    under no heading, with no hint of the mode it needs — or listed under two modes."""
    grouped = _grouped()
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(cc.ARM_ACTION_IDS)


def test_the_app_actions_are_grouped_under_run():
    """Measured 2026-10-01: from Walk the robot accepts these and does nothing."""
    run = set(cc.ARM_ACTION_GROUPS["Run mode"])
    assert {"hug", "clap", "face_wave", "left_kiss", "heart", "hands_up", "x_ray",
            "right_hand_up", "reject", "shake_hand"} <= run


def test_the_catalog_hands_the_groups_to_the_ui():
    """Catches: the groups defined but never reaching GET /skills."""
    spec = cc.catalog("g1")["skills"]["arm_action"]["params"]["action"]
    assert spec["groups"] == cc.ARM_ACTION_GROUPS
