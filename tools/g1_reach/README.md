# g1_reach — first computed reaches on the G1 (2026-11-03)

Field scripts, not library code: they move the real right arm through the executor
(`/joints/control`), so run them only with someone watching the robot.

- `reach_stage1.py` — locate the bottle three times (must agree within 1.5 cm), raise the hand
  near the body above the table edge, extend to the pre-grasp, hold for a picture, come back,
  release. Refuses to move if any arm point past the table edge is under 4 cm above the table.
  First run: smooth, no table contact, **but the hand reached the bottle and knocked it over** —
  the arm model puts the hand ~8-10 cm short of where it really is, and the palm faced up.
- `hand_eye.py` — the fix's measurement: three free points above the table, a depth snapshot at
  each, the hand the camera sees against the model's. Not yet run: its table estimate disagreed
  with the previous run (z 0.038 vs -0.029) and its own check refused to move. Next session:
  take the table height from the located bottle (as `reach_stage1.py` does), not a histogram.

Lesson from the user: on the way up the hand must stay closer to the body.
