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

## First full grasp (2026-11-03, `grasp_side.py`)

Lifted the bottle and put it back. The way up is TAUGHT (`taught_poses.json`, from
`capture_pose.py <name>` while the user holds the arm in the Joints page): hanging -> way_up_2
(wrist folded in) -> way_up_3 -> way_up_4 (forearm level, hand before the belly) -> ready, the
hand folded as the user set it (FIST). From there the planner straightens the wrist with the
forearm raised (also the relay's safe pose), the IK goes to the bottle's right, the hand opens
with the arm still, slides in, closes by touch, lifts, puts it down, lets go, and returns the
same way. Every segment is checked against the table edge where the right arm passes and
against the bottle, with a pessimistic hand (fist while folded), and so is the relay's straight
retreat to the safe pose. What failed on the way there, and why: see the commits of that day.
