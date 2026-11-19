import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/utils/shot_outcome.py'
spec = importlib.util.spec_from_file_location('outcome', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ShotTests(unittest.TestCase):
    def test_contact_is_not_save(self):
        self.assertFalse(m.resolved_save(True, False, .2, -.1, -.05, .1))

    def test_goal_overrides_contact(self):
        self.assertFalse(m.resolved_save(True, True, -.4, -1., -.05, .1))

    def test_safe_exit_requires_contact(self):
        self.assertTrue(m.resolved_save(True, False, -.4, -1., -.05, .1))
        self.assertFalse(m.resolved_save(False, False, -.4, -1., -.05, .1))
        self.assertTrue(m.resolved_save(True, False, 2.1, 1., -.05, .1))
        self.assertTrue(m.resolved_save(True, False, .8, .2, -.05, .1))
        self.assertTrue(m.resolved_save(True, False, .8, 0., -.05, .1))
        self.assertFalse(m.resolved_save(True, False, .0, 0., -.05, .1))
        self.assertFalse(m.resolved_save(True, False, 2.1, -1., -.05, .1))

    def test_slow_nearby_ball_not_contact(self):
        self.assertFalse(m.contact_candidate(.1, .1, 0., 0., 0., 10.))

    def test_impulse_needs_force_and_proximity(self):
        self.assertTrue(m.contact_candidate(.3, .1, 2., 0., 2., 10.))
        self.assertFalse(m.contact_candidate(.3, .1, 2., 0., 2., 0.))
        self.assertFalse(m.contact_candidate(.3, 1., 2., 0., 2., 10.))
        # Ground-bounce deceleration near feet (0.19 m) must not trigger low contact.
        self.assertFalse(m.contact_candidate(.10, .19, 3.6, .2, 3.6, 275.))
        self.assertTrue(m.contact_candidate(.11, .14, 3.6, .5, 3.6, 210.))
        # Size 1 ball (r=0.075): thresholds scale down by 2.5 cm with radius.
        self.assertFalse(m.contact_candidate(.08, .16, 3.6, .2, 3.6, 275., ball_radius=0.075))
        self.assertTrue(m.contact_candidate(.08, .12, 3.6, .5, 3.6, 210., ball_radius=0.075))
        self.assertFalse(m.contact_candidate(.30, .225, 2.0, .0, 2.0, 50., ball_radius=0.075))

    def test_side_clear_stopped_and_paired_partition(self):
        # Deflected wide past goalpost or stopped safely far in front of line resolves as save.
        self.assertTrue(m.resolved_save(True, False, .29, -.14, -.05, .1, y=2.11, half_width=1.30, speed=.15))
        self.assertTrue(m.resolved_save(True, False, .43, -.02, -.05, .1, y=-.27, half_width=1.30, speed=.03))
        self.assertFalse(m.resolved_save(True, False, -.15, -.02, -.05, .1, y=.1, half_width=1.30, speed=.02))

        self.assertEqual(m.shot_outcome_label(False, False, -.4, -10., -.05, .1, would_score=False), 'out_no_touch')
        self.assertEqual(m.shot_outcome_label(False, True, -.4, -9., -.05, .1, would_score=True), 'blocked_save')
        self.assertEqual(m.shot_outcome_label(False, True, -.4, -9., -.05, .1, would_score=False), 'off_target_touched')
        self.assertEqual(m.shot_outcome_label(True, True, -.4, -5., -.05, .1, would_score=True), 'goal')
        self.assertEqual(m.shot_outcome_label(False, True, -.15, -.02, -.05, .1, y=.1, half_width=1.30, speed=.02, would_score=True), 'unresolved')
