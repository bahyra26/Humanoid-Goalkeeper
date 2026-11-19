"""Run with python -m unittest discover -s tests (no simulator required)."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'goal_detection', ROOT / 'legged_gym/legged_gym/utils/goal_detection.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GoalDetectionTests(unittest.TestCase):
    def crossed(self, previous, current):
        return MODULE.ball_crossed_goal(previous, current, -0.05, 1.3, 1.2, 0.1)

    def test_center_crossing_is_not_enough(self):
        self.assertFalse(self.crossed((0.2, 0, 0.5), (-0.10, 0, 0.5)))

    def test_whole_ball_crosses(self):
        self.assertTrue(self.crossed((-0.10, 0, 0.5), (-0.20, 0, 0.5)))

    def test_touching_line_is_not_a_goal(self):
        threshold = -0.05 - 0.10
        self.assertFalse(self.crossed((0.2, 0, 0.5), (threshold, 0, 0.5)))
        self.assertTrue(self.crossed((threshold, 0, 0.5), (-0.2, 0, 0.5)))

    def test_side_miss_then_movement_behind_goal(self):
        self.assertFalse(self.crossed((0, 1.4, 0.5), (-0.3, 1.4, 0.5)))
        self.assertFalse(self.crossed((-0.3, 1.4, 0.5), (-0.4, 0, 0.5)))

    def test_radius_at_posts_and_crossbar(self):
        for y, z in [(1.25, 0.5), (-1.25, 0.5), (0, 1.15), (0, 0.05)]:
            with self.subTest(y=y, z=z):
                self.assertFalse(self.crossed((0, y, z), (-0.3, y, z)))

    def test_ground_ball(self):
        self.assertTrue(self.crossed((0, 0, 0.1), (-0.3, 0, 0.1)))
        self.assertTrue(self.crossed((0, 0, 0.098), (-0.3, 0, 0.098)))

    def test_size1_ball_threshold_and_crossbar(self):
        # At radius=0.075 m, crossing threshold is -0.125 m and crossbar limit is 1.125 m.
        self.assertFalse(MODULE.ball_crossed_goal((0.0, 0.0, 0.5), (-0.125, 0.0, 0.5), -0.05, 1.3, 1.2, 0.075))
        self.assertTrue(MODULE.ball_crossed_goal((-0.125, 0.0, 1.105), (-0.20, 0.0, 1.105), -0.05, 1.3, 1.2, 0.075))
        self.assertFalse(MODULE.ball_crossed_goal((0.0, 0.0, 1.13), (-0.20, 0.0, 1.13), -0.05, 1.3, 1.2, 0.075))
        self.assertTrue(MODULE.ball_crossed_goal((0.0, 0.0, 0.075), (-0.20, 0.0, 0.075), -0.05, 1.3, 1.2, 0.075))

    def test_crossing_position_not_end_position(self):
        self.assertTrue(self.crossed((0, 0, 0.5), (-0.3, 2.0, 0.5)))
        self.assertFalse(self.crossed((0, 2.6, 0.5), (-0.3, 0, 0.5)))
        self.assertTrue(self.crossed((0, 0, 0.5), (-0.3, 0, 1.5)))

    def test_outward_stationary_and_already_behind(self):
        for previous, current in [((-0.3, 0, 0.5), (0, 0, 0.5)),
                                  ((0, 0, 0.5), (0, 0, 0.5)),
                                  ((-0.3, 0, 0.5), (-0.4, 0, 0.5))]:
            with self.subTest(previous=previous, current=current):
                self.assertFalse(self.crossed(previous, current))


if __name__ == '__main__':
    unittest.main()
