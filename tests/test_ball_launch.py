import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/utils/ball_launch.py'
spec = importlib.util.spec_from_file_location('ball_launch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BallLaunchTests(unittest.TestCase):
    def test_speed_and_target_for_multiple_heights_and_sides(self):
        delta = torch.tensor([[-3.42, -.6, .02], [-3.42, .6, .77], [-3.42, 0., -.1]])
        velocity = module.launch_at_speed(delta, 7.)
        torch.testing.assert_close(torch.norm(velocity, dim=1), torch.full((3,), 7.))
        time = delta[:, :1] / velocity[:, :1]
        arrived = velocity * time
        arrived[:, 2:3] -= .5 * 9.81 * time.square()
        torch.testing.assert_close(arrived, delta)
        self.assertTrue((time > 0).all())

    def test_invalid_and_unreachable_speed(self):
        delta = torch.tensor([[-3.42, 0., .5]])
        for speed in (0., -1., float('nan'), float('inf'), 1.):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                module.launch_at_speed(delta, speed)

    def test_minimum_speed_and_high_speed_are_finite(self):
        delta = torch.tensor([[-3., 0., 0.]], dtype=torch.float64)
        for speed in ((9.81 * 3.)**.5, 100.):
            velocity = module.launch_at_speed(delta, speed)
            self.assertTrue(torch.isfinite(velocity).all())
            torch.testing.assert_close(velocity.norm(), torch.tensor(speed, dtype=torch.float64))
