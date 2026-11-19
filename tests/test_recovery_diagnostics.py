import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery', ROOT / 'legged_gym/legged_gym/utils/recovery_diagnostics.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
try:
    import torch
except ImportError:
    torch = None


def frame(step, tilt=False, upright=True):
    return dict(episode_steps=step, low_knee=False, excessive_tilt=tilt,
                excessive_foot_force=False, upper_gravity_z=-1. if upright else 1.,
                base_height_m=.53, contact_flag_at_end=False)


class RecoveryTests(unittest.TestCase):
    def test_recovers_then_relapses(self):
        frames = [frame(1, tilt=True)] + [frame(i) for i in range(2, 8)]
        result = recovery.summarize_recovery(frames, .1, .53)
        self.assertEqual(result['stability_status'], 'recovered_proxy')
        self.assertAlmostEqual(result['first_recovery_time_s'], .6)
        result = recovery.summarize_recovery(frames + [frame(8, tilt=True)], .1, .53)
        self.assertEqual(result['stability_status'], 'not_recovered_by_horizon')
        self.assertIsNotNone(result['first_recovery_time_s'])

    def test_inverted_or_low_body_is_not_recovered(self):
        for inverted in (True, False):
            frames = [frame(1, tilt=True)] + [frame(i, upright=not inverted) for i in range(2, 12)]
            if not inverted:
                for f in frames[1:]:
                    f['base_height_m'] = .1
            self.assertEqual(recovery.summarize_recovery(frames, .1, .53)['stability_status'], 'not_recovered_by_horizon')

    def test_contact_history_and_no_violation(self):
        frames = [frame(i) for i in range(1, 60)]
        frames[0]['contact_flag_at_end'] = True
        result = recovery.summarize_recovery(frames, .1, .53)
        self.assertTrue(result['contact_observed_any_step'])
        self.assertEqual(result['stability_status'], 'no_violation_recorded')
        self.assertEqual(result['frame_at_original_horizon']['episode_steps'], 51)
        with self.assertRaises(ValueError):
            recovery.summarize_recovery([], .1, .53)

    @unittest.skipIf(torch is None, 'PyTorch required')
    def test_training_termination_preserved_and_recovery_bounded(self):
        path = ROOT / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        method = next(n for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == 'check_termination')
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        env = SimpleNamespace(
            cfg=SimpleNamespace(asset=SimpleNamespace(name="k1"), env=SimpleNamespace(observe_recovery=False), amp=SimpleNamespace(amp_coef=0.),
                                rewards=SimpleNamespace(min_knee_height=.1, max_contact_force=100.)),
            rigid_body_states=torch.ones(1, 2, 3), knee_indices=[0, 1],
            projected_gravity=torch.tensor([[.9, 0., -.435]]),
            contact_forces=torch.zeros(1, 2, 3), contact_feet_indices=[0, 1],
            episode_length_buf=torch.tensor([5]), max_episode_length=400, play=True, num_envs=1,
        )
        check = scope['check_termination']
        check(env)
        self.assertTrue(env.reset_buf.item())
        env.cfg.env.observe_recovery = True
        check(env)
        self.assertFalse(env.reset_buf.item())
        self.assertTrue(env.gravity_termination_buf.item())
        env.episode_length_buf[:] = 401
        check(env)
        self.assertTrue(env.reset_buf.item())
        env.play = False
        with self.assertRaises(ValueError):
            check(env)
