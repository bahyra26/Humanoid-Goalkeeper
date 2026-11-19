"""Exercise the environment's actual diagnostic snapshot without Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required for tensor snapshot test')
class DiagnosticTests(unittest.TestCase):
    def test_terminal_snapshot_survives_reset(self):
        path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        tree = ast.parse(path.read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == '_evaluation_diagnostics')
        env = SimpleNamespace(
            evaluation_shot={}, extras={'shot_results': {}},
            ball_states=torch.zeros(2, 13), env_origins=torch.tensor([[0., 0., 0.], [10., 20., 1.]]),
            success_flag=torch.tensor([0., 1.]), goal_conceded_flag=torch.zeros(2),
            episode_length_buf=torch.tensor([0, 123]), projected_gravity=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
            root_states=torch.zeros(2, 13),
            time_out_buf=torch.tensor([False, True]),
            gravity_termination_buf=torch.tensor([False, True]),
            rigid_body_states=torch.ones(2, 2, 3), knee_indices=[0, 1],
            contact_forces=torch.zeros(2, 2, 3), contact_feet_indices=[0, 1],
            cfg=SimpleNamespace(rewards=SimpleNamespace(min_knee_height=.1, max_contact_force=100.)),
        )
        env.ball_states[1, :3] = torch.tensor([10.5, 19., 1.2])
        env.ball_states[1, 7:10] = torch.tensor([-2., 1., .5])
        env.rigid_body_states[1, 0, 2] = .05
        env.contact_forces[1, :, 2] = 200.
        scope = {'self': env, 'torch': torch, 'env_ids': torch.tensor([1]), 'contact': torch.tensor([True])}
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec'), scope)
        env.extras['shot_results'] = scope['_evaluation_diagnostics'](env, scope['env_ids'])
        env.success_flag.zero_()
        env.ball_states.zero_()
        env.time_out_buf.zero_()
        env.gravity_termination_buf.zero_()
        scope['contact'].zero_()
        record = env.extras['shot_results']
        self.assertAlmostEqual(record['final_ball_x_m'].item(), .5)
        self.assertAlmostEqual(record['final_ball_y_m'].item(), -1.)
        self.assertAlmostEqual(record['final_ball_z_m'].item(), .2, places=5)
        self.assertEqual(record['final_ball_vx_m_s'].item(), -2.)
        for key in ('timeout', 'low_knee', 'excessive_tilt', 'excessive_foot_force', 'contact_flag_at_end'):
            self.assertTrue(record[key].item(), key)


if __name__ == '__main__':
    unittest.main()
