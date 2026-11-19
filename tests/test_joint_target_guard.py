import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

class TargetGuardTests(unittest.TestCase):
    def test_enabled_disabled_and_torque_cap(self):
        path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        method = next(n for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == '_compute_torques')
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        env = NS(cfg=NS(asset=NS(name="k1"), amp=NS(amp_coef=0), control=NS(control_type='P', limit_joint_targets=True)),
                 action_scale_vec=torch.ones(2), default_dof_poses=torch.zeros(1, 2),
                 catchstep=torch.zeros(1), startstep=1, init_dof_pos=torch.zeros(1, 2),
                 dof_pos_limits=torch.tensor([[-1., 1.], [-.5, .5]]),
                 p_gains=10., Kp_factors=1., d_gains=1., Kd_factors=1.,
                 dof_pos=torch.zeros(1, 2), dof_vel=torch.zeros(1, 2),
                 actuation_offset=0., joint_injection=0., torque_limits=torch.tensor([2., 2.]))
        action = torch.tensor([[3., -3.]])
        torque = scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, torch.tensor([[1., -.5]]))
        torch.testing.assert_close(torque, torch.tensor([[2., -2.]]))
        env.cfg.control.limit_joint_targets = False
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, action)
        env.cfg.control.limit_joint_targets = True
        env.catchstep[:] = 2
        env.init_dof_pos[:] = .2
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, env.init_dof_pos)

        # K1 must hold at the same boundary where observations still hide the ball.
        env.catchstep[:] = env.startstep
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, env.init_dof_pos)
        env.catchstep[:] = env.startstep - 1
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, torch.tensor([[1., -.5]]))
        # Preserve the G1/AMP boundary behavior.
        env.cfg.asset.name = "g1"
        env.cfg.amp.amp_coef = .4
        env.catchstep[:] = env.startstep
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, torch.tensor([[1., -.5]]))
