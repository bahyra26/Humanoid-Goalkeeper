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
        # When hard_dof_pos_limits is present, clamp targets to hard limits (preserving soft-margin headroom).
        env.hard_dof_pos_limits = torch.tensor([[-1.2, 1.2], [-.6, .6]])
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, torch.tensor([[1.2, -.6]]))
        # Upper-only: joint 0 (upper body) clamps to hard limit, joint 1 (leg) stays free.
        env.cfg.asset.name = "k1"
        env.cfg.amp.amp_coef = 0
        env.catchstep[:] = env.startstep - 1
        env.cfg.control.limit_targets_upper_only = True
        env.upper_body_joint_indices = torch.tensor([0])
        scope['_compute_torques'](env, action)
        torch.testing.assert_close(env.joint_pos_target, torch.tensor([[1.2, -3.]]))

        # K1WalkEnv._compute_torques applies the identical upper-only hard-limit clamp.
        walk_path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/k1/k1_walk.py'
        walk_method = next(n for n in ast.walk(ast.parse(walk_path.read_text()))
                           if isinstance(n, ast.FunctionDef) and n.name == '_compute_torques')
        walk_scope = {'torch': torch}
        exec(compile(ast.Module(body=[walk_method], type_ignores=[]), str(walk_path), 'exec'), walk_scope)
        wenv = NS(cfg=NS(control=NS(limit_joint_targets=True, limit_targets_upper_only=True)),
                  action_scale_vec=torch.ones(2), default_dof_pos=torch.zeros(1, 2),
                  hard_dof_pos_limits=torch.tensor([[-1.2, 1.2], [-.6, .6]]),
                  upper_body_joint_indices=torch.tensor([0]),
                  p_gains=10., Kp_factors=1., d_gains=1., Kd_factors=1.,
                  dof_pos=torch.zeros(1, 2), dof_vel=torch.zeros(1, 2),
                  actuation_offset=0., joint_injection=0., torque_limits=torch.tensor([50., 50.]))
        wtorque = walk_scope['_compute_torques'](wenv, action)
        torch.testing.assert_close(wtorque, torch.tensor([[12., -30.]]))

        # apply_k1_legacy_cfg restores +-0.10 stance and disables target clamp without mutating registry copies.
        helpers_path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/utils/helpers.py'
        legacy_fn = next(n for n in ast.walk(ast.parse(helpers_path.read_text()))
                         if isinstance(n, ast.FunctionDef) and n.name == 'apply_k1_legacy_cfg')
        hscope = {}
        exec(compile(ast.Module(body=[legacy_fn], type_ignores=[]), str(helpers_path), 'exec'), hscope)
        reg_path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/utils/task_registry.py'
        get_cfgs_fn = next(n for n in ast.walk(ast.parse(reg_path.read_text()))
                           if isinstance(n, ast.FunctionDef) and n.name == 'get_cfgs')
        from copy import deepcopy
        from typing import Tuple
        rscope = {'deepcopy': deepcopy, 'Tuple': Tuple, 'LeggedRobotCfg': object, 'LeggedRobotCfgPPO': object}
        exec(compile(ast.Module(body=[get_cfgs_fn], type_ignores=[]), str(reg_path), 'exec'), rscope)
        base_cfg = NS(seed=1, control=NS(limit_joint_targets=True, limit_targets_upper_only=True),
                      init_state=NS(default_joint_angles={'left_hip_roll_joint': 0.04, 'left_ankle_roll_joint': -0.04,
                                                          'right_hip_roll_joint': -0.04, 'right_ankle_roll_joint': 0.04},
                                    init_pos=[0.0] * 22))
        reg = NS(env_cfgs={'k1': base_cfg}, train_cfgs={'k1': NS(seed=1)})
        c1, _ = rscope['get_cfgs'](reg, 'k1')
        hscope['apply_k1_legacy_cfg'](c1)
        self.assertFalse(c1.control.limit_joint_targets)
        self.assertEqual(c1.init_state.default_joint_angles['left_hip_roll_joint'], 0.10)
        c2, _ = rscope['get_cfgs'](reg, 'k1')
        self.assertTrue(c2.control.limit_joint_targets)
        self.assertEqual(c2.init_state.default_joint_angles['left_hip_roll_joint'], 0.04)
