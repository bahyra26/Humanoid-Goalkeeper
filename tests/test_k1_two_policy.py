"""CPU unit tests for the 2-policy Goalkeeper + Walk-Back state machine and WalkActorCritic."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rsl_rl'))

_tp_path = ROOT / 'legged_gym/legged_gym/utils/k1_two_policy.py'
_tp_spec = importlib.util.spec_from_file_location('k1_two_policy', _tp_path)
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)

K1TwoPolicyController = _tp.K1TwoPolicyController
STATE_GOALKEEPER = _tp.STATE_GOALKEEPER
STATE_READY_HOME = _tp.STATE_READY_HOME
STATE_WALK_BACK = _tp.STATE_WALK_BACK
compute_home_velocity_command = _tp.compute_home_velocity_command
quat_to_yaw = _tp.quat_to_yaw
from rsl_rl.modules.actor_critic import WalkActorCritic
from rsl_rl.algorithms.him_ppo import HIMPPO


class TwoPolicyTests(unittest.TestCase):
    def _make_env(self, num_envs=2):
        e = NS()
        e.cfg = NS(goal=NS(line_x=-0.05, ball_radius=0.075, half_width=1.30))
        e.env_origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])[:num_envs]
        e.root_states = torch.zeros(num_envs, 13)
        e.root_states[:, :3] = e.env_origins + torch.tensor([0.0, 0.8, 0.52])
        e.root_states[:, 6] = 1.0
        e.base_quat = e.root_states[:, 3:7].clone()
        e.base_lin_vel = torch.zeros(num_envs, 3)
        e.base_ang_vel = torch.zeros(num_envs, 3)
        e.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1)
        e.contact_feet_indices = [0, 1]
        e.contact_forces = torch.zeros(num_envs, 2, 3)
        e.contact_forces[:, :, 2] = 20.0
        e.dof_pos = torch.zeros(num_envs, 22)
        e.default_dof_pos = torch.zeros(1, 22)
        e.dof_vel = torch.zeros(num_envs, 22)
        e.ball_states = torch.zeros(num_envs, 13)
        e.ball_states[:, :3] = e.env_origins + torch.tensor([2.5, 0.0, 0.2])
        e.ball_states[:, 7] = 1.0
        e.goal_conceded_flag = torch.zeros(num_envs)
        e._play_done = torch.zeros(num_envs, dtype=torch.bool)
        e._k1_play_fully_done = lambda: e._play_done
        return e

    def test_home_velocity_command_body_frame_and_tilt_scaling(self):
        # Displaced to +y=1.0, facing forward (yaw=0): must move right (vy < 0).
        cmd = compute_home_velocity_command(
            torch.tensor([[0.0, 1.0]]), torch.tensor([0.0]), torch.tensor([-0.99])
        )
        self.assertAlmostEqual(cmd[0, 0].item(), 0.0, places=5)
        self.assertLess(cmd[0, 1].item(), -0.3)
        self.assertAlmostEqual(cmd[0, 2].item(), 0.0, places=5)

        # Facing right (yaw=-pi/2) at +y=1.0: home is directly in front (vx > 0), turn left (vyaw > 0).
        q_right = torch.tensor([[0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4)]])
        yaw = quat_to_yaw(q_right)
        self.assertAlmostEqual(yaw.item(), -math.pi / 2, places=5)
        cmd_rot = compute_home_velocity_command(torch.tensor([[0.0, 1.0]]), yaw, torch.tensor([-0.99]))
        self.assertGreater(cmd_rot[0, 0].item(), 0.5)
        self.assertGreater(cmd_rot[0, 2].item(), 0.5)

        # Heavily tilted robot pauses horizontal command to recover upright first.
        cmd_tilt = compute_home_velocity_command(torch.tensor([[0.0, 1.0]]), torch.tensor([0.0]), torch.tensor([-0.84]))
        self.assertEqual(cmd_tilt[0].tolist(), [0.0, 0.0, 0.0])

        # Inside deadzone at home: zero command.
        cmd_home = compute_home_velocity_command(torch.tensor([[0.03, -0.04]]), torch.tensor([0.05]), torch.tensor([-0.99]))
        self.assertEqual(cmd_home[0].tolist(), [0.0, 0.0, 0.0])

    def test_state_machine_handover_blend_arrival_and_emergency_preemption(self):
        env = self._make_env(2)
        ctrl = K1TwoPolicyController(num_envs=2, handover_hold_steps=3, arrive_hold_steps=3, blend_steps=4)
        gk_obs = torch.zeros(2, 750)
        gk_pol = lambda o: torch.zeros(o.shape[0], 22)
        walk_pol = lambda o: torch.ones(o.shape[0], 22)

        # Ball not yet fully done -> stays in STATE_GOALKEEPER.
        for _ in range(5):
            act = ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
            self.assertEqual(ctrl.state.tolist(), [STATE_GOALKEEPER, STATE_GOALKEEPER])
            torch.testing.assert_close(act, torch.zeros(2, 22))

        # Play fully done on env 0 only -> transitions after handover_hold_steps with smooth blend.
        env._play_done[0] = True
        for _ in range(2):
            ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
            self.assertEqual(ctrl.state[0].item(), STATE_GOALKEEPER)
        act_b1 = ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
        self.assertEqual(ctrl.state.tolist(), [STATE_WALK_BACK, STATE_GOALKEEPER])
        # First blend step is smooth (alpha = 1/4 = 0.25), not a 0->1 1-frame jump.
        self.assertAlmostEqual(act_b1[0, 0].item(), 0.25, places=4)
        self.assertAlmostEqual(act_b1[1, 0].item(), 0.0, places=4)
        for expected_alpha in (0.5, 0.75, 1.0):
            act_b = ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
            self.assertAlmostEqual(act_b[0, 0].item(), expected_alpha, places=4)

        # Hysteresis: minor walking tilt (-0.86) must NOT kick WALK_BACK back to GOALKEEPER.
        env.projected_gravity[0, 2] = -0.86
        ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
        self.assertEqual(ctrl.state[0].item(), STATE_WALK_BACK)

        # Arriving upright at (0, 0, yaw=0) transitions to STATE_READY_HOME with cmd=0.
        env.projected_gravity[0, 2] = -0.99
        env.root_states[0, :2] = env.env_origins[0, :2] + torch.tensor([0.02, -0.03])
        for _ in range(3):
            ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
        self.assertEqual(ctrl.state[0].item(), STATE_READY_HOME)
        self.assertEqual(ctrl.walk_commands[0].tolist(), [0.0, 0.0, 0.0])

        # Emergency preemption: incoming ball immediately switches back to STATE_GOALKEEPER with 0 blend lag.
        env.ball_states[0, :3] = env.env_origins[0] + torch.tensor([3.0, 0.1, 0.4])
        env.ball_states[0, 7] = -5.0
        act_pre = ctrl.step_action(env, gk_obs, gk_pol, walk_pol)
        self.assertEqual(ctrl.state[0].item(), STATE_GOALKEEPER)
        self.assertAlmostEqual(act_pre[0, 0].item(), 0.0, places=5)

    def test_walk_actor_critic_trains_cleanly_in_him_ppo(self):
        ac = WalkActorCritic(
            num_actor_obs=5 * 77,
            num_critic_obs=85,
            num_one_step_obs=77,
            actor_history_length=5,
            num_actions=22,
            actor_hidden_dims=[64, 32],
            critic_hidden_dims=[64, 32],
        )
        ppo = HIMPPO(ac, num_learning_epochs=1, num_mini_batches=2, device='cpu')
        ppo.init_storage(4, 3, [5 * 77], [85], [22], [1])
        for _ in range(3):
            obs = torch.randn(4, 5 * 77)
            cobs = torch.randn(4, 85)
            act = ppo.act(obs, cobs)
            self.assertEqual(tuple(act.shape), (4, 22))
            ppo.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {}, cobs)
        ppo.compute_returns(torch.randn(4, 85))
        v_loss, s_loss, est_loss, reg_loss, *_ = ppo.update()
        self.assertTrue(math.isfinite(v_loss) and math.isfinite(s_loss))
        self.assertEqual(est_loss, 0.0)
        self.assertEqual(reg_loss, 0.0)


if __name__ == '__main__':
    unittest.main()
