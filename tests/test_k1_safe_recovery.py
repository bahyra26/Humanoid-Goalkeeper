"""Check actual K1 reward and observation methods on CPU, without Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
import importlib.util
_outcome_path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/utils/shot_outcome.py'
_spec = importlib.util.spec_from_file_location('shot_outcome', _outcome_path)
_outcome = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outcome)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'legged_gym/legged_gym/envs/base/legged_robot.py'


class SafeRecoveryTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        names = {'_k1_ball_safe', '_k1_ball_settled_or_past', '_k1_both_feet_support', '_reward_success', '_reward_recovery',
                 '_reward_goal_conceded', '_reward_post_hold', '_reward_postorientation',
                 '_reward_smoothness', '_reward_collision',
                 'compute_observations', 'compute_reward',
                 '_reward_head_track', '_reset_root_states', '_reward_termination'}
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {'resolved_save': _outcome.resolved_save, 'torch': torch, 'quat_rotate_inverse': lambda q, v: v}
        cls = ast.ClassDef(name='Environment', bases=[], keywords=[], body=methods, decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), 'exec'), scope)
        self.env = e = scope['Environment']()
        e.num_envs, e.device, e.dt = 2, 'cpu', .02
        e.cfg = NS(asset=NS(name="k1"), amp=NS(amp_coef=0), goal=NS(line_x=-.05, ball_radius=.1),
                   init_state=NS(pos=[0., 0., .53]),
                   rewards=NS(min_knee_height=.08, only_positive_rewards=False))
        e.env_origins = torch.tensor([[0., 0., 0.], [10., 20., 0.]])
        e.ball_states = torch.zeros(2, 13)
        e.ball_states[:, :3] = e.env_origins + torch.tensor([.5, 0., .3])
        e.ball_states[:, 7] = -.5
        e.root_states = torch.zeros(2, 13)
        e.root_states[:, :3] = e.env_origins + torch.tensor([0., 0., .53])
        e.success_flag, e.goal_conceded_flag = torch.ones(2), torch.zeros(2)
        e.stop_flag = torch.ones(2)
        e.reset_buf, e.time_out_buf = torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
        e.recovery_stable_steps = torch.zeros(2, dtype=torch.long)
        e.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
        e.base_lin_vel, e.base_ang_vel = torch.zeros(2, 3), torch.zeros(2, 3)
        e.contact_feet_indices, e.knee_indices = [0, 1], [0, 1]
        e.contact_forces = torch.zeros(2, 2, 3)
        e.contact_forces[:, :, 2] = 10.
        e.rigid_body_states = torch.zeros(2, 2, 13)
        e.rigid_body_states[:, :, 2] = .2

    def make_safe(self):
        self.env.ball_states[:, 0] = self.env.env_origins[:, 0] + 2.1
        self.env.ball_states[:, 7] = .5

    def finish(self):
        self.env.reset_buf[:] = True
        self.env.time_out_buf[:] = True

    def test_amp_does_not_switch_k1_task_rewards(self):
        e = self.env
        self.make_safe()
        self.finish()
        expected = e._reward_success()
        e.cfg.amp.amp_coef = .1
        torch.testing.assert_close(e._reward_success(), expected)
        e.goal_conceded_flag[:] = 1
        self.assertFalse(e._reward_success().any())

    def test_touch_stationary_and_incoming_ball_do_not_pay(self):
        e = self.env
        self.finish()
        for x, vx in [(.5, -.5), (.0, 0.), (-.05, 0.), (2.1, -.1)]:
            e.ball_states[:, 0] = e.env_origins[:, 0] + x
            e.ball_states[:, 7] = vx
            self.assertFalse(e._reward_success().any())
            for _ in range(30):
                self.assertFalse(e._reward_recovery().any())
        self.make_safe()
        e.success_flag.zero_()
        self.assertFalse(e._reward_success().any())
        self.assertFalse(e._reward_recovery().any())

    def test_recovery_requires_safe_stable_tail_and_timeout(self):
        e = self.env
        self.make_safe()
        for _ in range(24):
            self.assertFalse(e._reward_recovery().any())
        self.assertFalse(e._reward_recovery().any())  # No payment during episode.
        self.finish()
        torch.testing.assert_close(e._reward_success() * e.dt, torch.ones(2))
        torch.testing.assert_close(e._reward_recovery() * e.dt, torch.ones(2))
        e.time_out_buf.zero_()  # Falling cannot cash out the safe-ball bonus.
        self.assertFalse(e._reward_success().any())
        self.assertFalse(e._reward_recovery().any())

    def test_instability_or_returning_ball_resets_stable_tail(self):
        e = self.env
        self.make_safe()
        for _ in range(25):
            e._reward_recovery()
        e.base_ang_vel[0, 0] = 2.
        e.ball_states[1, 7] = -.1
        self.finish()
        self.assertFalse(e._reward_recovery().any())
        self.assertEqual(e.recovery_stable_steps.tolist(), [0, 0])
        e.base_ang_vel.zero_()
        self.make_safe()
        for _ in range(24):
            self.assertFalse(e._reward_recovery().any())
        self.assertTrue(e._reward_recovery().all())

    def test_goal_cancels_all_terminal_bonuses(self):
        e = self.env
        self.make_safe()
        for _ in range(30):
            e._reward_recovery()
        e.goal_conceded_flag[:] = 1.
        self.finish()
        self.assertFalse(e._reward_success().any())
        self.assertFalse(e._reward_recovery().any())
        torch.testing.assert_close(e._reward_goal_conceded() * e.dt, torch.ones(2))

    def test_low_inverted_or_airborne_robot_cannot_recover(self):
        e = self.env
        self.make_safe()
        self.finish()
        for fault in ('low', 'inverted', 'airborne', 'low_knee'):
            with self.subTest(fault=fault):
                e.root_states[:, 2] = .53
                e.projected_gravity[:, 2] = -1.
                e.contact_forces[:, :, 2] = 10.
                e.rigid_body_states[:, :, 2] = .2
                e.recovery_stable_steps[:] = 100
                if fault == 'low':
                    e.root_states[:, 2] = .2
                elif fault == 'inverted':
                    e.projected_gravity[:, 2] = 1.
                elif fault == 'airborne':
                    e.contact_forces[:, :, 2] = 0.
                else:
                    e.rigid_body_states[:, :, 2] = .01
                self.assertFalse(e._reward_recovery().any())

    def test_reset_clears_recovery_only_for_selected_environment(self):
        e = self.env
        e.recovery_stable_steps[:] = 100
        e.cfg.domain_rand = NS(init_base_vel_range=[0., 0.])
        e.play = True
        e.base_init_state = torch.zeros(13)
        e.base_init_state[2] = .53
        e.ball_start = e.env_origins + torch.tensor([3.5, 0., .18])
        e.assign_ball_states = lambda ids: torch.tensor([[-4., 0., 4.]]).repeat(len(ids), 1)
        e.ball_vel, e.has_in_air = torch.zeros(2), torch.zeros(2, dtype=torch.bool)
        e.static_obs, e.ball_last = torch.zeros(2, 3), torch.zeros(2, 3)
        e.arm_save_flag, e.substep_max_ball_force = torch.zeros(2), torch.zeros(2)
        e.substep_min_arm_dist, e.substep_min_robot_dist = torch.zeros(2), torch.zeros(2)
        e.dist, e.vanish_step = torch.zeros(2), torch.zeros(2, dtype=torch.long)
        e.prev_ball_pos, e.prev_ball_lin_vel = torch.zeros(2, 3), torch.zeros(2, 3)
        e.sim, e.all_root_states = None, torch.zeros(4, 13)
        e.gym = NS(set_actor_root_state_tensor_indexed=lambda *args: None)
        e._reset_root_states.__globals__['gymtorch'] = NS(unwrap_tensor=lambda tensor: tensor)
        e._reset_root_states(torch.tensor([0]))
        self.assertEqual(e.recovery_stable_steps.tolist(), [0, 100])
        self.assertFalse(e._reward_success().any())
        self.assertFalse(e._reward_goal_conceded().any())

    def test_larger_safe_bonus_still_requires_timeout_and_no_goal(self):
        e = self.env
        self.make_safe()
        self.finish()
        self.assertEqual((e._reward_success() * e.dt * 220.).tolist(), [220., 220.])
        e.time_out_buf.zero_()
        self.assertFalse(e._reward_success().any())
        self.finish()
        e.goal_conceded_flag[:] = 1
        self.assertFalse(e._reward_success().any())

    def test_terminal_reward_scale_is_independent_of_timestep(self):
        e = self.env
        config = ast.parse((ROOT / 'legged_gym/legged_gym/envs/k1/k1_config.py').read_text())
        k1 = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'K1Cfg')
        rewards = next(n for n in k1.body if isinstance(n, ast.ClassDef) and n.name == 'rewards')
        scales = next(n for n in rewards.body if isinstance(n, ast.ClassDef) and n.name == 'scales')
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in scales.body if isinstance(n, ast.Assign)}
        self.assertEqual(values['postupperdofpos'], 0.)
        self.assertGreater(values['post_hold'], 0.)
        self.assertLess(values['collision'], 0.)
        for name in ('successland', 'feet_slippage', 'postorientation', 'postangvel', 'post_hold', 'postlinvel'):
            self.assertGreater(values[name], 0.)
        self.make_safe()
        e.rew_buf = torch.zeros(2)
        e.reward_names = ['goal_conceded', 'recovery', 'success']
        e.reward_functions = [getattr(e, '_reward_' + name) for name in e.reward_names]
        for dt in (.01, .02, .04):
            e.dt = dt
            e.reward_scales = {name: values[name] * dt for name in e.reward_names}
            e.episode_sums = {name: torch.zeros(2) for name in e.reward_names}
            e.goal_conceded_flag.zero_()
            e.recovery_stable_steps[:] = 100
            self.finish()
            e.compute_reward()
            torch.testing.assert_close(e.rew_buf, torch.full((2,), values['success'] + values['recovery']))
            e.goal_conceded_flag[:] = 1
            e.compute_reward()
            torch.testing.assert_close(e.rew_buf, torch.full((2,), values['goal_conceded']))

    def test_early_posture_termination_cannot_avoid_event_penalty(self):
        e = self.env
        e.reset_buf[:] = True
        for dt in (.01, .02, .04):
            e.dt = dt
            torch.testing.assert_close(e._reward_termination() * (-220. * dt), torch.full((2,), -220.))
        e.time_out_buf[:] = True
        self.assertFalse(e._reward_termination().any())

    def test_observation_and_head_tracking_survive_first_contact(self):
        e = self.env
        e.hand_indices, e.torso_index, e.head_joint_indices = [0, 1], 0, [0, 1]
        e.base_quat = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
        e.torso_pos = e.root_states[:, :3].clone()
        e.rigid_body_states[:, 0, :3] = e.torso_pos
        e.catchstep, e.startstep = torch.zeros(2), 49
        e.ball_last = torch.zeros(2, 3)
        e.vanish_step = torch.zeros(2)
        e.dof_pos, e.default_dof_pos, e.dof_vel, e.actions = [torch.zeros(2, 2) for _ in range(4)]
        e.end_regions, e.dist = torch.zeros(2), torch.zeros(2)
        e.end_target = e.ball_states[:, :3].clone()
        e.obs_scales = NS(ang_vel=1., dof_pos=1., dof_vel=1., lin_vel=1., ball_vel=1.)
        e.num_one_step_obs, e.actor_obs_length, e.num_ballobs = 15, 30, 3
        e.obs_buf = torch.zeros(2, 30)
        e.add_noise = False
        e.compute_observations()
        self.assertTrue((e.obs_buf[:, -15:-12].norm(dim=-1) > 0).all())
        self.assertTrue(e._reward_head_track().all())
        self.make_safe()
        e.compute_observations()
        self.assertFalse(e.obs_buf[:, -15:-12].any())
        self.assertFalse(e._reward_head_track().any())

    def test_post_hold_and_collision_rewards(self):
        e = self.env
        self.make_safe()
        e.upper_body_joint_indices = [0]
        e.leg_joint_indices = [1]
        e.dof_vel = torch.tensor([[0., 0.], [4., 4.]])
        e.actions = torch.tensor([[0., 0.], [1., 1.]])
        e.last_actions = torch.zeros(2, 2)
        hold = e._reward_post_hold()
        self.assertGreater(hold[0].item(), hold[1].item())
        e.penalised_contact_indices = [0]
        e.contact_forces[0, 0, 0] = 20.  # Near ball (env 0): ball block allowed
        e.contact_forces[1, 0, 0] = 20.  # Far from ball (env 1): self-collision penalized
        e.rigid_body_states[0, 0, :3] = e.ball_states[0, :3] + torch.tensor([.1, 0., 0.])
        e.rigid_body_states[1, 0, :3] = e.env_origins[1] + torch.tensor([0., 0., .5])
        col = e._reward_collision()
        self.assertEqual(col.tolist(), [0., 1.])

    def test_post_save_balance_prefers_double_support(self):
        e = self.env
        self.make_safe()
        e.upper_body_joint_indices, e.leg_joint_indices = [0], [1]
        e.dof_vel = torch.zeros(2, 2)
        e.contact_forces[:, :, 2] = 10.
        e.contact_forces[1, 1, 2] = 0.  # env 1 lingering on one foot after save
        self.assertGreater(e._reward_post_hold()[0].item(), e._reward_post_hold()[1].item())
        self.assertGreater(e._reward_postorientation()[0].item(), e._reward_postorientation()[1].item())

        # Post-save residual joint tremor gets steeper smoothness penalty than live reflex phase
        e.last_last_actions = torch.zeros(2, 2)
        e.last_actions = torch.zeros(2, 2)
        e.actions = torch.tensor([[.2, .2], [.2, .2]])
        e.ball_states[0, 0] = 0.5  # env 0 live (x<edge, not safe? reset to incoming)
        e.ball_states[0, 7] = -2.0
        sm = e._reward_smoothness()
        self.assertGreater(sm[1].item(), sm[0].item() * 2.0)

    def test_post_goal_balance_and_upright_latch(self):
        e = self.env
        e.upper_body_joint_indices, e.leg_joint_indices = [0], [1]
        e.dof_vel = torch.zeros(2, 2)
        e.dof_pos = torch.zeros(2, 2)
        e.last_last_actions = torch.zeros(2, 2)
        e.last_actions = torch.zeros(2, 2)
        e.actions = torch.tensor([[.2, .2], [.2, .2]])
        e.ball_states[:, 0] = e.env_origins[:, 0] - .5
        e.goal_conceded_flag[0] = 1.  # env 0 conceded, env 1 cleared safely
        self.assertGreater(e._reward_postorientation()[0].item(), 0.8)
        self.assertEqual(e._reward_post_hold()[0].item(), 0.)
        self.assertGreater(e._reward_post_hold()[1].item(), 0.8)
        self.assertAlmostEqual(e._reward_smoothness()[0].item(), e._reward_smoothness()[1].item(), places=5)

        # Hold latch must wait for double support & upright tilt, then release pose lock if leaning > 0.4
        self.make_safe()
        e.goal_conceded_flag.zero_()
        e.k1_safe_steps = torch.tensor([2, 2])
        e.k1_hold_pose_locked = torch.tensor([False, False])
        e.k1_hold_upper_dof_pos = None
        e.projected_gravity = torch.tensor([[0.40, 0., -0.916], [0., 0., -1.]])
        e._reward_post_hold()
        self.assertEqual(e.k1_hold_pose_locked.tolist(), [False, True])
        e.projected_gravity[1] = torch.tensor([0.50, 0., -0.866])  # env 1 starts leaning > 0.40 -> allows arm recovery
        e.dof_pos[1, 0] = 1.5
        e.k1_hold_upper_dof_pos[1, 0] = 0.0
        hold_tilted = e._reward_post_hold()[1].item()
        e.projected_gravity[1] = torch.tensor([0., 0., -1.])
        self.assertLess(hold_tilted, e._reward_post_hold()[1].item() * 0.5)


if __name__ == '__main__':
    unittest.main()
