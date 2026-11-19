"""Exercise the real reward without importing Isaac Gym."""
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


class K1ReachRewardTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        tree = ast.parse(path.read_text())
        fn_names = ('_reward_eereach', '_k1_ball_safe', '_k1_kolong_state',
                    '_reward_kolong_gap', '_k1_arm_excess_z', '_reward_arm_overhead',
                    '_reward_successland')
        nodes = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in fn_names]
        scope = {'resolved_save': _outcome.resolved_save, 'torch': torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        self.reward = scope['_reward_eereach']
        self.kolong_pen = scope['_reward_kolong_gap']
        self.overhead_pen = scope['_reward_arm_overhead']
        self.successland = scope['_reward_successland']
        self.env = NS(cfg=NS(asset=NS(name="k1"), amp=NS(amp_coef=0), goal=NS(line_x=-.05, ball_radius=.1), init_state=NS(pos=[0., 0., .54])), num_envs=1, device='cpu',
                      ball_states=torch.zeros(1, 13), env_origins=torch.zeros(1, 3),
                      stop_flag=torch.zeros(1), end_target=torch.tensor([[.08, .4, .6]]),
                      torso_pos=torch.tensor([[0., 0., .54]]), end_regions=torch.tensor([0]),
                      rigid_body_states=torch.zeros(1, 3, 13), upper_body_index=2,
                      hand_indices=[0, 1], dist=torch.tensor([.5]),
                      projected_gravity=torch.tensor([[0., 0., -1.]]),
                      success_flag=torch.zeros(1), goal_conceded_flag=torch.zeros(1),
                      _compute_k1_ball_dists=lambda: (torch.tensor([1.]), torch.tensor([1.])))
        self.env._k1_ball_safe = lambda include_behind=False: scope['_k1_ball_safe'](self.env, include_behind=include_behind)
        self.env._k1_kolong_state = lambda: scope['_k1_kolong_state'](self.env)
        self.env._k1_arm_excess_z = lambda: scope['_k1_arm_excess_z'](self.env)
        self.env.ball_states[:, 0] = 2.
        self.env.ball_states[:, 7] = -.5

    def test_hand_speed_alone_does_not_increase_reward(self):
        stationary = self.reward(self.env)
        direction = self.env.end_target / torch.norm(self.env.end_target, dim=1, keepdim=True)
        for speed in (-3., 3.):
            self.env.rigid_body_states[:, 0, 7:10] = direction * speed
            torch.testing.assert_close(self.reward(self.env), stationary)

    def test_closer_selected_hand_still_earns_more(self):
        far = self.reward(self.env)
        self.env.dist[:] = .1
        self.assertGreater(self.reward(self.env).item(), far.item())

    def test_reach_continues_after_contact_until_safe_or_goal(self):
        before_contact = self.reward(self.env)
        self.env.stop_flag[:] = 1
        self.env.success_flag[:] = 1
        torch.testing.assert_close(self.reward(self.env), before_contact)
        self.env.ball_states[:, 0] = 2.1
        self.env.ball_states[:, 7] = .1
        self.assertEqual(self.reward(self.env).item(), 0.)
        self.env.ball_states[:, 0] = 1.
        self.env.goal_conceded_flag[:] = 1
        self.assertEqual(self.reward(self.env).item(), 0.)
        self.env.goal_conceded_flag[:] = 0
        self.env.ball_states[:, 0] = -.1
        self.assertEqual(self.reward(self.env).item(), 0.)

    def test_wide_guard_beats_narrow_tuck_and_allows_dive_tilt(self):
        e = self.env
        e.ball_states[:, 0] = 2.6  # early phase >= 1.8 m
        e.rigid_body_states[:, 0, :3] = torch.tensor([.15, .45, .63])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.15, -.45, .63])
        wide = self.reward(e).item()
        e.rigid_body_states[:, 0, :3] = torch.tensor([.15, .12, .34])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.15, -.12, .34])
        tucked = self.reward(e).item()
        self.assertGreater(wide, tucked * 1.5)

        e.end_regions[:] = 2  # high corner dive with ~35 deg tilt keeps meaningful reach reward
        e.ball_states[:, 0] = 1.0
        e.ball_states[:, 2] = 0.90
        e.end_target[:, 2] = 0.95
        e.rigid_body_states[:, 0, :3] = torch.tensor([.10, .75, .90])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.10, -.10, .75])
        upright_high = self.reward(e).item()
        e.projected_gravity[:] = torch.tensor([[0.0, 0.57, -0.82]])
        tilted_high = self.reward(e).item()
        self.assertGreater(tilted_high, upright_high * 0.55)

        e.projected_gravity[:] = torch.tensor([[0.0, 0.0, -1.0]])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.10, .60, .90])  # clapped together 0.15 m
        clapped_high = self.reward(e).item()
        self.assertGreater(upright_high, clapped_high)

    def test_center_low_closed_legs_beats_straddle_and_arm_head_margin(self):
        e = self.env
        # Bodies: 0=left_hand/foot, 1=right_hand/foot, 2=torso
        e.contact_feet_indices = [0, 1]
        e.head_body_index = None
        e.end_regions[:] = 4
        e.end_target[:] = torch.tensor([[.08, .15, .14]])
        e.ball_end = e.end_target.clone()
        e.ball_states[:, :3] = torch.tensor([[1.0, .10, .15]])
        # Closed legs blocking y=0.15 (gap=0.20 m) vs straddling (gap=0.56 m)
        e.torso_pos[:, 1] = .15
        e.rigid_body_states[:, 0, 1] = .18
        e.rigid_body_states[:, 1, 1] = -.02
        reach_closed = self.reward(e).item()
        pen_closed = self.kolong_pen(e).item()
        e.rigid_body_states[:, 0, 1] = .46
        e.rigid_body_states[:, 1, 1] = -.10
        reach_straddle = self.reward(e).item()
        pen_straddle = self.kolong_pen(e).item()
        self.assertGreater(reach_closed, reach_straddle * 1.5)
        self.assertGreater(pen_straddle, pen_closed * 5.0)

        # Overhead hand with safe clearance is allowed, whereas hand colliding into head/body is penalized
        e.torso_pos[:, 1] = 0.0
        e.end_regions[:] = 2
        e.end_target[:] = torch.tensor([[.08, .50, .95]])
        e.ball_end = e.end_target.clone()
        e.ball_states[:, 0] = 1.5
        e.rigid_body_states[:, 0, :3] = torch.tensor([.10, .45, .90])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.02, -.02, .72])  # colliding into head_pos (0, 0, 0.755)
        ovh_collide = self.overhead_pen(e).item()
        reach_collide = self.reward(e).item()
        e.rigid_body_states[:, 1, :3] = torch.tensor([.10, -.30, .90])  # overhead with safe lateral clearance
        ovh_clear = self.overhead_pen(e).item()
        reach_clear = self.reward(e).item()
        self.assertEqual(ovh_clear, 0.0)
        self.assertGreater(ovh_collide, ovh_clear)
        self.assertGreater(reach_clear, reach_collide)

        # High jump takeoff window (ball_local_x=3.0) rewards lift progression above 0.58 m and airborne 0N flight
        e.contact_forces = torch.zeros(1, 3, 3)
        e.contact_forces[:, :2, 2] = 100.0
        e.has_in_air = torch.zeros(1, dtype=torch.bool)
        e.root_states = torch.tensor([[0., 0., .54, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        stand_land = self.successland(e).item()
        e.root_states[:, 2] = .60
        e.contact_forces[:, :2, 2] = 0.0  # airborne at 0.60 m triggers true jump & landing latch
        air_land = self.successland(e).item()
        self.assertEqual(stand_land, 0.0)
        self.assertTrue(bool(e.has_in_air.item()))
        self.assertGreater(air_land, 0.5)

        # Outer low shot (R4 Bot-L): low sliding foot beats high lifted foot step over the ball
        e.end_regions[:] = 4
        e.end_target[:] = torch.tensor([[.08, .50, .18]])
        e.ball_end = e.end_target.clone()
        e.ball_states[:, :3] = torch.tensor([[1.0, .40, .20]])
        e.rigid_body_states[:, 0, 1] = .42
        e.rigid_body_states[:, 1, 1] = .0
        e.rigid_body_states[:, 0, 2] = .06
        low_foot_rew = self.reward(e).item()
        e.rigid_body_states[:, 0, 2] = .26  # foot lifted above ball
        high_foot_rew = self.reward(e).item()
        self.assertGreater(low_foot_rew, high_foot_rew)

        # Center-low shot (y=0.15): offset pelvis (left foot aligned at ball) beats centered straddle
        e.end_target[:] = torch.tensor([[.08, .15, .14]])
        e.ball_end = e.end_target.clone()
        e.ball_states[:, :3] = torch.tensor([[1.0, .15, .14]])
        e.rigid_body_states[:, 0, :3] = torch.tensor([.10, .15, .05])
        e.rigid_body_states[:, 1, :3] = torch.tensor([.10, -.07, .05])
        e.torso_pos[:, 1] = -.05
        offset_rew = self.reward(e).item()
        e.torso_pos[:, 1] = .15
        centered_rew = self.reward(e).item()
        self.assertGreater(offset_rew, centered_rew)

        # True jump gating: 0.58 m 0N shuffle does not latch jump, whereas 0.60 m 0N flight does
        e.end_regions[:] = 2
        e.has_in_air = torch.zeros(1, dtype=torch.bool)
        e.contact_forces[:, :2, 2] = 0.0
        e.root_states[:, 2] = .58
        _ = self.successland(e)
        self.assertFalse(bool(e.has_in_air.item()))
        e.root_states[:, 2] = .60
        _ = self.successland(e)
        self.assertTrue(bool(e.has_in_air.item()))
