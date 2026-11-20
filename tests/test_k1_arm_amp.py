"""CPU checks for partial-arm motion, AMP persistence and robot/task separation."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import warnings
from typing import Tuple
from collections import deque
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


class ArmAMPTests(unittest.TestCase):
    def test_recovery_pilot_overrides_validate_and_preserve_defaults(self):
        tree = ast.parse((ROOT / 'evaluation/k1_recovery_calibration/pilot.py').read_text())
        options = ('success_bonus', 'recovery_bonus', 'episode_length_s')
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'name') or
                 (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and
                  isinstance(n.test.left, ast.Attribute) and n.test.left.attr in options)]
        code = compile(ast.Module(body=nodes, type_ignores=[]), 'pilot_overrides', 'exec')
        config = NS(rewards=NS(scales=NS(success=40., recovery=5.)), env=NS(episode_length_s=5.))
        args = NS(**dict.fromkeys(options))
        scope = {'args': args, 'cfg': config, 'math': math}
        exec(code, scope)
        self.assertEqual((config.rewards.scales.success, config.rewards.scales.recovery,
                          config.env.episode_length_s), (40., 5., 5.))
        args.success_bonus, args.recovery_bonus, args.episode_length_s = 220., 220., 8.
        exec(code, scope)
        self.assertEqual((config.rewards.scales.success, config.rewards.scales.recovery,
                          config.env.episode_length_s), (220., 220., 8.))
        for name in options:
            for value in (0., -1., float('nan'), float('inf')):
                scope['args'] = NS(**dict.fromkeys(options))
                setattr(scope['args'], name, value)
                with self.assertRaisesRegex(ValueError, name):
                    exec(code, scope)

    def test_actual_k1_arm_mapping_clips_and_urdf_limits(self):
        tree = ast.parse((ROOT / 'legged_gym/legged_gym/envs/k1/k1_config.py').read_text())
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'K1Cfg')
        amp = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'amp')
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in amp.body
                  if isinstance(n, ast.Assign) and n.targets[0].id != 'num_obs'}
        names = values['joint_names']
        self.assertEqual(values['amp_coef'], 0.)
        self.assertEqual(len(names), 8)
        self.assertEqual(len(set(names)), 8)
        folder = ROOT / 'legged_gym/resources/datasets/k1_arms'
        mapping = {name: int(index) for index, name in
                   (line.split() for line in (folder / 'joint_id.txt').read_text().splitlines())}
        self.assertEqual([mapping[name] for name in names], list(range(8)))
        joints = {j.get('name'): j for j in ET.parse(
            ROOT / 'legged_gym/resources/robots/k1/urdf/k1.urdf').getroot().findall('joint')}
        bounds = torch.tensor([[float(joints[name].find('limit').get(key))
                                for key in ('lower', 'upper', 'velocity')] for name in names])
        self.assertEqual({p.stem for p in folder.glob('*.pt')},
                         {'lefthand', 'righthand', 'leftjump', 'rightjump', 'leftstep', 'rightstep'})
        for path in folder.glob('*.pt'):
            clip = torch.load(path, map_location='cpu')
            position, velocity = clip['joint_position'], clip['joint_velocity']
            self.assertEqual(position.shape[1], 8)
            self.assertEqual(position.shape, velocity.shape)
            self.assertTrue(torch.isfinite(position).all() and torch.isfinite(velocity).all())
            self.assertTrue(((position >= bounds[:, 0]) & (position <= bounds[:, 1])).all(), path.name)
            self.assertTrue((velocity.abs() <= bounds[:, 2] + 1e-5).all(), path.name)
            torch.testing.assert_close(velocity[:-1], torch.diff(position, dim=0) * 30)

    def test_pyroki_clips_load_as_twenty_joint_amp_transitions(self):
        folder = ROOT / 'legged_gym/resources/datasets/g1_to_k1_retargeted'
        mapping = {name: int(i) for i, name in
                   (line.split() for line in (folder / 'joint_id.txt').read_text().splitlines())}
        joints = ET.parse(ROOT / 'legged_gym/resources/robots/k1/urdf/k1.urdf').findall('joint')
        names = [j.get('name') for j in joints if j.get('type') == 'revolute' and 'head' not in j.get('name')]
        self.assertEqual(len(names), 20)
        self.assertEqual(set(names), set(mapping))
        scope = extract('legged_gym/legged_gym/envs/g1/g1_utils.py', {'MotionLib'},
                        {'torch': torch, 'math': math, 'tqdm': lambda x: x})
        for skill in ('lefthand', 'righthand', 'leftjump', 'rightjump', 'leftstep', 'rightstep'):
            clip = torch.load(folder / (skill + '.pt'), map_location='cpu')
            lib = scope['MotionLib']([clip], mapping, names, [], amp_obs_type='dof')
            sample = lib.get_expert_obs(32)
            self.assertEqual(sample.shape, (32, 40))
            self.assertTrue(torch.isfinite(sample).all())
            torch.testing.assert_close(lib.motion_dof_pos, clip['joint_position'][:, [mapping[n] for n in names]])

    @unittest.skipUnless(importlib.util.find_spec("pybullet"), "requires offline retarget dependency PyBullet")
    def test_pyroki_urdf_clearance_at_240_hz(self):
        path = ROOT / 'legged_gym/legged_gym/scripts/k1_self_collision.py'
        spec = importlib.util.spec_from_file_location('k1_self_collision', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resources = ROOT / 'legged_gym/resources'
        folder = resources / 'datasets/g1_to_k1_retargeted'
        names = [line.split()[1] for line in (folder / 'joint_id.txt').read_text().splitlines()]
        check = module.K1CollisionCheck(resources / 'robots/k1/urdf/k1.urdf', names)
        self.addCleanup(check.close)
        for path in sorted(folder.glob('*.pt')):
            poses = torch.load(path, map_location='cpu')['joint_position'].numpy()
            for frame, (a, b) in enumerate(zip(poses, poses[1:])):
                for step in range(8):
                    pose = a + (b - a) * (step / 8)
                    self.assertFalse(check.collisions(pose), (path.name, frame, step))
            self.assertFalse(check.collisions(poses[-1]), path.name)

    def test_new_training_refuses_existing_checkpoints(self):
        config = NS(runner=NS(resume=False, experiment_name='k1'))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'logs/k1/existing'
            run.mkdir(parents=True)
            checkpoint = run / 'model_3400.pt'
            checkpoint.write_bytes(b'keep this model')
            scope = extract('legged_gym/legged_gym/utils/task_registry.py', {'make_alg_runner'},
                            {'os': os, 'Tuple': Tuple, 'VecEnv': object,
                             'HIMOnPolicyRunner': object, 'LeggedRobotCfgPPO': object,
                             'LEGGED_GYM_ROOT_DIR': directory,
                             'update_cfg_from_args': lambda *args: (None, config)})
            with self.assertRaises(FileExistsError):
                scope['make_alg_runner'](NS(), NS(), args=NS(exptid='existing'), train_cfg=config)
            self.assertEqual(checkpoint.read_bytes(), b'keep this model')

    def test_fixed_shots_keep_correct_k1_region_with_amp(self):
        scope = extract('legged_gym/legged_gym/envs/base/legged_robot.py', {'assign_ball_states'}, {'torch': torch})
        env = NS(cfg=NS(asset=NS(name='k1'), amp=NS(amp_coef=.1)), num_envs=1,
                 ball_start=torch.zeros(1, 3), ball_end=torch.zeros(1, 3),
                 end_target=torch.zeros(1, 3), env_origins=torch.zeros(1, 3),
                 end_regions=torch.zeros(1, dtype=torch.long), catchstep=torch.zeros(1))
        for y, z, region in [(.6, .2, 4), (-.6, .2, 5), (.6, .65, 0),
                              (-.6, .65, 1), (.6, .95, 2), (-.6, .95, 3)]:
            env.evaluation_shot = dict(start_x_m=3.5, start_y_m=0., start_z_m=.18,
                                       target_x_m=.08, target_y_m=y, target_z_m=z,
                                       vx_m_s=-4., vy_m_s=.5, vz_m_s=4.)
            velocity = scope['assign_ball_states'](env, torch.tensor([0]))
            self.assertEqual(env.end_regions.item(), region)
            torch.testing.assert_close(velocity, torch.tensor([[-4., .5, 4.]]))

    def test_terminal_amp_uses_pre_reset_pose_and_region(self):
        scope = extract('rsl_rl/rsl_rl/runners/him_on_policy_runner.py', {'learn'},
                        {'torch': torch, 'deque': deque, 'time': time, 'os': os,
                         'store_code_state': lambda *args: []})
        old = torch.stack((torch.zeros(8), torch.ones(8)))
        reset = torch.stack((torch.full((8,), 999.), torch.full((8,), 3.)))
        before = torch.zeros(2, 5)
        before[1, 4] = 1 / 3
        after = before.clone()
        after[0, 4] = 1 / 3
        calls, stored = {}, []

        def step(actions):
            env.pose = reset
            return (torch.zeros(2, 1), after, torch.zeros(2), torch.tensor([True, False]),
                    {'terminal_amp_observations': torch.full((1, 8), 2.)},
                    torch.tensor([0]), before[:1])

        def discriminator(key):
            def predict(samples, **kwargs):
                calls[key] = samples.clone()
                return torch.zeros(len(samples), 1)
            return NS(predict_reward=predict)

        env = NS(num_envs=2, pose=old, get_observations=lambda: torch.zeros(2, 1),
                 get_privileged_observations=lambda: before, step=step)
        env.get_amp_observations = lambda: env.pose.clone()
        alg = NS(actor_critic=NS(train=lambda: None, num_one_step_obs=1),
                 act=lambda *args: torch.zeros(2, 1),
                 amp={key: discriminator(key) for key in ('lefthand', 'righthand', 'leftjump', 'rightjump', 'leftstep', 'rightstep')},
                 amp_normalizer=None, process_amp_state=lambda samples: stored.append(samples.clone()),
                 process_env_step=lambda *args: None, compute_returns=lambda *args: None,
                 update=lambda: (0.,) * 7)
        with tempfile.TemporaryDirectory() as directory:
            runner = NS(log_dir=directory, writer=True, env=env, alg=alg, device='cpu', amp_coef=.1,
                        current_learning_iteration=0, num_steps_per_env=1, save_interval=50,
                        log=lambda *args: None, save=lambda *args: None,
                        git_status_repos=[], logger_type='tensorboard')
            scope['learn'](runner, 1)
        torch.testing.assert_close(stored[0][0], torch.cat((old[0], torch.full((8,), 2.))))
        torch.testing.assert_close(stored[0][1], torch.cat((old[1], reset[1])))
        self.assertEqual(len(calls['lefthand']), 1)
        self.assertEqual(len(calls['righthand']), 1)
        self.assertEqual(runner.current_learning_iteration, 1)

    def test_selected_joints_only(self):
        scope = extract('legged_gym/legged_gym/envs/base/legged_robot.py', {'get_amp_observations'}, {})
        env = NS(dof_pos=torch.arange(44).reshape(2, 22), amp_joint_indices=[2, 3, 4, 5, 6, 7, 8, 9])
        actual = scope['get_amp_observations'](env)
        torch.testing.assert_close(actual, env.dof_pos[:, 2:10])
        actual.zero_()
        self.assertEqual(env.dof_pos[0, 2], 2)

    def test_motion_library_accepts_arm_only_clips(self):
        scope = extract('legged_gym/legged_gym/envs/g1/g1_utils.py', {'MotionLib'},
                        {'torch': torch, 'math': math, 'tqdm': lambda x: x})
        clip = {'joint_position': torch.arange(40).reshape(5, 8).float(),
                'joint_velocity': torch.zeros(5, 8)}
        names = [f'joint{i}' for i in range(8)]
        lib = scope['MotionLib']([clip], dict(zip(names, range(8))), names, [], amp_obs_type='dof')
        obs = lib.get_expert_obs(32)
        self.assertEqual(obs.shape, (32, 16))
        self.assertTrue(torch.isfinite(obs).all())
        self.assertTrue((obs[:, 8:] >= obs[:, :8]).all())

    def test_checkpoint_round_trip_and_legacy_warm_start(self):
        scope = extract('rsl_rl/rsl_rl/runners/him_on_policy_runner.py', {'save', 'load'},
                        {'torch': torch, 'os': os, 'tempfile': tempfile, 'warnings': warnings})
        model = torch.nn.Linear(16, 1)
        actor = torch.nn.Linear(3, 2)
        normalizer = NS(mean=np.full(16, 2.), var=np.full(16, 4.), count=123.)
        runner = NS(alg=NS(actor_critic=actor, amp={'lefthand': model}, amp_normalizer=normalizer,
                           optimizer=torch.optim.Adam(list(actor.parameters()) + list(model.parameters()))),
                    env=NS(cfg=NS(amp=NS(joint_names=['arm'])), dof_names=['arm']),
                    current_learning_iteration=12, device='cpu')
        sample = torch.randn(2, 16)
        expected = model(sample).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model_12.pt')
            scope['save'](runner, path)
            with torch.no_grad():
                model.weight.zero_()
            normalizer.mean = np.zeros(16)
            scope['load'](runner, path)
            torch.testing.assert_close(model(sample), expected)
            np.testing.assert_equal(normalizer.mean, np.full(16, 2.))
            self.assertEqual(runner.current_learning_iteration, 12)
            runner.env.cfg.amp.joint_names = ['other_arm']
            with self.assertRaises(ValueError):
                scope['load'](runner, path)
            runner.env.cfg.amp.joint_names = ['arm']
            state = torch.load(path)
            del state['amp_state_dict']
            torch.save(state, path)
            with self.assertRaises(ValueError):
                scope['load'](runner, path)
            scope['load'](runner, path, load_optimizer=False)

    def test_normalizer_updates_raw_samples(self):
        utils = extract('rsl_rl/rsl_rl/utils/utils.py', {'RunningMeanStd', 'Normalizer'},
                        {'np': np, 'torch': torch, 'Tuple': Tuple})
        normalizer = utils['Normalizer'](16)
        normalizer.mean[:] = 10.
        normalizer.count = 0.
        samples = torch.full((12, 16), 2.)
        expert = torch.full_like(samples, 4.)
        method = next(n for n in ast.walk(ast.parse((ROOT / 'rsl_rl/rsl_rl/algorithms/him_ppo.py').read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == 'update')
        updates = [node for node in ast.walk(method) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == 'update']
        scope = {'self': NS(amp_normalizer=normalizer), 'raw_amp_obs_batch': samples,
                 'amp_expert_obs_batch_mask': expert,
                 'amp_obs_batch': normalizer.normalize_torch(samples, 'cpu'),
                 'amp_expert_obs_batch': normalizer.normalize_torch(expert, 'cpu')}
        exec(compile(ast.Module(body=updates, type_ignores=[]), 'normalizer_update', 'exec'), scope)
        np.testing.assert_allclose(normalizer.mean, np.full(16, 3.))

    def test_urdf_retarget_reduces_error_within_joint_limits(self):
        path = ROOT / 'legged_gym/legged_gym/scripts/retarget_k1_arms.py'
        spec = importlib.util.spec_from_file_location('retarget', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        chain = module.ArmChain(ROOT / 'legged_gym/resources/robots/k1/urdf/k1.urdf', 'left', 'k1')
        reference = torch.tensor([[-.4, -.5, .1, -.8], [-.42, -.52, .1, -.82], [-.44, -.54, .1, -.84]])
        initial = [-.25, -.35, 0., -.65]
        before = torch.linalg.vector_norm(chain.points(torch.tensor(initial).repeat(3, 1)) - chain.points(reference), dim=-1).mean()
        result, errors = module.retarget(chain, chain, reference, initial, iterations=180)
        self.assertLess(errors.mean(), before)
        self.assertLess(errors.max(), .02)
        bounds = torch.tensor(chain.limits)
        self.assertTrue(((result >= bounds[:, 0]) & (result <= bounds[:, 1])).all())


if __name__ == '__main__':
    unittest.main()
