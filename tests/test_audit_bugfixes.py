"""CPU regression tests for training, checkpoint resolution, runner, and benchmark bug fixes."""
import ast
from collections import deque
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace as NS
from typing import Tuple
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


class AuditBugfixTests(unittest.TestCase):
    def test_get_load_path_rejects_prefix_typo_collision(self):
        scope = extract('legged_gym/legged_gym/utils/helpers.py', {'get_load_path'}, {'os': os, 're': re})
        get_load_path = scope['get_load_path']
        with tempfile.TemporaryDirectory() as directory:
            real_run = Path(directory) / 'k1_v9_s1_gresume'
            real_run.mkdir()
            ckpt = real_run / 'model_5600.pt'
            ckpt.write_bytes(b'weights')
            self.assertEqual(get_load_path(str(real_run), checkpoint=-1), str(ckpt))
            with self.assertRaises(FileNotFoundError):
                get_load_path(str(Path(directory) / 'k1_v9_s1_typo'), checkpoint=-1)

    def test_resume_into_different_existing_exptid_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'logs/k1/k1_src'
            dst = Path(directory) / 'logs/k1/k1_dst'
            src.mkdir(parents=True)
            dst.mkdir(parents=True)
            (src / 'model_5350.pt').write_bytes(b'source')
            (dst / 'model_5400.pt').write_bytes(b'keep dst')
            loaded = []

            class DummyRunner:
                def __init__(self, *args, **kwargs):
                    pass

                def load(self, path):
                    loaded.append(path)

            def make_scope(cfg):
                return extract(
                    'legged_gym/legged_gym/utils/task_registry.py',
                    {'make_alg_runner'},
                    {
                        'os': os, 'Tuple': Tuple, 'VecEnv': object,
                        'HIMOnPolicyRunner': DummyRunner, 'LeggedRobotCfgPPO': object,
                        'LEGGED_GYM_ROOT_DIR': directory,
                        'class_to_dict': lambda x: {},
                        'get_load_path': lambda root, **kw: str(Path(root) / 'model_5350.pt'),
                        'update_cfg_from_args': lambda *a: (None, cfg),
                    },
                )

            cfg = NS(runner=NS(resume=True, experiment_name='k1', load_run=-1, checkpoint=5350))
            scope = make_scope(cfg)
            with self.assertRaises(FileExistsError):
                scope['make_alg_runner'](
                    NS(), NS(),
                    args=NS(exptid='k1_dst', resumeid='k1_src', rl_device='cpu'),
                    train_cfg=cfg,
                )
            self.assertEqual((dst / 'model_5400.pt').read_bytes(), b'keep dst')

            # Resuming in-place (same directory) remains allowed.
            scope['make_alg_runner'](
                NS(), NS(),
                args=NS(exptid='k1_src', resumeid=None, rl_device='cpu'),
                train_cfg=cfg,
            )
            self.assertEqual(loaded, [str(src / 'model_5350.pt')])

    def test_process_env_step_ignores_stale_timeout_when_not_done(self):
        import sys
        sys.path.insert(0, str(ROOT / 'rsl_rl'))
        from rsl_rl.algorithms.him_ppo import HIMPPO
        from rsl_rl.modules.actor_critic import ActorCritic

        ac = ActorCritic(10, 10, 10, 1, num_actions=2)
        ppo = HIMPPO(ac, num_learning_epochs=1, num_mini_batches=1, device='cpu')
        ppo.init_storage(num_envs=2, num_transitions_per_env=2, actor_obs_shape=[10], critic_obs_shape=[10], action_shape=[2], amp_obs_shape=[4])

        ppo.transition.observations = torch.zeros(2, 10)
        ppo.transition.critic_observations = torch.zeros(2, 10)
        ppo.transition.actions = torch.zeros(2, 2)
        ppo.transition.actions_log_prob = torch.zeros(2)
        ppo.transition.action_mean = torch.zeros(2, 2)
        ppo.transition.action_sigma = torch.ones(2, 2)
        ppo.transition.values = torch.tensor([[10.0], [5.0]])

        # Stale time_outs=[True, True] with dones=[False, True]: only env 1 should bootstrap.
        ppo.process_env_step(
            rewards=torch.zeros(2),
            dones=torch.tensor([False, True]),
            infos={'time_outs': torch.tensor([True, True])},
            next_critic_obs=torch.zeros(2, 10),
        )
        torch.testing.assert_close(ppo.storage.rewards[0, :, 0], torch.tensor([0.0, ppo.gamma * 5.0]))
        ppo.test_mode()
        self.assertFalse(ac.training)

    def test_post_physics_clears_episode_and_refreshes_timeouts_in_source(self):
        src = (ROOT / 'legged_gym/legged_gym/envs/base/legged_robot.py').read_text()
        tree = ast.parse(src)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'post_physics_step')
        text = ast.get_source_segment(src, method)
        self.assertIn('self.extras.pop("episode", None)', text)
        self.assertIn('self.extras["time_outs"] = self.time_out_buf', text)

    def test_runner_learn_none_log_dir_and_resume_eta(self):
        scope = extract(
            'rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
            {'learn', 'log', 'train_mode', 'eval_mode'},
            {'torch': torch, 'deque': deque, 'time': time, 'os': os, 'statistics': __import__('statistics'),
             'store_code_state': lambda *args: (_ for _ in ()).throw(AssertionError('should not be called when log_dir is None'))},
        )
        ac = torch.nn.Linear(2, 2)
        ac.std = torch.ones(2)
        env = NS(
            num_envs=2,
            get_observations=lambda: torch.zeros(2, 1),
            get_privileged_observations=lambda: torch.zeros(2, 5),
            step=lambda a: (torch.zeros(2, 1), torch.zeros(2, 5), torch.zeros(2), torch.zeros(2), {}, torch.tensor([], dtype=torch.long), torch.zeros(0, 5)),
        )
        alg = NS(
            actor_critic=NS(train=ac.train, eval=ac.eval, num_one_step_obs=1, std=torch.ones(2)),
            learning_rate=1e-3,
            act=lambda *args: torch.zeros(2, 1),
            process_env_step=lambda *args: None,
            compute_returns=lambda *args: None,
            update=lambda: (0.,) * 7,
        )
        runner = NS(
            log_dir=None, writer=None, env=env, alg=alg, device='cpu', amp_coef=0.0,
            current_learning_iteration=5350, num_steps_per_env=1, save_interval=1,
            log=lambda *args: None, save=lambda *args: (_ for _ in ()).throw(AssertionError('should not save when log_dir is None')),
            git_status_repos=[], logger_type='tensorboard', tot_timesteps=0, tot_time=0.0,
        )
        scope['learn'](runner, 2)
        self.assertEqual(runner.current_learning_iteration, 5352)

        # Verify train_mode/eval_mode don't crash on missing empirical_normalization
        scope['train_mode'](runner)
        self.assertTrue(ac.training)
        scope['eval_mode'](runner)
        self.assertFalse(ac.training)

        # Verify resume ETA is positive, not negative
        printed = []
        scalars = {}
        runner.writer = NS(add_scalar=lambda k, v, s: scalars.__setitem__(k, v))
        log_scope = dict(scope)
        log_scope['print'] = lambda s: printed.append(s)
        runner.tot_time = 0.0
        locs = {
            'ep_infos': [], 'it': 5350, 'start_iter': 5350, 'tot_iter': 5400,
            'num_learning_iterations': 50, 'collection_time': 0.5, 'learn_time': 0.5,
            'mean_value_loss': 0.1, 'mean_surrogate_loss': 0.1, 'mean_est_loss': 0.1,
            'mean_region_loss': 0.1, 'amp_loss': 0.0, 'expert_loss': 0.0, 'policy_loss': 0.0,
            'rewbuffer': [], 'raw_rewbuffer': [], 'amp_rewbuffer': [], 'lenbuffer': [],
        }
        # Recompile log with custom print capture
        tree = ast.parse((ROOT / 'rsl_rl/rsl_rl/runners/him_on_policy_runner.py').read_text())
        log_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'log')
        exec(compile(ast.Module(body=[log_node], type_ignores=[]), 'runner_log', 'exec'), log_scope)
        log_scope['log'](runner, locs)
        eta_match = re.search(r'ETA:\s+(-?[0-9.]+)s', printed[0])
        self.assertIsNotNone(eta_match)
        self.assertAlmostEqual(float(eta_match.group(1)), 49.0, places=1)

    def test_benchmark_summarize_slice_isolates_per_suite_metrics(self):
        scope = extract('legged_gym/legged_gym/scripts/benchmark_k1.py', {'_summarize_slice'}, {'torch': torch})
        summarize = scope['_summarize_slice']
        res = {
            'blocked_save': torch.tensor([True, True, False, True]),
            'goal': torch.tensor([False, False, True, False]),
            'out_no_touch': torch.tensor([False, False, False, False]),
            'off_target_touched': torch.tensor([False, False, False, False]),
            'unresolved': torch.tensor([False, False, False, False]),
            'fall': torch.tensor([False, False, True, False]),
            'end_2ft': torch.tensor([True, True, False, True]),
            'ov2': torch.tensor([False, False, False, False]),
            'daL_sum': torch.tensor([1.0, 1.0, 10.0, 10.0]),
            'daL_cnt': torch.tensor([10.0, 10.0, 10.0, 10.0]),
            'daP_sum': torch.tensor([0.5, 0.5, 5.0, 5.0]),
            'daP_cnt': torch.tensor([10.0, 10.0, 10.0, 10.0]),
            'lim_sum': torch.tensor([0.2, 0.2, 4.0, 4.0]),
            'lim_cnt': torch.tensor([10.0, 10.0, 10.0, 10.0]),
        }
        nk_goals = torch.tensor([True, True, True, True])
        s1 = summarize(res, slice(0, 2), nk_goals)
        s2 = summarize(res, slice(2, 4), nk_goals)
        self.assertAlmostEqual(s1['daL'], 0.1, places=4)
        self.assertAlmostEqual(s2['daL'], 1.0, places=4)
        self.assertAlmostEqual(s1['daP'], 0.05, places=4)
        self.assertAlmostEqual(s2['daP'], 0.5, places=4)
        self.assertAlmostEqual(s1['lim_pct'], 2.0, places=2)
        self.assertAlmostEqual(s2['lim_pct'], 40.0, places=2)

    def test_benchmark_and_evaluator_reset_compute_observations(self):
        for rel_path, fn_name in [
            ('legged_gym/legged_gym/scripts/benchmark_k1.py', '_settle_reset'),
            ('legged_gym/legged_gym/scripts/evaluate_k1.py', 'reset_env_clean'),
        ]:
            src = (ROOT / rel_path).read_text()
            tree = ast.parse(src)
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
            body_src = ast.get_source_segment(src, fn)
            self.assertIn('env.compute_observations()', body_src, f'{rel_path}:{fn_name}')
            self.assertLess(body_src.index('env.compute_observations()'), body_src.index('env.get_observations()'))

    def test_amp_update_returns_averaged_python_floats(self):
        import sys
        sys.path.insert(0, str(ROOT / 'rsl_rl'))
        from rsl_rl.algorithms.him_ppo import HIMPPO
        from rsl_rl.modules.actor_critic import ActorCritic
        from rsl_rl.modules.amp import AMP
        from rsl_rl.utils.utils import Normalizer

        # Minimal one_step=10, critic_obs=24 so -14 is num_one_step_obs+3 (region) and -13:-7 is gtball (6 dims)
        ac = ActorCritic(10, 24, 10, 1, num_actions=2, actor_hidden_dims=[16], critic_hidden_dims=[16])
        amp = AMP(num_obs=4, amp_coef=0.1, hidden_dims=[16], device='cpu')
        norm = Normalizer(4)
        motion_buf = {k: NS(get_expert_obs=lambda batch_size: torch.zeros(batch_size, 4))
                      for k in ('lefthand', 'righthand', 'leftjump', 'rightjump', 'leftstep', 'rightstep')}
        ppo = HIMPPO(ac, num_learning_epochs=2, num_mini_batches=2, device='cpu', amp=amp, amp_normalizer=norm, motion_buffer=motion_buf)
        ppo.init_storage(num_envs=4, num_transitions_per_env=3, actor_obs_shape=[10], critic_obs_shape=[24], action_shape=[2], amp_obs_shape=[4])

        for _ in range(3):
            obs = torch.zeros(4, 10)
            cobs = torch.zeros(4, 24)
            ppo.act(obs, cobs)
            ppo.process_amp_state(torch.zeros(4, 4))
            ppo.process_env_step(torch.zeros(4), torch.zeros(4, dtype=torch.bool), {}, cobs)
        ppo.compute_returns(torch.zeros(4, 24))
        losses = ppo.update()
        for val in losses[4:]:
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)


if __name__ == '__main__':
    unittest.main()
