"""Run a bounded, isolated K1 fine-tuning pilot using existing project APIs."""
import hashlib
import json
import math
from pathlib import Path

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict

args = get_args([{'name': '--arm_amp', 'action': 'store_true',
                  'help': 'Fine-tune with the retargeted eight-joint K1 arm prior.'},
                 {'name': '--success_bonus', 'type': float, 'default': None,
                  'help': 'Override only the terminal safe-ball bonus for a calibration pilot.'},
                 {'name': '--recovery_bonus', 'type': float, 'default': None,
                  'help': 'Override the terminal safe, stable recovery bonus for a calibration pilot.'},
                 {'name': '--episode_length_s', 'type': float, 'default': None,
                  'help': 'Training horizon in seconds for an isolated recovery pilot.'}])
for name in ('success_bonus', 'recovery_bonus', 'episode_length_s'):
    value = getattr(args, name)
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError('--' + name + ' must be finite and positive')
assert args.task == 'k1' and args.exptid != 'k1_v2'
root = Path(LEGGED_GYM_ROOT_DIR)
experiment = 'k1_amp' if args.arm_amp else 'k1_calibration'
destination = root / 'logs' / experiment / args.exptid
destination.mkdir(parents=True, exist_ok=False)
source = root / 'logs/k1/k1_v2/model_3400.pt'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
cfg, train_cfg = task_registry.get_cfgs('k1')
if args.success_bonus is not None:
    cfg.rewards.scales.success = args.success_bonus
if args.recovery_bonus is not None:
    cfg.rewards.scales.recovery = args.recovery_bonus
if args.episode_length_s is not None:
    cfg.env.episode_length_s = args.episode_length_s
if args.arm_amp:
    cfg.amp.amp_coef = 0.1
    train_cfg.amp = cfg.amp
train_cfg.algorithm.learning_rate = 1e-4
train_cfg.algorithm.schedule = 'fixed'
train_cfg.runner.resume = False
train_cfg.runner.experiment_name = experiment
args.resume = False
env, _ = task_registry.make_env('k1', args=args, env_cfg=cfg)
runner, _ = task_registry.make_alg_runner(env=env, name='k1', args=args, train_cfg=train_cfg)
runner.load(str(source), load_optimizer=False)
metadata = dict(source=str(source), source_sha256=digest, seed=args.seed,
                num_envs=args.num_envs, iterations=args.max_iterations,
                learning_rate=1e-4, schedule='fixed', optimizer_reset=True,
                episode_length_s=cfg.env.episode_length_s,
                amp=class_to_dict(cfg.amp),
                reward_scales=class_to_dict(cfg.rewards.scales))
(destination / 'calibration.json').write_text(json.dumps(metadata, indent=2) + '\n')
runner.learn(args.max_iterations, init_at_random_ep_len=False)
assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
print('CALIBRATION COMPLETE:', destination, flush=True)
