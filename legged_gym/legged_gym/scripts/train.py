# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym import gymapi
import torch

def train(args):
    if args.arm_amp and args.pyroki_amp:
        raise ValueError('Choose one AMP dataset')
    if args.arm_amp or args.pyroki_amp:
        if args.task != 'k1':
            raise ValueError('K1 AMP requires --task k1')
        env_cfg, train_cfg = task_registry.get_cfgs('k1')
        if args.pyroki_amp:
            env_cfg.dataset.folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/g1_to_k1_retargeted"
            env_cfg.dataset.joint_mapping = env_cfg.dataset.folder + "/joint_id.txt"
            ctrl = env_cfg.control
            env_cfg.amp.joint_names = (ctrl.left_arm_joints + ctrl.right_arm_joints
                                       + ctrl.left_leg_joints + ctrl.right_leg_joints)
            env_cfg.amp.num_obs = len(env_cfg.amp.joint_names) * env_cfg.amp.num_steps
        env_cfg.amp.amp_coef = 0.1
        train_cfg.amp = env_cfg.amp
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    if env.viewer is not None:
        if args.task == 'k1':
            camera_pos = gymapi.Vec3(3.0, 2.0, 1.2)
            camera_target = gymapi.Vec3(0.0, 0.0, 0.5)
        else:
            camera_pos = gymapi.Vec3(4.0, 3.0, 1.8)
            camera_target = gymapi.Vec3(0.0, 0.0, 0.8)
        env.gym.viewer_camera_look_at(env.viewer, None, camera_pos, camera_target)

    save_path = os.path.join(LEGGED_GYM_ROOT_DIR)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=save_path)

    if train_cfg.runner.resume and getattr(train_cfg.algorithm, 'schedule', None) == 'fixed':
        target_lr = float(train_cfg.algorithm.learning_rate)
        ppo_runner.alg.learning_rate = target_lr
        for g in ppo_runner.alg.optimizer.param_groups:
            g['lr'] = target_lr

    if (
        args.task == 'k1'
        and train_cfg.runner.resume
        and getattr(ppo_runner, 'current_learning_iteration', 0) >= 1000
        and hasattr(env, 'curriculumupdate')
        and hasattr(env, 'command_init_ranges')
    ):
        env.common_step_counter = max(int(env.common_step_counter), 3000)
        env.curriculumupdate = max(int(env.curriculumupdate), 1)
        expand = 0.18 * env.curriculumupdate
        env.command_ranges[:, 0] = torch.clip(env.command_init_ranges[:, 0] - expand, env.command_bound[:, 0], env.command_bound[:, 1])
        env.command_ranges[:, 1] = torch.clip(env.command_init_ranges[:, 1] + expand, env.command_bound[:, 0], env.command_bound[:, 1])
        env.command_ranges[:, 2] = torch.clip(env.command_init_ranges[:, 2] - expand, env.command_bound[:, 2], env.command_bound[:, 3])
        env.command_ranges[:, 3] = torch.clip(env.command_init_ranges[:, 3] + expand, env.command_bound[:, 2], env.command_bound[:, 3])

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args([{'name': '--arm_amp', 'action': 'store_true',
                      'help': 'Enable the retargeted eight-joint K1 arm AMP prior.'},
                     {'name': '--pyroki_amp', 'action': 'store_true',
                      'help': 'Enable the PyRoki K1 arm and leg AMP prior.'}])
    if args.exptid is None:
        args.exptid = 'goalkeeper'

    # Jika pengguna menambahkan opsi --gui, buka jendela tampilan 3D
    if '--gui' in sys.argv:
        args.headless = False

    train(args)
