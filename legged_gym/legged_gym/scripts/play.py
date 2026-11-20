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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import select
import math

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, export_jit_to_onnx, load_onnx_policy, task_registry
from isaacgym import gymtorch, gymapi, gymutil
import torch
import faulthandler
from copy import deepcopy
from time import time, sleep

EXPORT_POLICY = False


def play(args):
    faulthandler.enable()
    
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if getattr(args, 'legacy', False):
        if args.task != 'k1':
            raise ValueError('--legacy is supported only for --task k1')
        # Original stance (hip/ankle roll +-0.10) and no target clamp, as older checkpoints were trained.
        env_cfg.control.limit_joint_targets = False
        init = env_cfg.init_state
        for name, v in (('left_hip_roll_joint', 0.10), ('left_ankle_roll_joint', -0.10),
                        ('right_hip_roll_joint', -0.10), ('right_ankle_roll_joint', 0.10)):
            init.default_joint_angles[name] = v
        init.init_pos = list(init.init_pos)
        init.init_pos[11], init.init_pos[15], init.init_pos[17], init.init_pos[21] = 0.10, -0.10, -0.10, 0.10
        print('[LEGACY] stance hip/ankle roll +-0.10, limit_joint_targets=False')
    env_cfg.env.num_envs = 6 if getattr(args, 'num_envs', None) is None else args.num_envs
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**17
    env_cfg.sim.physx.default_buffer_size_multiplier = 1
    env_cfg.env.episode_length_s = 12.0 if (args.task == 'k1' and getattr(args, 'walk_ckpt', None)) else 8.0
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    if getattr(args, 'seed', None) is None:
        env_cfg.seed = int(time() * 1000) % 100000

    if args.task != 'k1':
        k1_only_flags = [
            f"--{name}" for name in ('target_y', 'target_z', 'ball_speed', 'flight_time', 'start_x', 'start_y')
            if getattr(args, name, None) is not None
        ]
        if k1_only_flags:
            raise ValueError(f"{', '.join(k1_only_flags)} supported only for --task k1")
    if getattr(args, 'region', None) is not None:
        env_cfg.commands.custom_region = args.region
        print(f"[CUSTOM BALL TARGET] Target wilayah gawang dipaksa ke Region {args.region}")
    if getattr(args, 'target_y', None) is not None:
        env_cfg.commands.custom_target_y = args.target_y
        print(f"[CUSTOM BALL TARGET] Koordinat horizontal (Y) diset: {args.target_y} m")
    if getattr(args, 'target_z', None) is not None:
        env_cfg.commands.custom_target_z = args.target_z
        print(f"[CUSTOM BALL TARGET] Ketinggian bola (Z) diset: {args.target_z} m")
    if getattr(args, 'ball_speed', None) is not None:
        if args.task != 'k1':
            raise ValueError('--ball_speed is supported only for K1')
        if not math.isfinite(args.ball_speed) or args.ball_speed <= 0:
            raise ValueError('--ball_speed must be finite and positive (m/s)')
        if getattr(args, 'flight_time', None) is not None:
            raise ValueError('Choose --ball_speed or --flight_time, not both')
        env_cfg.commands.custom_ball_speed = args.ball_speed
        print(f'[BALL] Initial speed = {args.ball_speed:g} m/s (before drag/contact)')
    if getattr(args, 'flight_time', None) is not None:
        env_cfg.commands.custom_flight_time = args.flight_time
        print(f"[CUSTOM BALL TARGET] Durasi meluncur (flight_time) diset: {args.flight_time} detik")
    if getattr(args, 'start_x', None) is not None:
        env_cfg.commands.custom_start_x = args.start_x
        print(f"[CUSTOM BALL TARGET] Jarak awal bola (start_x) diset: {args.start_x} m")
    if getattr(args, 'start_y', None) is not None:
        env_cfg.commands.custom_start_y = args.start_y
        print(f"[CUSTOM BALL TARGET] Posisi samping awal bola (start_y) diset: {args.start_y} m")

    for name in dir(env_cfg.domain_rand):
        if name.startswith('randomize_') or name in ('push_robots', 'delay', 'continue_keep'):
            setattr(env_cfg.domain_rand, name, False)
    env_cfg.domain_rand.push_interval_s = 6

    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if env.viewer is not None:
        o0 = env.env_origins[0].detach().cpu().tolist()
        if args.task == 'k1':
            camera_pos = gymapi.Vec3(o0[0] + 3.4, o0[1] + 1.8, o0[2] + 1.35)
            camera_target = gymapi.Vec3(o0[0] - 0.10, o0[1], o0[2] + 0.50)
        else:
            camera_pos = gymapi.Vec3(o0[0] + 3.0, o0[1] + 2.2, o0[2] + 1.4)
            camera_target = gymapi.Vec3(o0[0], o0[1], o0[2] + 0.8)
        env.gym.viewer_camera_look_at(env.viewer, None, camera_pos, camera_target)

        key_bindings = [
            (gymapi.KEY_0, "region_0"), (gymapi.KEY_NUMPAD_0, "region_0"),
            (gymapi.KEY_1, "region_1"), (gymapi.KEY_NUMPAD_1, "region_1"),
            (gymapi.KEY_2, "region_2"), (gymapi.KEY_NUMPAD_2, "region_2"),
            (gymapi.KEY_3, "region_3"), (gymapi.KEY_NUMPAD_3, "region_3"),
            (gymapi.KEY_4, "region_4"), (gymapi.KEY_NUMPAD_4, "region_4"),
            (gymapi.KEY_5, "region_5"), (gymapi.KEY_NUMPAD_5, "region_5"),
            (gymapi.KEY_6, "region_random"), (gymapi.KEY_NUMPAD_6, "region_random"), (gymapi.KEY_C, "region_random"),
            (gymapi.KEY_LEFT, "target_left"), (gymapi.KEY_A, "target_left"),
            (gymapi.KEY_RIGHT, "target_right"), (gymapi.KEY_D, "target_right"),
            (gymapi.KEY_UP, "target_up"), (gymapi.KEY_W, "target_up"),
            (gymapi.KEY_DOWN, "target_down"), (gymapi.KEY_S, "target_down"),
            (gymapi.KEY_EQUAL, "faster"), (gymapi.KEY_NUMPAD_ADD, "faster"), (gymapi.KEY_F, "faster"),
            (gymapi.KEY_MINUS, "slower"), (gymapi.KEY_NUMPAD_SUBTRACT, "slower"), (gymapi.KEY_G, "slower"),
            (gymapi.KEY_RIGHT_BRACKET, "dist_farther"), (gymapi.KEY_E, "dist_farther"),
            (gymapi.KEY_LEFT_BRACKET, "dist_closer"), (gymapi.KEY_Q, "dist_closer"),
            (gymapi.KEY_R, "shoot_now"), (gymapi.KEY_ENTER, "shoot_now"), (gymapi.KEY_NUMPAD_ENTER, "shoot_now"),
        ]
        for key, act_name in key_bindings:
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, act_name)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    all_env_ids = torch.arange(env.num_envs, device=env.device)

    walk_policy = None
    two_policy_ctrl = None
    if args.task == 'k1' and getattr(args, 'walk_ckpt', None):
        from legged_gym.utils.k1_two_policy import (
            K1TwoPolicyController,
            STATE_GOALKEEPER,
            STATE_READY_HOME,
            STATE_WALK_BACK,
            load_walk_policy,
        )
        walk_policy, walk_meta = load_walk_policy(args.walk_ckpt, device=env.device)
        two_policy_ctrl = K1TwoPolicyController(
            num_envs=env.num_envs,
            num_actions=env.num_actions,
            dt=env.dt,
            device=env.device,
            walk_history_len=walk_meta['history_len'],
            walk_one_step_dim=walk_meta['one_step_dim'],
        )
        print(f"[2-POLICY] Policy 2 walk-back aktif: {args.walk_ckpt}")

    def safe_reset_all():
        try:
            env.reset_idx(all_env_ids)
        except ValueError as err:
            msg = str(err)
            if "need at least" in msg and getattr(env.cfg.commands, "custom_ball_speed", None) is not None:
                import re
                m = re.search(r"need at least ([0-9.]+) m/s", msg)
                min_spd = round(float(m.group(1)) + 0.15, 2) if m else 8.5
                print(f"[AUTO-ADJUST] Kecepatan {env.cfg.commands.custom_ball_speed:g} m/s terlalu pelan untuk jarak ini -> dinaikkan ke {min_spd:g} m/s")
                env.cfg.commands.custom_ball_speed = min_spd
                env.reset_idx(all_env_ids)
            else:
                raise

    safe_reset_all()
    env.compute_observations()
    obs = env.get_observations()

    if EXPORT_POLICY and hasattr(ppo_runner.alg.actor_critic, 'ball_estimator'):
        policy_name = 'goalkeeper'
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print('Exported policy as jit script to: ', path)

        jit_path = os.path.join(path, f'{policy_name}.pt')
        jit_model = torch.jit.load(jit_path)
        dummy_input = torch.randn(1, obs.shape[1], device='cpu')
        onnx_path = os.path.join(path, f'{policy_name}.onnx')
        export_jit_to_onnx(jit_model, onnx_path, dummy_input)

        policy = load_onnx_policy(onnx_path)

    region_names = {
        0: "Region 0 (Tengah Kiri)",
        1: "Region 1 (Tengah Kanan)",
        2: "Region 2 (Pojok Atas Kiri)",
        3: "Region 3 (Pojok Atas Kanan)",
        4: "Region 4 (Bawah Kiri)",
        5: "Region 5 (Bawah Kanan)",
    }

    shot_outcome_printed = False

    def shoot_now(status_msg):
        nonlocal obs, shot_outcome_printed
        shot_outcome_printed = False
        safe_reset_all()
        if two_policy_ctrl is not None:
            two_policy_ctrl.reset()
        env.compute_observations()
        obs = env.get_observations()
        x0_now = float(env.ball_start[0, 0] - env.env_origins[0, 0])
        y0_now = float(env.ball_start[0, 1] - env.env_origins[0, 1])
        y_now = float(env.ball_end[0, 1] - env.env_origins[0, 1])
        z_now = float(env.ball_end[0, 2])
        vx_now = float(env.ball_states[0, 7])
        spd_now = float(torch.norm(env.ball_states[0, 7:10]))
        print(f"⚽ [TEMBAK] {status_msg} | Jarak X={x0_now:.2f}m (Y0={y0_now:+.2f}m) -> Sasaran (Y={y_now:+.2f}m, Z={z_now:.2f}m) | Speed={spd_now:.1f}m/s (Vx={vx_now:.1f}m/s)")

    def handle_action(act):
        if args.task != 'k1' and act != "shoot_now":
            print("[INFO] Kontrol target/kecepatan/jarak kustom hanya didukung untuk --task k1.")
            return
        cmds = env.cfg.commands
        if act.startswith("region_") and act != "region_random":
            r = int(act.split("_")[1])
            cmds.custom_region = r
            cmds.custom_target_y = None
            cmds.custom_target_z = None
            shoot_now(f"Target dikunci ke {region_names[r]}")
        elif act == "region_random":
            cmds.custom_region = None
            cmds.custom_target_y = None
            cmds.custom_target_z = None
            shoot_now("Mode Acak Semua Region")
        elif act == "reset_all":
            cmds.custom_region = None
            cmds.custom_target_y = None
            cmds.custom_target_z = None
            cmds.custom_flight_time = None
            cmds.custom_ball_speed = None
            cmds.custom_start_x = None
            cmds.custom_start_y = None
            shoot_now("Reset Semua (Target, Kecepatan & Jarak kembali acak default)")
        elif act in ("target_left", "target_right", "target_up", "target_down"):
            cmds.custom_region = None
            cur_y = cmds.custom_target_y if getattr(cmds, 'custom_target_y', None) is not None else float(env.ball_end[0, 1] - env.env_origins[0, 1])
            cur_z = cmds.custom_target_z if getattr(cmds, 'custom_target_z', None) is not None else float(env.ball_end[0, 2])
            if act == "target_left":
                cur_y = min(0.85, round(cur_y + 0.10, 2))
            elif act == "target_right":
                cur_y = max(-0.85, round(cur_y - 0.10, 2))
            elif act == "target_up":
                cur_z = min(1.15, round(cur_z + 0.10, 2))
            elif act == "target_down":
                cur_z = max(0.12, round(cur_z - 0.10, 2))
            cmds.custom_target_y = cur_y
            cmds.custom_target_z = cur_z
            shoot_now(f"Target manual digeser ke Y={cur_y:+.2f}m, Z={cur_z:.2f}m")
        elif act in ("faster", "slower"):
            if args.task == 'k1':
                cur_spd = getattr(cmds, 'custom_ball_speed', None)
                if cur_spd is None:
                    cur_spd = round(float(torch.norm(env.ball_states[0, 7:10])) * 2.0) / 2.0
                cur_spd = min(25.0, round(cur_spd + 0.5, 2)) if act == "faster" else max(4.5, round(cur_spd - 0.5, 2))
                cmds.custom_flight_time = None
                cmds.custom_ball_speed = cur_spd
                shoot_now(f"Kecepatan bola diset = {cur_spd:g} m/s")
            else:
                cur_t = cmds.custom_flight_time if getattr(cmds, 'custom_flight_time', None) is not None else 0.70
                cur_t = max(0.40, round(cur_t - 0.05, 2)) if act == "faster" else min(1.20, round(cur_t + 0.05, 2))
                cmds.custom_ball_speed = None
                cmds.custom_flight_time = cur_t
                shoot_now(f"Durasi terbang bola (flight_time) = {cur_t:.2f}s")
        elif act in ("dist_farther", "dist_closer"):
            cur_x = getattr(cmds, 'custom_start_x', None)
            if cur_x is None:
                cur_x = round(float(env.ball_start[0, 0] - env.env_origins[0, 0]) * 2.0) / 2.0
            cur_x = min(15.0, round(cur_x + 0.5, 2)) if act == "dist_farther" else max(1.5, round(cur_x - 0.5, 2))
            cmds.custom_start_x = cur_x
            shoot_now(f"Jarak tendangan (start_x) diset = {cur_x:g} m")
        elif act == "shoot_now":
            shoot_now("Tembak ulang")

    def poll_terminal_input():
        if args.headless or not sys.stdin.isatty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return
        line = sys.stdin.readline().strip()
        if not line:
            handle_action("shoot_now")
            return
        low = line.lower()
        if low in ("q", "quit", "exit"):
            sys.exit(0)
        if low in ("reset", "default"):
            handle_action("reset_all")
            return
        if low in ("acak", "random", "6"):
            handle_action("region_random")
            return
        parts = low.replace(",", " ").replace("=", " ").split()
        if args.task != 'k1':
            print("[INFO] Kontrol target/kecepatan/jarak kustom hanya didukung untuk --task k1; tekan Enter untuk tembak ulang atau `q` untuk keluar.")
            return
        cmds = env.cfg.commands
        try:
            if len(parts) == 1 and int(parts[0]) in range(6):
                handle_action(f"region_{int(parts[0])}")
                return
            if len(parts) == 2 and parts[0] in ("r", "reg", "region") and int(parts[1]) in range(6):
                handle_action(f"region_{int(parts[1])}")
                return
            if len(parts) == 2 and parts[0] in ("v", "spd", "speed", "kecepatan", "ball_speed"):
                spd = max(4.0, min(25.0, float(parts[1])))
                cmds.custom_flight_time = None
                cmds.custom_ball_speed = spd
                shoot_now(f"Input terminal kecepatan = {spd:g} m/s")
                return
            if len(parts) == 2 and parts[0] in ("x", "start_x", "dist", "distance", "jarak", "d"):
                dx = max(1.5, min(15.0, float(parts[1])))
                cmds.custom_start_x = dx
                shoot_now(f"Input terminal jarak (start_x) = {dx:g} m")
                return
            if len(parts) == 2 and parts[0] in ("y0", "start_y"):
                sy = max(-1.2, min(1.2, float(parts[1])))
                cmds.custom_start_y = sy
                shoot_now(f"Input terminal posisi samping awal (start_y) = {sy:+.2f} m")
                return
            if len(parts) == 3 and parts[0] in ("vx", "sd", "speed_dist", "kecepatan_jarak"):
                spd = max(4.0, min(25.0, float(parts[1])))
                dx = max(1.5, min(15.0, float(parts[2])))
                cmds.custom_flight_time = None
                cmds.custom_ball_speed = spd
                cmds.custom_start_x = dx
                shoot_now(f"Input terminal kecepatan = {spd:g} m/s, jarak = {dx:g} m")
                return
            if len(parts) in (2, 3, 4):
                ty, tz = float(parts[0]), float(parts[1])
                cmds.custom_region = None
                cmds.custom_target_y = max(-1.10, min(1.10, ty))
                cmds.custom_target_z = max(0.11, min(1.20, tz))
                if len(parts) >= 3:
                    val3 = float(parts[2])
                    if val3 >= 2.0 and args.task == 'k1':
                        cmds.custom_flight_time = None
                        cmds.custom_ball_speed = min(25.0, val3)
                    else:
                        cmds.custom_ball_speed = None
                        cmds.custom_flight_time = max(0.30, min(1.50, val3))
                if len(parts) == 4:
                    cmds.custom_start_x = max(1.5, min(15.0, float(parts[3])))
                shoot_now(f"Input terminal Y={cmds.custom_target_y:+.2f}m, Z={cmds.custom_target_z:.2f}m")
                return
        except ValueError:
            pass
        print("[INFO] Perintah terminal: `0`-`5` (region), `v 12` (kecepatan 12 m/s), `x 4.5` (jarak 4.5 m), `vx 12 4.5` (kecepatan & jarak), `Y Z [V] [X]` (misal `-0.5 0.8 12 4.5`), `acak`, `reset`, atau Enter.")

    step_idx = 0
    prev_ctrl_state0 = 0
    max_steps = 3000 if args.headless else 10**9
    while step_idx < max_steps:
        while env.viewer_action_events:
            handle_action(env.viewer_action_events.pop(0))
        poll_terminal_input()

        if two_policy_ctrl is not None:
            actions = two_policy_ctrl.step_action(env, obs, policy, walk_policy)
            cur_state0 = int(two_policy_ctrl.state[0].item())
            if cur_state0 != prev_ctrl_state0:
                rx0 = float(env.root_states[0, 0] - env.env_origins[0, 0])
                ry0 = float(env.root_states[0, 1] - env.env_origins[0, 1])
                if cur_state0 == STATE_WALK_BACK:
                    print(f"🚶 [WALK-BACK] Bola 100% aman -> jalan pulang dari (X={rx0:+.2f}m, Y={ry0:+.2f}m) ke (0, 0)")
                elif cur_state0 == STATE_READY_HOME:
                    print(f"🏠 [HOME] Tiba di depan gawang (X={rx0:+.2f}m, Y={ry0:+.2f}m), stance siap")
                elif cur_state0 == STATE_GOALKEEPER and prev_ctrl_state0 != STATE_GOALKEEPER:
                    print("🧤 [GOALKEEPER] Bola datang -> Policy 1 aktif")
                prev_ctrl_state0 = cur_state0
        else:
            actions = policy(obs.detach())
        obs, privileged_obs, rews, dones, infos, _, _ = env.step(actions.detach())
        if two_policy_ctrl is not None:
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                two_policy_ctrl.reset(done_ids)
                if bool(dones[0]):
                    prev_ctrl_state0 = 0
        if args.task == 'k1' and not shot_outcome_printed:
            bx0 = float(env.ball_states[0, 0] - env.env_origins[0, 0])
            by0 = float(env.ball_states[0, 1] - env.env_origins[0, 1])
            bz0 = float(env.ball_states[0, 2] - env.env_origins[0, 2])
            if env.goal_conceded_flag[0] > 0:
                print(f"🥅 [HASIL] GOL! Bola masuk gawang (X={bx0:.2f}m, Y={by0:+.2f}m, Z={bz0:.2f}m)")
                shot_outcome_printed = True
            elif bool(env._k1_ball_safe()[0]):
                print(f"🧤 [HASIL] SAVE! Bola berhasil ditepis/ditahan (X={bx0:.2f}m, Y={by0:+.2f}m, Z={bz0:.2f}m)")
                shot_outcome_printed = True
            elif bx0 < (env.cfg.goal.line_x - env.cfg.goal.ball_radius):
                print(f"↗️ [HASIL] OUT! Bola meleset di luar gawang (X={bx0:.2f}m, Y={by0:+.2f}m, Z={bz0:.2f}m)")
                shot_outcome_printed = True
        if dones[0]:
            shot_outcome_printed = False
        step_idx += 1
        if not args.headless:
            sleep(env.dt)

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    if args.exptid is None:
        args.exptid = 'goalkeeper'
    if '--headless' not in sys.argv:
        args.headless = False
    play(args)
