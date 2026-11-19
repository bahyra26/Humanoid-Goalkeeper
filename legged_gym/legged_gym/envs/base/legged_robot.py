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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import random
import torch
from torch import Tensor
import torchvision
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.shot_outcome import resolved_save, contact_candidate, shot_outcome_masks
from legged_gym.utils.goal_detection import ball_crossed_goal
from legged_gym.utils.goal_visuals import build_goal_geometry, make_goal_lines
from legged_gym.utils.ball_launch import launch_at_speed
from .legged_robot_config import LeggedRobotCfg


from legged_gym.envs.g1.g1_utils import (
    MotionLib, 
    load_imitation_dataset
)



def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.amp_obs_type = self.cfg.amp.obs_type
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history

        self.actor_obs_length = self.cfg.env.num_observations

        self._init_buffers()
        self._prepare_reward_function()
        self.num_amp_obs = cfg.amp.num_obs
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.
        # step physics and render each frame
        self.render()
        need_substep = False
        if self.cfg.asset.name == "k1":
            self.substep_max_ball_force.zero_()
            self.substep_min_arm_dist.fill_(10.0)
            self.substep_min_robot_dist.fill_(10.0)
            need_substep = bool(((self.stop_flag == 0) & ((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.85)).any())
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[_]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if need_substep:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                sub_f = torch.norm(self.ball_contact_forces[:, 0, :], dim=-1)
                sub_arm_d, sub_rob_d = self._compute_k1_ball_dists()
                self.substep_max_ball_force = torch.maximum(self.substep_max_ball_force, sub_f)
                self.substep_min_arm_dist = torch.minimum(self.substep_min_arm_dist, sub_arm_d)
                self.substep_min_robot_dist = torch.minimum(self.substep_min_robot_dist, sub_rob_d)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def get_amp_observations(self):
        """ with keys
        """
        indices = getattr(self, "amp_joint_indices", None)
        if indices is not None:
            return self.dof_pos[:, indices].clone()
        return self.dof_pos.clone()



    def _draw_viewer_overlays(self):
        """Draw non-physical 3D goal frame, net mesh, and outcome status color in GUI viewer."""
        if self.viewer is None or not hasattr(self.cfg, "goal") or not getattr(self, "envs", None):
            return
        goal = self.cfg.goal
        cache = getattr(self, "_goal_wireframe_cache", None)
        if cache is None:
            cache = build_goal_geometry(
                line_x=goal.line_x,
                half_width=goal.half_width,
                height=goal.height,
                ball_radius=goal.ball_radius,
            )
            self._goal_wireframe_cache = cache
        frame_verts, net_verts = cache
        self.gym.clear_lines(self.viewer)

        is_k1 = getattr(self.cfg.asset, "name", None) == "k1"
        safe_mask = self._k1_ball_safe() if (is_k1 and hasattr(self, "ball_states")) else None
        target_tensor = getattr(self, "ball_end", getattr(self, "end_target", None))
        n_draw = min(self.num_envs, 16)
        origins_np = self.env_origins[:n_draw].detach().cpu().numpy()
        for i in range(n_draw):
            status = "live"
            if is_k1 and hasattr(self, "goal_conceded_flag") and self.goal_conceded_flag[i] > 0:
                status = "goal"
            elif safe_mask is not None and bool(safe_mask[i]):
                status = "save" if (hasattr(self, "stop_flag") and self.stop_flag[i] > 0) else "out"
            elif is_k1 and hasattr(self, "ball_states") and float(self.ball_states[i, 0] - self.env_origins[i, 0]) < (goal.line_x - goal.ball_radius):
                status = "out"
            target_yz = None
            if status == "live" and target_tensor is not None:
                target_yz = (
                    float(target_tensor[i, 1] - self.env_origins[i, 1]),
                    float(target_tensor[i, 2] - self.env_origins[i, 2]),
                )
            eo_gym = self.gym.get_env_origin(self.envs[i])
            offset_xyz = (
                float(origins_np[i, 0]) - float(eo_gym.x),
                float(origins_np[i, 1]) - float(eo_gym.y),
                float(origins_np[i, 2]) - float(eo_gym.z),
            )
            verts, colors = make_goal_lines(
                frame_verts,
                net_verts,
                status=status,
                line_x=goal.line_x,
                target_yz=target_yz,
                offset_xyz=offset_xyz,
            )
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.catchstep -= 1
        self.extras.pop("shot_results", None)
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3].clone()

        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity[:] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        
        # compute contact related quantities

        # compute joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)
        
        self._post_physics_step_callback()


        balllocal =  self.ball_states[:, 0] - self.env_origins[:, 0]
        if self.cfg.asset.name != "k1":
            approachidx = ((balllocal < 0.5) & (balllocal > 0.1) & (self.ball_states[:,7]  - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()
            self.end_target[approachidx, :] = self.ball_states[approachidx, :3].clone()
            self.end_target[:, 0] = torch.clip(self.end_target[:, 0], min = self.env_origins[:, 0] + 0.1, max = self.env_origins[:, 0] + 1.0)
        else:
            self.cur_min_arm_dist, self.cur_min_robot_dist = self._compute_k1_ball_dists()
            approachidx = ((balllocal < 0.80) & (balllocal > 0.05) & (~self._k1_ball_safe()) & (self.goal_conceded_flag == 0)).nonzero(as_tuple=False).flatten()
            self.end_target[approachidx, :] = self.ball_states[approachidx, :3].clone()
            self.end_target[:, 0] = torch.clip(self.end_target[:, 0], min = self.env_origins[:, 0] + 0.08, max = self.env_origins[:, 0] + 0.80)
            goal = self.cfg.goal
            conceded = ball_crossed_goal(
                (self.prev_ball_pos - self.env_origins).T,
                (self.ball_states[:, :3] - self.env_origins).T,
                goal.line_x, goal.half_width, goal.height, goal.ball_radius,
            )
            self.goal_conceded_flag[conceded] = 1.0
            self.success_flag[conceded] = 0.0
            self.arm_save_flag[conceded] = 0.0

        hand_pos = self.rigid_body_states[:, self.hand_indices, :3].clone() 
        hand_pos_l, hand_pos_r =  hand_pos[:,0,:], hand_pos[:,1,:]
        
        region0_dis = torch.norm(self.end_target[self.end_regions == 0] - hand_pos_l[self.end_regions == 0], dim = 1)
        region1_dis = torch.norm(self.end_target[self.end_regions == 1] - hand_pos_r[self.end_regions == 1], dim = 1)
        region2_dis = torch.norm(self.end_target[self.end_regions == 2] - hand_pos_l[self.end_regions == 2], dim = 1)
        region3_dis = torch.norm(self.end_target[self.end_regions == 3] - hand_pos_r[self.end_regions == 3], dim = 1)
        region4_dis = torch.norm(self.end_target[self.end_regions == 4] - hand_pos_l[self.end_regions == 4], dim = 1)
        region5_dis = torch.norm(self.end_target[self.end_regions == 5] - hand_pos_r[self.end_regions == 5], dim = 1)

        self.dist[self.end_regions == 0] = region0_dis
        self.dist[self.end_regions == 1] = region1_dis
        self.dist[self.end_regions == 2] = region2_dis
        self.dist[self.end_regions == 3] = region3_dis
        self.dist[self.end_regions == 4] = region4_dis
        self.dist[self.end_regions == 5] = region5_dis        


        # compute observations, rewards, resets, ...
        self.check_termination()

        self.compute_reward()

        if getattr(self.cfg.env, 'observe_recovery', False) and hasattr(self, 'evaluation_shot'):
            self.extras['evaluation_frame'] = self._evaluation_diagnostics(torch.arange(self.num_envs, device=self.device))

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        termination_privileged_obs = self.compute_termination_observations(env_ids)

        if self.cfg.amp.amp_coef > 0:
            self.extras['terminal_amp_observations'] = self.get_amp_observations()[env_ids].clone()
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.prev_ball_lin_vel[:] = self.ball_states[:, 7:10]
        self.prev_ball_pos[:] = self.ball_states[:, :3]
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        min_knee_h = getattr(self.cfg.rewards, "min_knee_height", 0.10)
        self.reset_buf = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < min_knee_h
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        tilt = torch.norm(self.projected_gravity[:, 0:2], dim=-1)
        if getattr(self.cfg.asset, "name", None) == "k1" and hasattr(self, "ball_states") and hasattr(self, "_k1_ball_safe"):
            # Allow controlled diving/kicking tilt (~62 deg) while live; tighten to 0.80 once safe for 2-foot recovery.
            ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
            in_flight_live = (ball_local_x > -0.02) & (~self._k1_ball_safe(include_behind=True))
            tilt_limit = torch.where(in_flight_live, torch.full_like(tilt, 0.90), torch.full_like(tilt, 0.82))
            self.gravity_termination_buf = tilt > tilt_limit
        else:
            self.gravity_termination_buf = tilt > 0.8
        sharpforce_buf = torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) > 1.5 * self.cfg.rewards.max_contact_force

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= sharpforce_buf
        if getattr(self.cfg.env, 'observe_recovery', False):
            if not self.play or self.cfg.asset.name != "k1" or self.num_envs != 1:
                raise ValueError('Recovery observation requires K1 play mode with one environment')
            # Observe threshold violations without cutting off recovery; bounded by timeout.
            self.reset_buf = self.time_out_buf.clone()



    def _evaluation_diagnostics(self, env_ids):
        """Snapshot measured state before automatic reset; evaluation only."""
        diagnostics = {
            'contact_flag_at_end': self.success_flag[env_ids] > 0,
            'goal': self.goal_conceded_flag[env_ids] > 0,
            'episode_steps': self.episode_length_buf[env_ids],
            'upper_gravity_z': self.projected_gravity[env_ids, 2],
            'upper_tilt_sine': torch.norm(self.projected_gravity[env_ids, :2], dim=-1),
            'base_height_m': self.root_states[env_ids, 2] - self.env_origins[env_ids, 2],
            'timeout': self.time_out_buf[env_ids],
            'low_knee': torch.min(self.rigid_body_states[env_ids][:, self.knee_indices, 2], dim=-1).values < getattr(self.cfg.rewards, 'min_knee_height', 0.10),
            'excessive_tilt': self.gravity_termination_buf[env_ids],
            'excessive_foot_force': torch.mean(torch.norm(self.contact_forces[env_ids][:, self.contact_feet_indices, :], dim=-1), dim=-1) > 1.5 * self.cfg.rewards.max_contact_force,
        }
        for axis, index in zip('xyz', range(3)):
            diagnostics[f'final_ball_{axis}_m'] = self.ball_states[env_ids, index] - self.env_origins[env_ids, index]
            diagnostics[f'final_ball_v{axis}_m_s'] = self.ball_states[env_ids, 7 + index]
        return {key: value.clone() for key, value in diagnostics.items()}

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        if self.cfg.asset.name != "k1":
            self.success_rate[env_ids, 0] += self.success_flag[env_ids]
            self.success_rate[env_ids, 1] += (self.episode_length_buf[env_ids] > 0).float()
            saved_true_sr = None
            saved_arm_sr = None
            saved_goal_rate = None
            saved_fall_rate = None
        else:
            saved_fall_rate = (~self.time_out_buf[env_ids]).float().mean()
            local_x = self.ball_states[env_ids, 0] - self.env_origins[env_ids, 0]
            local_y = self.ball_states[env_ids, 1] - self.env_origins[env_ids, 1]
            ball_speed = torch.norm(self.ball_states[env_ids, 7:10], dim=-1)
            conceded = self.goal_conceded_flag[env_ids] > 0
            contact = self.success_flag[env_ids] > 0
            saved, _, missed, _, unresolved = shot_outcome_masks(
                conceded, contact, local_x, self.ball_states[env_ids, 7],
                self.cfg.goal.line_x, self.cfg.goal.ball_radius,
                y=local_y, half_width=self.cfg.goal.half_width, speed=ball_speed,
            )
            true_saves = saved.float()
            self.success_rate[env_ids, 0] += true_saves
            self.success_rate[env_ids, 1] += (self.episode_length_buf[env_ids] > 0).float()
            saved_true_sr = true_saves.mean()
            saved_arm_sr = (true_saves * self.arm_save_flag[env_ids]).mean()
            saved_goal_rate = conceded.float().mean()
            # Capture terminal records before reset overwrites ball and flags.
            self.extras['shot_results'] = {
                'env_ids': env_ids.clone(), 'save': saved.clone(), 'goal': conceded.clone(),
                'miss_without_contact': missed.clone(), 'unresolved': unresolved.clone(),
                'fall': (~self.time_out_buf[env_ids]).clone(),
                'episode_steps': self.episode_length_buf[env_ids].clone(),
            }
            if hasattr(self, 'evaluation_shot'):
                self.extras['shot_results'].update(self._evaluation_diagnostics(env_ids))

        reachcount = self.success_rate[:,1] > 9    
        self.success_rate[reachcount,2] = self.success_rate[reachcount, 0] / self.success_rate[reachcount, 1]
        self.success_rate[reachcount, 0:2] *= 0.        


        self.refresh_actor_rigid_shape_props(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.obs_buf[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # sinkronisasi state post-reset agar observasi frame pertama episode baru bersih
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.torso_pos[env_ids] = self.root_states[env_ids, 0:3]
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])

        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if saved_true_sr is not None:
            self.extras["episode"]["true_save_rate"] = saved_true_sr
            self.extras["episode"]["arm_save_rate"] = saved_arm_sr
            self.extras["episode"]["goal_conceded_rate"] = saved_goal_rate
            self.extras["episode"]["miss_without_contact_rate"] = missed.float().mean()
            self.extras["episode"]["unresolved_rate"] = unresolved.float().mean()
        if saved_fall_rate is not None:
            self.extras["episode"]["fall_rate"] = saved_fall_rate


        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

            
        if (self.common_step_counter - self.last_step_counter) > 500 and not self.play:
            if self.cfg.asset.name != "k1":
                self.startstep = 50 - random.randint(3, 10)
                survival_level = min(int(torch.mean(self.episode_length_buf[env_ids].float()) / 50.), 3)
                self.curriculumupdate = survival_level
                expand = 0.3 * self.curriculumupdate
                self.command_ranges[:, 0] = torch.clip(self.command_ranges[:, 0] - expand, self.command_bound[:,0], self.command_bound[:,1])
                self.command_ranges[:, 1] = torch.clip(self.command_ranges[:, 1] + expand, self.command_bound[:,0], self.command_bound[:,1])
                self.command_ranges[:, 2] = torch.clip(self.command_ranges[:, 2] - expand, self.command_bound[:,2], self.command_bound[:,3])
                self.command_ranges[:, 3] = torch.clip(self.command_ranges[:, 3] + expand, self.command_bound[:,2], self.command_bound[:,3])
            else:
                self.startstep = 49
                mean_sr = torch.mean(self.success_rate[:, 2]).item()
                if self.common_step_counter < 3000 or mean_sr < 0.30:
                    sr_level = 0
                elif mean_sr < 0.50:
                    sr_level = 1
                elif mean_sr < 0.65:
                    sr_level = 2
                else:
                    sr_level = 3
                self.curriculumupdate = max(int(self.curriculumupdate), sr_level)
                expand = 0.18 * self.curriculumupdate
                self.command_ranges[:, 0] = torch.clip(self.command_init_ranges[:, 0] - expand, self.command_bound[:,0], self.command_bound[:,1])
                self.command_ranges[:, 1] = torch.clip(self.command_init_ranges[:, 1] + expand, self.command_bound[:,0], self.command_bound[:,1])
                self.command_ranges[:, 2] = torch.clip(self.command_init_ranges[:, 2] - expand, self.command_bound[:,2], self.command_bound[:,3])
                self.command_ranges[:, 3] = torch.clip(self.command_init_ranges[:, 3] + expand, self.command_bound[:,2], self.command_bound[:,3])

            self.last_step_counter = self.common_step_counter


        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        if self.cfg.asset.name != "k1":
            if "eereach" in self.reward_scales:
                self.reward_scales["eereach"] = self.eereach_init * (1 + 0.5 * self.curriculumupdate)
            if "success" in self.reward_scales:
                self.reward_scales["success"] = self.success_init * (1 + 0.5 * self.curriculumupdate)
            if "stopball" in self.reward_scales:
                self.reward_scales["stopball"] = self.stop_init * (1 + 0.5 * self.curriculumupdate)

            if "dof_pos_limits" in self.reward_scales:
                self.reward_scales["dof_pos_limits"] = self.dof_pos_init * (1.0 + max(0, int(self.curriculumupdate) - 1))
            if "torque_limits" in self.reward_scales:
                self.reward_scales["torque_limits"] = self.torque_init * (1.0 + max(0, int(self.curriculumupdate) - 1))


        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """

        hand_pos = self.rigid_body_states[:, self.hand_indices, :3].clone()
        rb_torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        rb_torso_quat = self.rigid_body_states[:, self.torso_index, 3:7]
        hand_pos_l = quat_rotate_inverse(rb_torso_quat, hand_pos[:, 0, :] - rb_torso_pos)
        hand_pos_r = quat_rotate_inverse(rb_torso_quat, hand_pos[:, 1, :] - rb_torso_pos)

        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish        

        if self.cfg.asset.name != "k1":
            flying = ((end_target_local[:,0] > 0.05) & (end_target_local[:,0] < 3.4) & (end_target_local[:,1] > -2.0) & (end_target_local[:,1] < 2.0) & (end_target_local[:,2] < 1.8) & (self.catchstep > 0.) & ((end_target_local[:,0] < self.ball_last[:,0]) | (self.ball_last[:,0] == 0.))).view(-1, 1)
            random_vanish = (self.catchstep > self.vanish_step).view(-1, 1)
        else:
            flying = ((end_target_local[:,0] > -0.02) & (end_target_local[:,0] < 16.0) & (end_target_local[:,1] > -2.2) & (end_target_local[:,1] < 2.2) & (end_target_local[:,2] < 2.2) & (~self._k1_ball_safe()) & (self.goal_conceded_flag == 0)).view(-1, 1)
            random_vanish = (~((self.catchstep < 25) & (self.catchstep > 25 - self.vanish_step))).view(-1, 1)
        self.ball_last = end_target_local

        current_obs = torch.cat((   
                                    end_target_local, # * ball_in_camera_fov,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1) / 3.,
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    hand_pos_r,
                                    hand_pos_l,
                                    self.dist.unsqueeze(-1),
                                    ),dim=-1)

        # add noise if needed
        current_actor_obs = torch.clone(current_obs[:, :self.num_one_step_obs])

        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[:(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)]
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying * random_vanish
        else:
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)

        self.privileged_obs_buf = current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        
        hand_pos = self.rigid_body_states[:, self.hand_indices, :3].clone()
        rb_torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        rb_torso_quat = self.rigid_body_states[:, self.torso_index, 3:7]
        hand_pos_l = quat_rotate_inverse(rb_torso_quat, hand_pos[:, 0, :] - rb_torso_pos)
        hand_pos_r = quat_rotate_inverse(rb_torso_quat, hand_pos[:, 1, :] - rb_torso_pos)
        

        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish

        current_obs = torch.cat((   
                                    end_target_local, # * ball_in_camera_fov,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1) / 3.,
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    hand_pos_r,
                                    hand_pos_l,
                                    self.dist.unsqueeze(-1),
                                    ),dim=-1)



        return current_obs[env_ids]
    
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        start = time()
        print("*"*80)
        print("Start creating ground...")
        self._create_ground_plane()
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

        

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if not (self.cfg.domain_rand.randomize_friction or self.cfg.domain_rand.randomize_restitution):
            return
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
                
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.curriculum_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                m = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
                r = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
                self.curriculum_dof_pos_limits[i, 0] = m - 0.5 * r * 1.1
                self.curriculum_dof_pos_limits[i, 1] = m + 0.5 * r * 1.1


        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def refresh_actor_rigid_body_props(self, env_ids):
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload[env_ids] = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (len(env_ids), 1), device=self.device)
            
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement[env_ids] = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (len(env_ids), 3), device=self.device)
            
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            rigid_body_props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            rigid_body_props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
            
            if self.cfg.domain_rand.randomize_link_mass:
                rng = self.cfg.domain_rand.link_mass_range
                for i in range(1, len(rigid_body_props)):
                    scale = np.random.uniform(rng[0], rng[1])
                    rigid_body_props[i].mass = scale * self.default_rigid_body_mass[i]
            
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, rigid_body_props, recomputeInertia=True)



    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """        
        self._randomize_balls()
        if self.cfg.asset.name != "k1":
            if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
                self._push_robots()
        else:
            if self.cfg.domain_rand.push_robots and not self.play:
                self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller

        actions_scaled = actions * self.action_scale_vec
        self.joint_pos_target = self.default_dof_poses + actions_scaled
                
        hold_initial_pose = self.catchstep > self.startstep
        if self.cfg.asset.name == "k1":
            # K1 observations hide the ball at equality too; do not act blind for one step.
            hold_initial_pose = self.catchstep >= self.startstep
        self.joint_pos_target[hold_initial_pose] = self.init_dof_pos[hold_initial_pose]
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if getattr(self.cfg.control, "limit_joint_targets", False):
                # Simulation guard only: URDF limits do not certify hardware safety.
                self.joint_pos_target = torch.maximum(
                    torch.minimum(self.joint_pos_target, self.dof_pos_limits[:, 1]),
                    self.dof_pos_limits[:, 0])
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)

        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.standpos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.standpos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        if self.cfg.domain_rand.continue_keep and len(env_ids) < self.num_envs:
            valid_donors = (
                (self.reset_buf == 0)
                & (self.episode_length_buf > 5)
                & (torch.abs(self.root_states[:, 2] - self.base_init_state[2]) < 0.05)
                & (torch.norm(self.projected_gravity[:, :2], dim=-1) < 0.25)
            )
            keep_mask = torch.rand(len(env_ids), device=self.device) > 0.2
            if keep_mask.any() and valid_donors.any():
                target_envs = env_ids[keep_mask]
                target_regions = self.end_regions[target_envs]
                for reg in range(6):
                    reg_targets = target_envs[target_regions == reg]
                    if len(reg_targets) == 0:
                        continue
                    reg_donors = (valid_donors & (self.end_regions == reg)).nonzero(as_tuple=False).flatten()
                    if len(reg_donors) > 0:
                        picked = reg_donors[torch.randint(0, len(reg_donors), (len(reg_targets),), device=self.device)]
                        self.dof_pos[reg_targets] = self.dof_pos[picked].clone()

        self.init_dof_pos[env_ids] = self.dof_pos[env_ids].clone()


        self.dof_vel[env_ids] = 0.

        self._reset_dof_actor_ids = (2 * env_ids).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self._reset_dof_actor_ids), len(self._reset_dof_actor_ids))


        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # self.root_states[env_ids, 1:2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        # self.root_states[env_ids, 2:3] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        # base velocities
        vel_rng = getattr(self.cfg.domain_rand, "init_base_vel_range", [-0.3, 0.3])
        if self.play:
            self.root_states[env_ids, 7:13] = 0.
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(vel_rng[0], vel_rng[1], (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        ball_ids = env_ids
        self.ball_states[ball_ids] = self.base_init_state

        ball_vel = self.assign_ball_states(ball_ids)

        self.ball_states[ball_ids, :3] = self.ball_start[ball_ids, :]
        self.prev_ball_pos[ball_ids] = self.ball_states[ball_ids, :3]

        self.ball_vel[ball_ids] = ball_vel[:,0]
        self.has_in_air[ball_ids] = False
        self.static_obs[ball_ids] *= 0. 
        self.stop_flag[ball_ids] = 0.
        self.success_flag[ball_ids] = 0.
        self.goal_conceded_flag[ball_ids] = 0.
        self.arm_save_flag[ball_ids] = 0.
        self.recovery_stable_steps[ball_ids] = 0
        if hasattr(self, 'k1_safe_steps'):
            self.k1_safe_steps[ball_ids] = 0
        if hasattr(self, 'k1_hold_pose_locked'):
            self.k1_hold_pose_locked[ball_ids] = False
        self.substep_max_ball_force[ball_ids] = 0.
        self.substep_min_arm_dist[ball_ids] = 10.
        self.substep_min_robot_dist[ball_ids] = 10.
        if hasattr(self, 'ball_force_tensor'):
            self.ball_force_tensor[ball_ids] = 0.
        self.dist[ball_ids] = 5.0
        
        self.ball_last[ball_ids] = 0.
        if self.cfg.asset.name != "k1":
            self.vanish_step[ball_ids] = torch.randint(low=0, high=30, size=(len(ball_ids),), device=self.device)
        else:
            self.vanish_step[ball_ids] = torch.randint(low=0, high=4, size=(len(ball_ids),), device=self.device)

        # move towards the robot
        self.ball_states[ball_ids, 7:13] *= 0. 
        self.ball_states[ball_ids, 7:10] = ball_vel
        self.prev_ball_lin_vel[ball_ids] = ball_vel
        
        self._reset_root_actor_ids = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(self._reset_root_actor_ids), len(self._reset_root_actor_ids))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        if self.cfg.asset.name != "k1":
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
            all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))
        else:
            # Dorongan asinkron per-robot hanya setelah bola lewat agar grafik tidak gergaji (sawtooth)
            behind_ball = (self.ball_states[:, 0] - self.env_origins[:, 0]) < -0.1
            push_prob = 1.0 / max(1.0, float(self.cfg.domain_rand.push_interval))
            push_ids = ((torch.rand(self.num_envs, device=self.device) < push_prob) & behind_ball).nonzero(as_tuple=False).flatten()
            if len(push_ids) > 0:
                self.root_states[push_ids, 7:9] += torch_rand_float(-max_vel, max_vel, (len(push_ids), 2), device=self.device)
                all_states = torch.cat((self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
                actor_ids = (2 * push_ids).to(dtype=torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(all_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

    def _randomize_balls(self):
        """ Applies aerodynamic drag force (and optional perturbation) to balls. 
        """
        radius = self.cfg.goal.ball_radius if self.cfg.asset.name == "k1" else 0.1
        airforce = self.compute_drag_force(radius=radius)
        if self.cfg.asset.name != "k1":
            noise = torch.empty_like(airforce).uniform_(-0.5, 0.5)
        else:
            noise = torch.empty_like(airforce).uniform_(-0.05, 0.05)
        if not self.play:
            airforce += noise * (self.stop_flag == 0).unsqueeze(-1)
        self.ball_force_tensor[:, -1, :3] = airforce
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ball_force_tensor))

        if self.cfg.asset.name != "k1" and (self.common_step_counter % self.cfg.domain_rand.ball_interval == 0):
            max_vel = self.cfg.domain_rand.max_ball_vel
            self.ball_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
            all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))



    def compute_drag_force(self, rho=1.225, drag_coeff=0.47, radius=0.1):
        velocity = self.ball_states[:, 7:10]
        cross_area = torch.pi * (radius ** 2)
        speed = torch.norm(velocity, dim=1, keepdim=True)  # Magnitude of velocity
        return -0.5 * rho * drag_coeff * cross_area * speed * velocity


    def assign_ball_states(self, ball_ids, g=9.81):

        dtype = torch.float
        device = self.ball_start.device
        shot = getattr(self, 'evaluation_shot', None)
        if shot is not None:
            if self.cfg.asset.name != "k1" or self.num_envs != 1:
                raise ValueError("Fixed shots require K1 without AMP and num_envs=1")
            self.catchstep[ball_ids] = 50
            start = torch.tensor([shot['start_x_m'], shot['start_y_m'], shot['start_z_m']], device=device)
            target = torch.tensor([shot['target_x_m'], shot['target_y_m'], shot['target_z_m']], device=device)
            self.ball_start[ball_ids] = start + self.env_origins[ball_ids]
            self.ball_end[ball_ids] = target + self.env_origins[ball_ids]
            self.end_target[ball_ids] = self.ball_end[ball_ids]
            right = shot['target_y_m'] < 0.0
            self.end_regions[ball_ids] = ((3 if right else 2) if shot['target_z_m'] > .72
                                         else (5 if right else 4) if shot['target_z_m'] <= .30
                                         else (1 if right else 0))
            return torch.tensor([[shot['vx_m_s'], shot['vy_m_s'], shot['vz_m_s']]], device=device).repeat(len(ball_ids), 1)


        self.catchstep[ball_ids] = 50 * torch.ones(len(ball_ids), dtype = torch.int, device = self.device)

        if self.cfg.asset.name == "k1":
            n = len(ball_ids)
            custom_reg = getattr(self.cfg.commands, 'custom_region', None)
            if custom_reg is not None:
                reg_idx = max(0, min(5, int(custom_reg)))
                self.end_regions[ball_ids] = reg_idx
                reg_cfg = getattr(self.cfg.commands, f"ranges_{reg_idx}")
                w = reg_cfg.evalw if self.play else reg_cfg.width
                h = reg_cfg.evalh if self.play else reg_cfg.height
                self.command_ranges[ball_ids, 0] = w[0]
                self.command_ranges[ball_ids, 1] = w[1]
                self.command_ranges[ball_ids, 2] = h[0]
                self.command_ranges[ball_ids, 3] = h[1]
            elif self.play:
                for idx in ball_ids:
                    i = int(idx.item())
                    reg_idx = i % 6 if self.num_envs > 1 else int(torch.randint(0, 6, (1,), device=device).item())
                    self.end_regions[i] = reg_idx
                    reg_cfg = getattr(self.cfg.commands, f"ranges_{reg_idx}")
                    self.command_ranges[i, 0] = reg_cfg.evalw[0]
                    self.command_ranges[i, 1] = reg_cfg.evalw[1]
                    self.command_ranges[i, 2] = reg_cfg.evalh[0]
                    self.command_ranges[i, 3] = reg_cfg.evalh[1]

            sx = getattr(self.cfg.commands, 'custom_start_x', None)
            sy = getattr(self.cfg.commands, 'custom_start_y', None)
            speed_range = getattr(self.cfg.commands, 'train_speed_range', None)
            fast_ratio = float(getattr(self.cfg.commands, 'train_fast_ratio', 1.0))
            use_fast_train = (speed_range is not None) and (not self.play) and (sx is None) and (getattr(self.cfg.commands, 'custom_ball_speed', None) is None) and (getattr(self.cfg.commands, 'custom_flight_time', None) is None)
            std_start_x = torch.full((n,), float(sx), dtype=dtype, device=device) if sx is not None else (1.4 * torch.rand(n, dtype=dtype, device=device) + 3.0)
            if use_fast_train:
                fast_mask = torch.rand((n, 1), dtype=dtype, device=device) < fast_ratio
                s_lo, s_hi = float(speed_range[0]), float(speed_range[1])
                speed_target = s_lo + (s_hi - s_lo) * torch.rand((n, 1), dtype=dtype, device=device)
                t_fast = 0.42 + 0.14 * torch.rand((n, 1), dtype=dtype, device=device)
                fast_x = torch.clamp(0.08 + (speed_target * t_fast) / 1.04, min=3.6, max=7.2).squeeze(-1)
                start_x_vec = torch.where(fast_mask.squeeze(-1), fast_x, std_start_x)
            else:
                fast_mask = None
                speed_target = None
                start_x_vec = std_start_x
            ball_start_local = torch.stack([
                start_x_vec,
                torch.full((n,), float(sy), dtype=dtype, device=device) if sy is not None else (1.8 * torch.rand(n, dtype=dtype, device=device) - 0.9),
                0.15 * torch.rand(n, dtype=dtype, device=device) + 0.11,
            ], dim=1)

            goal_y = torch.rand(n, dtype=dtype, device=device) * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0]) + self.command_ranges[ball_ids, 0]
            goal_z = torch.rand(n, dtype=dtype, device=device) * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2]) + self.command_ranges[ball_ids, 2]

            if use_fast_train:
                kolong_ratio = float(getattr(self.cfg.commands, 'train_kolong_ratio', 0.0))
                if kolong_ratio > 0.0:
                    low_std = (~fast_mask.squeeze(-1)) & ((self.end_regions[ball_ids] == 4) | (self.end_regions[ball_ids] == 5))
                    kol_mask = low_std & (torch.rand(n, dtype=dtype, device=device) < kolong_ratio)
                    if kol_mask.any():
                        sign_l = torch.where(self.end_regions[ball_ids] == 4, 1.0, -1.0)
                        kol_y = sign_l * (0.01 + 0.23 * torch.rand(n, dtype=dtype, device=device))
                        kol_z = 0.12 + 0.12 * torch.rand(n, dtype=dtype, device=device)
                        goal_y = torch.where(kol_mask, kol_y, goal_y)
                        goal_z = torch.where(kol_mask, kol_z, goal_z)

            custom_y = getattr(self.cfg.commands, 'custom_target_y', None)
            custom_z = getattr(self.cfg.commands, 'custom_target_z', None)
            if custom_y is not None:
                goal_y = torch.full((n,), float(custom_y), dtype=dtype, device=device)
            if custom_z is not None:
                goal_z = torch.full((n,), float(custom_z), dtype=dtype, device=device)
            if (custom_y is not None or custom_z is not None) and custom_reg is None:
                is_right = goal_y < 0.0
                is_high = goal_z > 0.72
                is_low = goal_z <= 0.30
                inferred_reg = torch.where(
                    is_high,
                    torch.where(is_right, 3, 2),
                    torch.where(is_low, torch.where(is_right, 5, 4), torch.where(is_right, 1, 0)),
                )
                self.end_regions[ball_ids] = inferred_reg.to(dtype=torch.long)

            ball_goal_local = torch.stack([
                torch.full((n,), 0.08, dtype=dtype, device=device),
                goal_y,
                goal_z,
            ], dim=1)

            self.ball_start[ball_ids, :] = ball_start_local + self.env_origins[ball_ids]
            self.ball_start[ball_ids, 2] = ball_start_local[:, 2]

            self.ball_end[ball_ids, :] = ball_goal_local + self.env_origins[ball_ids]
            self.ball_end[ball_ids, 2] = ball_goal_local[:, 2]

            self.end_target[ball_ids, :] = self.ball_end[ball_ids, :].clone()

            delta_pos = self.ball_end[ball_ids, :] - self.ball_start[ball_ids, :]

            custom_speed = getattr(self.cfg.commands, 'custom_ball_speed', None)
            if custom_speed is not None:
                return launch_at_speed(delta_pos, float(custom_speed), g)

            custom_t = getattr(self.cfg.commands, 'custom_flight_time', None)
            if custom_t is not None:
                t_flight = torch.full((n, 1), float(custom_t), dtype=dtype, device=device)
            else:
                t_std = 0.65 + 0.45 * torch.rand((n, 1), dtype=dtype, device=device)
                is_low = (self.end_regions[ball_ids] == 4) | (self.end_regions[ball_ids] == 5)
                if is_low.any():
                    n_low = int(is_low.sum().item())
                    t_std[is_low] = 0.58 + 0.28 * torch.rand((n_low, 1), dtype=dtype, device=device)
                if fast_mask is not None:
                    xy_dist = torch.norm(delta_pos[:, 0:2], dim=-1, keepdim=True)
                    t_fast_exact = torch.clamp((xy_dist * 1.04) / speed_target, min=0.40, max=0.58)
                    t_flight = torch.where(fast_mask, t_fast_exact, t_std)
                else:
                    t_flight = t_std

            ball_vel = torch.empty_like(delta_pos)
            ball_vel[:, 0:2] = (delta_pos[:, 0:2] / t_flight) * 1.04
            ball_vel[:, 2:3] = (delta_pos[:, 2:3] + 0.5 * g * (t_flight ** 2)) / t_flight
            return ball_vel
                
        ball_start_local = torch.stack([
            2.0 * torch.rand(len(ball_ids), dtype=dtype, device=device) + 3.0,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[1] - self.init_ranges[0]) + self.init_ranges[0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[3] - self.init_ranges[2]) + self.init_ranges[2]
        ], dim=1)

        ball_end_local = torch.stack([
            -0.5 * torch.rand(len(ball_ids), dtype=dtype, device=device) - 0.1,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0]) + self.command_ranges[ball_ids, 0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2]) + self.command_ranges[ball_ids, 2]
        ], dim=1)

        # in world frame
        self.ball_start[ball_ids,:] = ball_start_local + self.env_origins[ball_ids]  # convert the local target cord in world frame 
        self.ball_start[ball_ids, 2] = ball_start_local[:, 2]

        self.ball_end[ball_ids,:] = ball_end_local + self.env_origins[ball_ids] # convert the local target cord in world frame 
        self.ball_end[ball_ids, 2] = ball_end_local[:, 2]

        catch_prop = (0.1 - ball_start_local[:,0:1]) / (ball_end_local[:,0:1] - ball_start_local[:,0:1])

        # Compute velocity
        delta_pos = self.ball_end[ball_ids,:] - self.ball_start[ball_ids,:]


        t_flight = 0.4 + 0.6 * torch.rand((1), dtype=dtype, device=device)

        if self.play:
            t_flight = 0.5 + 0.5 * torch.rand((1), dtype=dtype, device=device)     

        self.end_target[ball_ids,:] = self.ball_start[ball_ids,:] + delta_pos * catch_prop

        ball_vel = torch.empty_like(delta_pos)

        ball_vel[:, 0:2] = delta_pos[:, 0:2] / t_flight
        ball_vel[:, 2] = (delta_pos[:, 2] + 0.5 * g * t_flight**2) / t_flight

        return ball_vel



    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\

        noise_vec = torch.zeros( self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:self.num_ballobs] = noise_scales.ball * noise_level
        noise_vec[self.num_ballobs:self.num_ballobs + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[self.num_ballobs + 3: self.num_ballobs + 6] = noise_scales.gravity * noise_level
        noise_vec[self.num_ballobs + 6:(self.num_ballobs + 6 + self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(self.num_ballobs + 6 + self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(self.num_ballobs + 6 + 2 * self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)] = 0. # previous actions
        return noise_vec




    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        all_states = self.all_root_states.view(self.num_envs, 2, 13)
        self.root_states, self.ball_states = all_states[:, 0, :], all_states[:, 1, :]

    

        all_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_states = all_body_states[:, :-1, :]

        all_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies + 1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = all_contact_forces[:, :-1, :]

        self.ball_contact_forces = all_contact_forces[:, -1:, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.last_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3].clone()

        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.ball_force_tensor = torch.zeros([self.num_envs, self.num_bodies + 1, 3], device=self.device)
        self.end_target = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)

        custom_reg = getattr(self.cfg.commands, 'custom_region', None)
        if custom_reg is not None:
            self.end_regions = torch.full((self.num_envs,), int(custom_reg), dtype=torch.long, device=self.device)
        elif self.num_envs % 6 == 0:
            six = self.num_envs // 6
            self.end_regions = torch.cat([
                torch.zeros(six, dtype=torch.long, device = self.device),
                torch.ones(six, dtype=torch.long, device = self.device),
                torch.full((six,), 2, dtype=torch.long, device = self.device),
                torch.full((six,), 3, dtype=torch.long, device = self.device),
                torch.full((six,), 4, dtype=torch.long, device = self.device),
                torch.full((six,), 5, dtype=torch.long, device = self.device)
            ])
        else:
            self.end_regions = torch.remainder(torch.arange(self.num_envs, device=self.device), 6)
    
        command_dict = class_to_dict(self.cfg.commands)

        # Initialize an empty tensor for command ranges
        num_envs = len(self.end_regions)
        self.command_ranges = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.command_bound  = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.init_ranges  = torch.zeros((4), dtype=torch.float32, device=self.device)
        # For each environment, set the appropriate ranges based on end_region
        for env_idx in range(num_envs):
            region = self.end_regions[env_idx].item()  # Get the region (0, 1, 2, or 3)
            region_key = f"ranges_{region}"
            
            # Get the ranges for this region
            region_ranges = command_dict[region_key]
            
            # Assign height and width ranges
            self.command_ranges[env_idx, 0] = region_ranges["width"][0]   # width_0
            self.command_ranges[env_idx, 1] = region_ranges["width"][1]   # width_1
            self.command_ranges[env_idx, 2] = region_ranges["height"][0]  # height_0
            self.command_ranges[env_idx, 3] = region_ranges["height"][1]  # height_1

            if self.play:
                self.command_ranges[env_idx, 0] = region_ranges["evalw"][0]   # width_0
                self.command_ranges[env_idx, 1] = region_ranges["evalw"][1]   # width_1
                self.command_ranges[env_idx, 2] = region_ranges["evalh"][0]  # height_0
                self.command_ranges[env_idx, 3] = region_ranges["evalh"][1]  # height_1



            self.command_bound[env_idx, 0] = region_ranges["maxw"][0]   # width_0
            self.command_bound[env_idx, 1] = region_ranges["maxw"][1]   # width_1
            self.command_bound[env_idx, 2] = region_ranges["maxh"][0]  # height_0
            self.command_bound[env_idx, 3] = region_ranges["maxh"][1]  # height_1
        
        self.command_init_ranges = self.command_ranges.clone()
        self.init_ranges[0] = command_dict["ranges_1"]["maxw"][0]
        self.init_ranges[1] = command_dict["ranges_0"]["maxw"][1]
        self.init_ranges[2] = command_dict["ranges_4"]["maxh"][0]
        self.init_ranges[3] = command_dict["ranges_2"]["maxh"][1]




        self.catchstep = 50 * torch.ones(self.num_envs, dtype = torch.int, device= self.device)
        self.dist = 5 * torch.ones(self.num_envs, dtype = torch.float, device= self.device)
        self.ball_vel = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.has_in_air = torch.zeros(self.num_envs, dtype = torch.bool, device= self.device)
        self.static_obs = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.stop_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.goal_conceded_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.arm_save_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.recovery_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.k1_safe_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.k1_hold_pose_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.k1_hold_upper_dof_pos = None
        self.substep_max_ball_force = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.substep_min_arm_dist = torch.full((self.num_envs,), 10.0, dtype = torch.float, device= self.device)
        self.substep_min_robot_dist = torch.full((self.num_envs,), 10.0, dtype = torch.float, device= self.device)
        self.prev_ball_lin_vel = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.prev_ball_pos = self.ball_states[:, :3].clone()
        self.success_rate = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)


        self.ball_start = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.ball_end = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)        
        self.ball_last = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)   
        if self.cfg.asset.name != "k1":
            self.vanish_step = torch.randint(low=0, high=30, size=(self.num_envs,), device=self.device)
            self.startstep = 50 - random.randint(3, 10)
        else:
            self.vanish_step = torch.randint(low=0, high=4, size=(self.num_envs,), device=self.device)
            self.startstep = 49
        self.sr = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)  


        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # self.standpos = self.default_dof_pos
        self.standpos = torch.tensor([self.cfg.init_state.init_pos], dtype=torch.float32, device=self.default_dof_pos.device)
        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)
        self.init_dof_pos = self.default_dof_poses.clone()

        self.action_scale_vec = torch.full((1, self.num_dof), float(self.cfg.control.action_scale), dtype=torch.float, device=self.device)
        if hasattr(self.cfg.control, "arm_action_scale") and len(self.arm_joint_indices) > 0:
            self.action_scale_vec[0, self.arm_joint_indices] = float(self.cfg.control.arm_action_scale)
        if hasattr(self.cfg.control, "leg_action_scale") and len(self.leg_joint_indices) > 0:
            self.action_scale_vec[0, self.leg_joint_indices] = float(self.cfg.control.leg_action_scale)
        if hasattr(self.cfg.control, "head_action_scale") and len(self.head_joint_indices) > 0:
            self.action_scale_vec[0, self.head_joint_indices] = float(self.cfg.control.head_action_scale)



        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


        # create mocap dataset
        self.init_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_pos_xy[:] = self.base_init_state[:2] + self.env_origins[:, 0:2]
        self.init_base_quat[:] = self.base_init_state[3:7]



        self.motions = {}
        self.motion_ids = {}
        self.motion_time = {}
        self.motion_dict = {}
        amp_joint_names = getattr(self.cfg.amp, "joint_names", None)
        if amp_joint_names is not None:
            self.amp_joint_indices = torch.tensor(
                [self.dof_names.index(n) for n in amp_joint_names],
                dtype=torch.long, device=self.device,
            )
            motion_dof_names = list(amp_joint_names)
        else:
            self.amp_joint_indices = None
            motion_dof_names = self.dof_names
        if self.cfg.amp.amp_coef > 0:
            multidataset, mapping = load_imitation_dataset(self.cfg.dataset.folder.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),self.cfg.dataset.joint_mapping.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
            for key in multidataset.keys():
                # Here 'key' will be the dataset's key name, and 'dataset' is the actual data

                # Initialize the MotionLib class for the given dataset
                self.motions[key] = MotionLib(multidataset[key], mapping, motion_dof_names, self.keyframe_names,
                                            self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, self.device,
                                            self.amp_obs_type)
            
    

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue

            if name == "eereach":
                self.eereach_init =  self.reward_scales["eereach"]
            if name == "success":
                self.success_init = self.reward_scales["success"]
            if name == "stopball":
                self.stop_init = self.reward_scales["stopball"]
       

            if name == "torque_limits":
                self.torque_init = self.reward_scales["torque_limits"]
            if name == "dof_pos_limits":
                self.dof_pos_init = self.reward_scales["dof_pos_limits"]

            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.ball_gravity = self.cfg.env.ball_gravity
        self.num_ballobs = self.cfg.env.num_ballobs
        self.play = self.cfg.env.play

        ### load ball ###
        ball_options = gymapi.AssetOptions()
        ball_options.disable_gravity = False
        ball_options.armature = 0.0
        ball_options.linear_damping = 0.0
        ball_options.angular_damping = 0.0
        ball_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        ball_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        ball_path = self.cfg.asset.ballfile.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ball_root = os.path.dirname(ball_path)
        ball_file = os.path.basename(ball_path)
        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, ball_options)

    

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        hand_names  = [s for s in body_names if self.cfg.asset.hand_name in s]

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.ball_handles = []
        self.envs = []
        self.curriculumupdate = 0.
        self.curriculumsigma = self.cfg.rewards.catch_sigma
    
        self.task_scale = 0.
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.3, 0.3, (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(actor_handle)

            ballpos = pos
            ballpos[:0] += torch_rand_float(0.5, 1.0, (1,1), device=self.device).squeeze(1)
            ballpos[:1] += torch_rand_float(-0.5, 0.5, (1,1), device=self.device).squeeze(1)
            ballpos[2] = 1.5
            
            start_pose.p = gymapi.Vec3(*ballpos)
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, 0, 1)
            c = 0.5 + 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_handles.append(ball_handle)

            self.envs.append(env_handle)



        self.left_hip_joint_indices = torch.zeros(len(self.cfg.control.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.control.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_names[i])
        head_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "aahead_yaw_link")
        self.head_body_index = head_handle if head_handle >= 0 else None

        self.camera_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "d435_link")
        if self.camera_index == -1:
            self.camera_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "head_realsense_rgb_link")
        
        contact_feet_names = [s for s in body_names if self.cfg.asset.contact_foot_names in s]
        
        self.contact_feet_indices = torch.zeros(len(contact_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(contact_feet_names)):
            self.contact_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], contact_feet_names[i])


        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
        self.curriculum_dof_indices = torch.zeros(len(self.cfg.control.curriculum_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.curriculum_joints)):
            self.curriculum_dof_indices[i] = self.dof_names.index(self.cfg.control.curriculum_joints[i])
            
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.control.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.control.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
            
        self.left_arm_joint_indices = torch.zeros(len(self.cfg.control.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.control.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.right_arm_joints[i])
            
        self.elbow_joint_indices = torch.zeros(len(self.cfg.control.elbow_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.elbow_joints)):
            self.elbow_joint_indices[i] = self.dof_names.index(self.cfg.control.elbow_joints[i])

        self.wrist_joint_indices = torch.zeros(len(self.cfg.control.wrist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.wrist_joints)):
            self.wrist_joint_indices[i] = self.dof_names.index(self.cfg.control.wrist_joints[i])



        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])


        head_joint_names = getattr(self.cfg.control, "head_joints", [])
        self.head_joint_indices = torch.tensor(
            [self.dof_names.index(n) for n in head_joint_names if n in self.dof_names],
            dtype=torch.long, device=self.device
        )

        if self.cfg.asset.name != "k1":
            self.upper_body_joint_indices = torch.cat([self.elbow_joint_indices, self.wrist_joint_indices])
        else:
            self.upper_body_joint_indices = torch.cat([self.arm_joint_indices, self.wrist_joint_indices, self.head_joint_indices])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.torso_link)
            
        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        arm_bnames = [s for s in body_names if any(k in s for k in ["shoulder", "elbow", "wrist", "hand"])]
        self.arm_body_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in arm_bnames],
            dtype=torch.long, device=self.device
        )
        forearm_bnames = [s for s in body_names if "elbow_yaw" in s]
        self.forearm_body_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in forearm_bnames],
            dtype=torch.long, device=self.device
        )
        other_bnames = [s for s in body_names if not any(k in s for k in ["shoulder", "elbow", "wrist", "hand", "rgb_link"])]
        self.other_save_body_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in other_bnames],
            dtype=torch.long, device=self.device
        )

    def _compute_k1_ball_dists(self):
        ball_p = self.ball_states[:, :3].unsqueeze(1)
        arm_pts = self.rigid_body_states[:, self.arm_body_indices, :3]
        if len(self.forearm_body_indices) == 2 and len(self.hand_indices) == 2:
            mid_pts = 0.5 * (self.rigid_body_states[:, self.forearm_body_indices, :3] + self.rigid_body_states[:, self.hand_indices, :3])
            arm_pts = torch.cat([arm_pts, mid_pts], dim=1)
        min_arm_dist = torch.norm(arm_pts - ball_p, dim=-1).min(dim=-1).values
        other_pts = self.rigid_body_states[:, self.other_save_body_indices, :3]
        min_other_dist = torch.norm(other_pts - ball_p, dim=-1).min(dim=-1).values
        min_robot_dist = torch.minimum(min_arm_dist, min_other_dist)
        return min_arm_dist, min_robot_dist

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """

        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(-num_rows//2, num_rows//2), torch.arange(-num_cols//2, num_cols//2))
    
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.ball_interval = np.ceil(self.cfg.domain_rand.ball_interval_s / self.dt)

        
    def _get_base_heights(self, env_ids=None):

        return self.root_states[:, 2].clone()


    #------------ reward functions----------------


    def _k1_arm_excess_z(self):
        """Return active/off body-collision intrusion and active hand height/y (overhead hands allowed if collision-free)."""
        head_idx = getattr(self, 'head_body_index', None)
        if head_idx is not None:
            head_pos = self.rigid_body_states[:, head_idx, :3]
        else:
            head_pos = self.torso_pos + torch.tensor([0.0, 0.0, 0.215], device=self.device)
        hand_indices = getattr(self, 'hand_indices', None)
        is_high = (self.end_regions == 2) | (self.end_regions == 3)
        if hand_indices is None or len(hand_indices) != 2:
            zeros = torch.zeros_like(self.dist)
            return zeros, zeros, zeros, self.torso_pos[:, 2], zeros, is_high
        hand_pos = self.rigid_body_states[:, hand_indices, :3]
        left_side = (self.end_regions % 2) == 0
        act_hand = torch.where(left_side.unsqueeze(-1), hand_pos[:, 0, :], hand_pos[:, 1, :])
        off_hand = torch.where(left_side.unsqueeze(-1), hand_pos[:, 1, :], hand_pos[:, 0, :])
        act_hand_z = act_hand[:, 2]
        act_hand_y = act_hand[:, 1] - self.env_origins[:, 1]

        # Hands are allowed above the head as long as they maintain safe clearance from head, torso column, and each other.
        act_head_pen = torch.clamp(0.18 - torch.norm(act_hand - head_pos, dim=-1), min=0.0)
        off_head_pen = torch.clamp(0.18 - torch.norm(off_hand - head_pos, dim=-1), min=0.0)
        act_in_body_z = ((act_hand[:, 2] >= self.torso_pos[:, 2] - 0.14) & (act_hand[:, 2] <= head_pos[:, 2] + 0.08)).float()
        off_in_body_z = ((off_hand[:, 2] >= self.torso_pos[:, 2] - 0.14) & (off_hand[:, 2] <= head_pos[:, 2] + 0.08)).float()
        act_torso_pen = torch.clamp(0.16 - torch.norm(act_hand[:, :2] - self.torso_pos[:, :2], dim=-1), min=0.0) * act_in_body_z
        off_torso_pen = torch.clamp(0.16 - torch.norm(off_hand[:, :2] - self.torso_pos[:, :2], dim=-1), min=0.0) * off_in_body_z
        hand_pair_pen = torch.clamp(0.20 - torch.norm(hand_pos[:, 0, :] - hand_pos[:, 1, :], dim=-1), min=0.0)

        act_over = act_head_pen + act_torso_pen + 0.5 * hand_pair_pen
        off_over = off_head_pen + off_torso_pen + 0.5 * hand_pair_pen
        off_roll_over = torch.zeros_like(self.dist)
        return act_over, off_over, off_roll_over, act_hand_z, act_hand_y, is_high

    def _k1_kolong_state(self):
        """Return between-legs defense metrics (low, center_low, straddle, gap, foot_err, track_y)."""
        feet_indices = getattr(self, 'contact_feet_indices', None)
        is_low = (self.end_regions == 4) | (self.end_regions == 5)
        shot_target_z = getattr(self, 'ball_end', self.end_target)[:, 2]
        low_ball = is_low | (shot_target_z <= 0.36)
        ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
        ball_local_y = self.ball_states[:, 1] - self.env_origins[:, 1]
        end_y_local = self.end_target[:, 1] - self.env_origins[:, 1]
        track_y = torch.where(ball_local_x < 1.8, ball_local_y, end_y_local)
        center_low = low_ball & (torch.abs(end_y_local) < 0.30)
        if feet_indices is None or len(feet_indices) != 2:
            zeros = torch.zeros_like(self.dist)
            return low_ball, center_low, torch.zeros_like(low_ball), zeros, zeros, track_y
        foot_l_y = self.rigid_body_states[:, feet_indices[0], 1] - self.env_origins[:, 1]
        foot_r_y = self.rigid_body_states[:, feet_indices[1], 1] - self.env_origins[:, 1]
        foot_lo = torch.minimum(foot_l_y, foot_r_y)
        foot_hi = torch.maximum(foot_l_y, foot_r_y)
        straddling = low_ball & (track_y >= foot_lo - 0.04) & (track_y <= foot_hi + 0.04) & (ball_local_x < 2.6)
        ankle_gap = torch.abs(foot_l_y - foot_r_y)
        nearest_foot_err = torch.minimum(torch.abs(foot_l_y - track_y), torch.abs(foot_r_y - track_y))
        return low_ball, center_low, straddling, ankle_gap, nearest_foot_err, track_y

    def _reward_arm_overhead(self):
        if self.cfg.asset.name != "k1":
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
        live = (ball_local_x > -0.02) & (~self._k1_ball_safe()) & (self.goal_conceded_flag == 0)
        act_over, off_over, _, _, _, _ = self._k1_arm_excess_z()
        close_pen = 4.0 * (off_over + act_over)
        tuck_pen = torch.zeros_like(act_over)
        hand_indices = getattr(self, 'hand_indices', None)
        if hand_indices is not None and len(hand_indices) == 2:
            hand_pos = self.rigid_body_states[:, hand_indices, :3]
            hand_gap_y = torch.abs(hand_pos[:, 0, 1] - hand_pos[:, 1, 1])
            hand_dist = torch.norm(hand_pos[:, 0, :] - hand_pos[:, 1, :], dim=-1)
            hand_min_z = torch.minimum(hand_pos[:, 0, 2], hand_pos[:, 1, 2])
            early = ball_local_x >= 2.0
            is_low = (self.end_regions == 4) | (self.end_regions == 5)
            early_tuck = (3.5 * torch.clamp(0.72 - hand_gap_y, min=0.0) + 2.5 * torch.clamp(0.46 - hand_min_z, min=0.0) * (~is_low).float()) * early.float()
            clap_pen = 2.5 * torch.clamp(0.24 - hand_dist, min=0.0) * (~early).float()
            tuck_pen = early_tuck + clap_pen
        return torch.clamp(close_pen + tuck_pen, max=4.5) * live.float()

    def _reward_kolong_gap(self):
        if self.cfg.asset.name != "k1":
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
        active = (ball_local_x > -0.02) & (~self._k1_ball_safe()) & (self.goal_conceded_flag == 0)
        _, center_low, straddling, ankle_gap, nearest_foot_err, _ = self._k1_kolong_state()
        foot_clear_err = torch.clamp(nearest_foot_err - 0.06, min=0.0, max=0.40) * straddling.float()
        split_excess = torch.clamp(ankle_gap - 0.22, min=0.0, max=0.45)
        center_split_err = (2.2 * split_excess + 5.0 * torch.square(split_excess)) * (center_low & (ball_local_x < 3.6)).float()
        return torch.clamp(1.8 * foot_clear_err + center_split_err, max=2.2) * active.float()

    def _reward_eereach(self):
        if self.cfg.asset.name == "k1":
            ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
            active = (ball_local_x > -0.02) & (~self._k1_ball_safe()) & (self.goal_conceded_flag == 0)
            is_low = (self.end_regions == 4) | (self.end_regions == 5)
            is_high = (self.end_regions == 2) | (self.end_regions == 3)
            end_y_local = self.end_target[:, 1] - self.env_origins[:, 1]
            kolong_fn = getattr(self, '_k1_kolong_state', None)
            if kolong_fn is not None:
                low_ball, center_low, straddling, ankle_gap, nearest_foot_err, track_y = kolong_fn()
            else:
                low_ball = is_low | (self.ball_states[:, 2] < 0.38)
                center_low = low_ball & (torch.abs(end_y_local) < 0.30)
                straddling = torch.zeros_like(center_low)
                ankle_gap = torch.zeros_like(self.dist)
                nearest_foot_err = torch.zeros_like(self.dist)
                track_y = end_y_local

            # Shift torso (~0.20 m = natural stance half-width) on center_low shots so the near foot directly blocks ball_y.
            kolong_offset = torch.where(end_y_local >= 0.0, 0.20, -0.20)
            target_y = torch.where(center_low, end_y_local - kolong_offset, torch.clamp(end_y_local, -1.25, 1.25) * 0.85)
            curr_y = self.torso_pos[:, 1] - self.env_origins[:, 1]
            lat_rew = torch.exp(-4.0 * torch.square(curr_y - target_y))
            dir_y = torch.sign(target_y - curr_y)
            lat_vel = self.rigid_body_states[:, self.upper_body_index, 8]
            lat_vel_bonus = 0.85 * torch.clamp(lat_vel * dir_y, 0.0, 2.2) * (torch.abs(target_y - curr_y) > 0.08).float()

            target_z = torch.full((self.num_envs,), self.cfg.init_state.pos[2], dtype=torch.float, device=self.device)
            target_z[is_low] = 0.40
            target_z[is_high] = 0.72
            jump_prep = is_high & (ball_local_x < 4.0)
            vz_up = self.rigid_body_states[:, self.upper_body_index, 9]
            # Allow countermovement dip (>=0.42 m) while loading legs for an approaching high shot.
            high_z_err = torch.where(
                jump_prep & (ball_local_x > 2.2) & (self.torso_pos[:, 2] >= 0.42) & (vz_up <= 0.15),
                torch.zeros_like(self.torso_pos[:, 2]),
                self.torso_pos[:, 2] - target_z,
            )
            vert_rew = torch.where(
                is_high,
                torch.exp(-8.0 * torch.square(high_z_err)),
                torch.exp(-8.0 * torch.square(self.torso_pos[:, 2] - target_z)),
            )
            vert_weight = torch.where(
                jump_prep & ((self.torso_pos[:, 2] > 0.56) | (vz_up > 0.15) | (ball_local_x < 2.2)),
                0.34,
                torch.where(is_low & (ball_local_x < 3.0), 0.24, 0.10),
            )
            jump_vel_bonus = 1.25 * torch.clamp(vz_up, 0.0, 2.5) * (jump_prep & (ball_local_x > 0.15)).float()

            hand_reach = 0.50 * torch.exp(-2.0 * self.dist) + 0.50 * torch.exp(-10.0 * torch.square(self.dist))

            excess_fn = getattr(self, '_k1_arm_excess_z', None)
            if excess_fn is not None:
                act_over, off_over, _, act_hand_z, act_hand_y, _ = excess_fn()
            else:
                zeros = torch.zeros_like(self.dist)
                act_over, off_over, act_hand_z, act_hand_y = zeros, zeros, self.torso_pos[:, 2], zeros
            act_style = torch.exp(-12.0 * torch.square(act_over))
            off_low = 1.00 * torch.exp(-12.0 * torch.square(off_over))
            act_z_match = 1.10 * torch.exp(-10.0 * torch.square(act_hand_z - torch.clamp(self.end_target[:, 2], 0.24, 1.18)))

            min_arm_dist = getattr(self, 'cur_min_arm_dist', None)
            min_robot_dist = getattr(self, 'cur_min_robot_dist', None)
            if min_arm_dist is None or min_robot_dist is None:
                min_arm_dist, min_robot_dist = self._compute_k1_ball_dists()
            shot_target_z = getattr(self, 'ball_end', self.end_target)[:, 2]
            block_dist = torch.where(low_ball | (shot_target_z <= 0.50), min_robot_dist, min_arm_dist)
            live_track = 0.50 * torch.exp(-4.5 * block_dist) * (ball_local_x < 1.3).float()

            fivehole_rew = torch.zeros_like(self.dist)
            center_gate = torch.ones_like(self.dist)
            feet_indices = getattr(self, 'contact_feet_indices', None)
            if feet_indices is not None and len(feet_indices) == 2:
                foot_y = self.rigid_body_states[:, feet_indices, 1] - self.env_origins[:, 1:2]
                foot_z = self.rigid_body_states[:, feet_indices, 2] - self.env_origins[:, 2:3]
                near_idx = torch.argmin(torch.abs(foot_y - track_y.unsqueeze(1)), dim=1)
                near_foot_z = foot_z.gather(1, near_idx.unsqueeze(1)).squeeze(1)
                foot_low_cap = torch.clamp(shot_target_z * 0.75, min=0.10, max=0.22)
                foot_low_z = torch.exp(-45.0 * torch.square(torch.clamp(near_foot_z - foot_low_cap, min=0.0)))
                leg_block = torch.exp(-20.0 * torch.square(torch.clamp(nearest_foot_err - 0.05, min=0.0))) * foot_low_z
                compact_gap = torch.exp(-20.0 * torch.square(torch.clamp(ankle_gap - 0.22, min=0.0)))
                foot_block = torch.where(center_low, leg_block * compact_gap, leg_block)
                outer_low = low_ball & (~center_low) & (ball_local_x < 2.2)
                leg_weight = torch.where(center_low, 1.35, torch.where(straddling | outer_low, 0.95, 0.0))
                fivehole_rew = leg_weight * foot_block
                center_gate = torch.where(center_low, 0.35 + 0.65 * foot_block, 1.0)
                hand_indices = getattr(self, 'hand_indices', None)
                if hand_indices is not None and len(hand_indices) == 2:
                    low_hand_z = torch.exp(-16.0 * torch.square(torch.clamp(act_hand_z - 0.28, min=0.0)))
                    low_hand_y = torch.exp(-10.0 * torch.square(act_hand_y - track_y))
                    fivehole_rew = fivehole_rew + 1.15 * low_hand_z * low_hand_y * (low_ball & (ball_local_x < 2.2)).float()

            wide_rew = torch.zeros_like(self.dist)
            wide_gate = torch.ones_like(self.dist)
            hand_indices = getattr(self, 'hand_indices', None)
            if hand_indices is not None and len(hand_indices) == 2:
                hand_pos = self.rigid_body_states[:, hand_indices, :3]
                hand_gap_y = torch.abs(hand_pos[:, 0, 1] - hand_pos[:, 1, 1])
                hand_min_z = torch.minimum(hand_pos[:, 0, 2], hand_pos[:, 1, 2])
                wide_ready = torch.exp(-8.0 * torch.square(torch.clamp(0.75 - hand_gap_y, min=0.0)))
                height_ready = torch.exp(-10.0 * torch.square(torch.clamp(0.48 - hand_min_z, min=0.0)))
                early_phase = ball_local_x >= 2.0
                late_wide = torch.exp(-6.0 * torch.square(torch.clamp(0.50 - hand_gap_y, min=0.0))) * ((~low_ball) & (~early_phase)).float()
                wide_rew = 0.90 * wide_ready * height_ready * early_phase.float() + 0.40 * late_wide
                wide_gate = torch.where(early_phase, 0.55 + 0.45 * wide_ready * height_ready, 1.0)

            upright = torch.exp(-4.0 * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
            tilt_factor = 0.65 + 0.35 * upright
            base_reach = 0.42 * lat_rew + lat_vel_bonus + vert_weight * vert_rew + jump_vel_bonus + (0.90 * hand_reach + act_z_match) * act_style + off_low + live_track + fivehole_rew + wide_rew
            return base_reach * center_gate * wide_gate * active.float() * tilt_factor

        phase1 = ((self.ball_states[:, 0] - self.env_origins[:, 0] > 1.5) & (self.ball_states[:,7] - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()
        
        taskrew = torch.zeros(self.num_envs, dtype = torch.float, device = self.device)

        end_target_local = self.end_target - self.torso_pos

        
        asidegoal = torch.clip(end_target_local[:, 1], -1.0, 1.0)
        asidegoal[torch.abs(asidegoal) < 0.3] = 0.

        verticalgoal = torch.clip(self.torso_pos[:, 2] - torch.clip(self.end_target[:,2], 0.3, 1.2), 0.0, 1.0)

        phase1_rew = 1.0 - (verticalgoal + torch.abs(asidegoal)) / 2.0
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        jump_scale =  3.0 + 3.0 * self.curriculumupdate

        vel_sigma = taskrew.clone()
        vel_sigma[self.end_regions == 0] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 0, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 1] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 1, self.upper_body_index, 8], -3.0, 0.0)
        vel_sigma[self.end_regions == 4] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 4, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 5] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 5, self.upper_body_index, 8], -3.0, 0.0)

        vel_sigma[self.end_regions == 2] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 2, self.upper_body_index, 9], 0.0, 3.0)
        vel_sigma[self.end_regions == 3] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 3, self.upper_body_index, 9], 0.0, 3.0)

        vel_sigma[behind] = 2.0

        taskrew =(1 - 1 / (1 + torch.exp(-self.curriculumsigma * (self.dist - self.cfg.rewards.reach_th))))


        taskrew *= vel_sigma

        taskrew[phase1] = phase1_rew[phase1]


        return taskrew * (1 - torch.clip(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0., 1.0))

    def _k1_ball_safe(self, include_behind=False):
        # Cleared only after leaving the immediate goal-line zone or rebounding cleanly forward/sideways.
        local = self.ball_states[:, :3] - self.env_origins
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        vx = self.ball_states[:, 7]
        goal = self.cfg.goal
        edge_x = goal.line_x + goal.ball_radius
        half_w = getattr(goal, 'half_width', 1.30)
        side_clear = (torch.abs(y) > half_w + goal.ball_radius) & (vx >= -0.15)
        escaped = (
            (x < goal.line_x - goal.ball_radius)
            | ((x > goal.line_x + 0.02) & (vx >= 0.8))
            | ((x > edge_x) & (vx >= 0.6))
            | ((x > edge_x + 0.30) & (vx >= 0.05))
            | ((x > edge_x + 0.30) & (vx >= 0.0) & (z <= 0.14))
            | side_clear
        )
        safe = (self.success_flag > 0) & escaped
        if include_behind:
            safe = safe | (x < -0.10)
        return safe & (self.goal_conceded_flag == 0)

    def _k1_ball_settled_or_past(self):
        # True once the shot is saved/cleared OR has already entered the goal, so post-jump landing balance activates either way.
        return self._k1_ball_safe(include_behind=True) | (self.goal_conceded_flag > 0)

    def _k1_both_feet_support(self):
        """Return 1.0 when both feet bear load (>1 N) vs 0.25 otherwise to incentivize double-support recovery."""
        if not hasattr(self, 'contact_forces') or not hasattr(self, 'contact_feet_indices'):
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        both_feet = torch.all(self.contact_forces[:, self.contact_feet_indices, 2] > 1.0, dim=-1)
        return torch.where(both_feet, torch.ones_like(both_feet, dtype=torch.float), torch.full_like(both_feet, 0.25, dtype=torch.float))

    def _reward_post_hold(self):
        # Hold the last blocking pose once landed upright on both feet; release to low-velocity balance if tilting > ~23 deg.
        safe_mask = self._k1_ball_safe(include_behind=True)
        active = safe_mask.float()
        upper_vel_sq = torch.mean(torch.square(self.dof_vel[:, self.upper_body_joint_indices]), dim=-1)
        leg_vel_sq = torch.mean(torch.square(self.dof_vel[:, self.leg_joint_indices]), dim=-1)
        tilt_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        upright = torch.exp(-6.5 * tilt_sq)
        support = self._k1_both_feet_support() if hasattr(self, '_k1_both_feet_support') else torch.ones_like(tilt_sq)
        upper_still = 0.55 * torch.exp(-0.80 * upper_vel_sq) + 0.45 * torch.exp(-0.02 * upper_vel_sq)
        leg_still = 0.60 * torch.exp(-1.00 * leg_vel_sq) + 0.40 * torch.exp(-0.05 * leg_vel_sq)
        upper_term = upper_still
        if hasattr(self, 'k1_safe_steps') and hasattr(self, 'k1_hold_pose_locked'):
            self.k1_safe_steps = torch.where(safe_mask, self.k1_safe_steps + 1, 0)
            if self.k1_hold_upper_dof_pos is None:
                self.k1_hold_upper_dof_pos = self.dof_pos[:, self.upper_body_joint_indices].clone()
            both_feet = support > 0.99
            latch = safe_mask & (self.k1_safe_steps >= 3) & (~self.k1_hold_pose_locked) & both_feet & (tilt_sq <= 0.09)
            if latch.any():
                self.k1_hold_upper_dof_pos[latch] = self.dof_pos[latch][:, self.upper_body_joint_indices]
                self.k1_hold_pose_locked[latch] = True
            self.k1_hold_pose_locked &= safe_mask
            pose_err_sq = torch.mean(
                torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.k1_hold_upper_dof_pos),
                dim=-1,
            )
            pose_ok = self.k1_hold_pose_locked & (tilt_sq <= 0.16)
            pose_hold = torch.where(pose_ok, torch.exp(-6.0 * pose_err_sq), upper_still)
            upper_term = 0.50 * upper_still + 0.50 * pose_hold
        return (0.70 * upper_term + 0.30 * leg_still) * upright * support * active

    def _reward_collision(self):
        if len(self.penalised_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.asset.name == "k1":
            indices = torch.as_tensor(self.penalised_contact_indices, dtype=torch.long, device=self.device)
            hand_indices = getattr(self, 'hand_indices', None)
            if hand_indices is not None and len(hand_indices) > 0:
                h_idx = torch.as_tensor(hand_indices, dtype=torch.long, device=self.device)
                indices = torch.unique(torch.cat([indices, h_idx]))
            forces = torch.norm(self.contact_forces[:, indices, :], dim=-1)
            dist_to_ball = torch.norm(
                self.rigid_body_states[:, indices, :3] - self.ball_states[:, :3].unsqueeze(1),
                dim=-1,
            )
            # Penalize any non-ball self/ground collision (>2N) on hands, elbows, shoulders, or head.
            hit = forces > 2.0
            return torch.sum((hit & (dist_to_ball > 0.25)).float(), dim=1)
        forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        return torch.sum((forces > 1.0).float(), dim=1)

    def _reward_goal_conceded(self):
        # Event rewards divide by dt because reward scales are multiplied by dt at setup.
        return ((self.goal_conceded_flag > 0) & self.reset_buf).float() / self.dt

    def _reward_recovery(self):
        # shortcut: requires 25 consecutive calm double-support steps; zero under noisy PPO rollouts (std~0.39), see Run I EMA ablation.
        stable = self._k1_ball_safe()
        stable &= self.projected_gravity[:, 2] <= -0.82
        stable &= self.root_states[:, 2] - self.env_origins[:, 2] >= 0.7 * self.cfg.init_state.pos[2]
        stable &= torch.norm(self.base_lin_vel, dim=-1) < 0.60
        stable &= torch.norm(self.base_ang_vel, dim=-1) < 1.25
        stable &= torch.all(self.contact_forces[:, self.contact_feet_indices, 2] > 1.0, dim=-1)
        stable &= torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values >= self.cfg.rewards.min_knee_height
        self.recovery_stable_steps = torch.where(stable, self.recovery_stable_steps + 1, 0)
        recovered = self.recovery_stable_steps * self.dt >= 0.5
        return (recovered & stable & self.time_out_buf & self.reset_buf).float() / self.dt

    def _reward_success(self):
        if self.cfg.asset.name == "k1":
            return (self._k1_ball_safe() & self.time_out_buf & self.reset_buf).float() / self.dt
        return (self.success_flag + 1.0) * (self.dist < self.cfg.rewards.strict_th)

    

    def _reward_stopball(self):
        if self.cfg.asset.name == "k1":
            ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
            ball_local_y = self.ball_states[:, 1] - self.env_origins[:, 1]
            ball_z = self.ball_states[:, 2]
            min_arm_dist = getattr(self, 'cur_min_arm_dist', None)
            min_robot_dist = getattr(self, 'cur_min_robot_dist', None)
            if min_arm_dist is None or min_robot_dist is None:
                min_arm_dist, min_robot_dist = self._compute_k1_ball_dists()
            min_arm_dist = torch.minimum(min_arm_dist, self.substep_min_arm_dist)
            min_robot_dist = torch.minimum(min_robot_dist, self.substep_min_robot_dist)

            vx_post = self.ball_states[:, 7]
            dv_x = vx_post - self.prev_ball_lin_vel[:, 0]
            dv_y = torch.abs(self.ball_states[:, 8] - self.prev_ball_lin_vel[:, 1])
            dv_xy = torch.norm(self.ball_states[:, 7:9] - self.prev_ball_lin_vel[:, 0:2], dim=-1)

            force = torch.maximum(torch.norm(self.ball_contact_forces[:, 0, :], dim=-1), self.substep_max_ball_force)
            hit = contact_candidate(
                ball_z, min_robot_dist, dv_x, dv_y, dv_xy, force,
                ball_radius=self.cfg.goal.ball_radius,
            )
            threshold_x = self.cfg.goal.line_x - self.cfg.goal.ball_radius
            prev_local_x = self.prev_ball_pos[:, 0] - self.env_origins[:, 0]
            in_play_zone = (prev_local_x > threshold_x) | (ball_local_x > threshold_x)
            changevel = (self.stop_flag == 0) & (self.goal_conceded_flag == 0) & in_play_zone & hit

            dt_goal = torch.clamp((-0.05 - ball_local_x) / torch.clamp(vx_post, max=-0.35), min=0.0, max=1.5)
            y_pred = ball_local_y + self.ball_states[:, 8] * dt_goal
            z_pred = ball_z + self.ball_states[:, 9] * dt_goal - 0.5 * 9.81 * (dt_goal ** 2)
            clean_parry = (vx_post >= -0.25) | (torch.abs(y_pred) > 1.32) | (z_pred > 1.22)
            parry_mult = torch.where(clean_parry, torch.ones_like(vx_post), torch.zeros_like(vx_post))

            is_arm_hit = min_arm_dist < 0.24
            is_foot_kick = (~is_arm_hit) & (min_robot_dist < 0.24) & (ball_z <= 0.52)
            one_arm_clean = is_arm_hit
            hand_indices = getattr(self, 'hand_indices', None)
            head_idx = getattr(self, 'head_body_index', None)
            if hand_indices is not None and len(hand_indices) == 2 and head_idx is not None:
                hz_hand = self.rigid_body_states[:, hand_indices, 2]
                head_z = self.rigid_body_states[:, head_idx, 2]
                one_arm_clean = is_arm_hit & (torch.minimum(hz_hand[:, 0], hz_hand[:, 1]) <= (head_z - 0.10))
            style_bonus = torch.where(ball_z > 0.45, 0.40 * one_arm_clean.float(), 0.35 * (is_arm_hit | is_foot_kick).float())
            rew_stop = parry_mult * (1.0 + style_bonus) * (self.stop_flag == 0).float() * changevel.float()
            self.success_flag[changevel] = 1.0
            self.arm_save_flag[changevel & is_arm_hit] = 1.0
            self.stop_flag[changevel | (ball_local_x < threshold_x)] = 1.0
            return rew_stop

        changevel = (self.ball_states[:,7] - self.ball_vel > 2.0) & ((self.ball_states[:, 0] - self.env_origins[:, 0]) > 0.0)
        stopped_ids = (changevel |((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0)).nonzero(as_tuple=False).flatten()
        success_ids = ((self.stop_flag == 0) & changevel) .nonzero(as_tuple=False).flatten()
        rew_stop = 1.0 * (self.stop_flag == 0) * changevel
        self.success_flag[success_ids] = 1.0
        self.stop_flag[stopped_ids] = 1.0

        return rew_stop

    def _reward_feetorientaion(self):

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)
        rew = torch.exp(feet_orientation * -5)
        if getattr(self.cfg.asset, "name", None) == "k1":
            ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
            end_y_local = self.end_target[:, 1] - self.env_origins[:, 1]
            shot_target_z = getattr(self, 'ball_end', self.end_target)[:, 2]
            is_low = (self.end_regions == 4) | (self.end_regions == 5) | (shot_target_z <= 0.36)
            is_high = (self.end_regions == 2) | (self.end_regions == 3)
            center_low = is_low & (torch.abs(end_y_local) < 0.30)
            # Keep feet planted flat on center_low shots (anti-kolong); allow early push-off for high jumps & dynamic kicks.
            kick_win = torch.where(is_high, 3.2, 2.2)
            allow_kick_phase = (ball_local_x <= kick_win) & (~self._k1_ball_settled_or_past()) & (~center_low)
            return rew * (~allow_kick_phase).float()
        return rew

    def _reward_airfeetorientation(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        inair = torch.sum(foot_contact_forces_z, dim=-1) < 1.

        falling = (self.rigid_body_states[:, self.upper_body_index, 9] < 0.0) & inair
        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)

        if getattr(self.cfg.asset, "name", None) == "k1":
            return torch.exp(feet_orientation * -3) * falling
        return torch.exp(feet_orientation * -3) * falling * jump_ids



    def _reward_successland(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        
        if getattr(self.cfg.asset, "name", None) == "k1":
            in_air_now = torch.all(foot_contact_forces_z < 1.0, dim=-1)
            jump = (self.root_states[:, 2] > (self.cfg.init_state.pos[2] + 0.07)) | (
                in_air_now & (self.root_states[:, 2] > (self.cfg.init_state.pos[2] + 0.055))
            )
        else:
            jump_threshold = self.cfg.init_state.pos[2] + 0.15
            jump = self.root_states[:, 2] > jump_threshold
        
        self.has_in_air = torch.logical_or(self.has_in_air, jump)
        
        has_contact = (foot_contact_forces_z[:, 0] > 1.) & (foot_contact_forces_z[:, 1] > 1.)

        one_feet_contact = (((foot_contact_forces_z[:, 0] >  1.) & (foot_contact_forces_z[:, 1] < 1.)) | ((foot_contact_forces_z[:, 0] <  1.) & (foot_contact_forces_z[:, 1] > 1.))) & (self.has_in_air)

        successful_landings = torch.logical_and(has_contact, self.has_in_air)
        
        if getattr(self.cfg.asset, "name", None) == "k1":
            air_lift = torch.clamp((self.root_states[:, 2] - (self.cfg.init_state.pos[2] + 0.03)) / 0.10, min=0.0, max=1.3)
            ball_live = (~self._k1_ball_safe(include_behind=True)) & (self.goal_conceded_flag == 0)
            air_reward = torch.maximum(self.has_in_air.float(), 0.60 * air_lift * ball_live.float())
        else:
            air_reward = self.has_in_air.float()
        landing_reward = successful_landings.float() * 5.0

        one_feet_punish = torch.zeros_like(air_reward) if getattr(self.cfg.asset, "name", None) == "k1" else one_feet_contact.float() * -1.0

        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)
        
        return (air_reward + landing_reward + one_feet_punish) * jump_ids


    def _reward_feet_slippage(self):
        foot_vel = self.rigid_body_states[:,  self.contact_feet_indices, 7:10]  
        contactvel = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1) > 1.), dim=1)
        if self.cfg.asset.name == "k1":
            return torch.exp(contactvel * -10) * self._k1_both_feet_support() * self._k1_ball_settled_or_past().float()
        return torch.exp(contactvel * -10)

    def _reward_penalize_sharpcontact(self):

        return (torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) >  self.cfg.rewards.max_contact_force) * 1.0 


    def _reward_postorientation(self):
        if self.cfg.asset.name == "k1":
            return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -6.0) * self._k1_both_feet_support() * self._k1_ball_settled_or_past().float()
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -3) * behind


    def _reward_penalize_kneeheight(self):
        pen_knee_h = getattr(self.cfg.rewards, "penalize_knee_height", 0.15)
        return (torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < pen_knee_h) * 1.0

    def _reward_postangvel(self):
        if self.cfg.asset.name == "k1":
            return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3) * self._k1_both_feet_support() * self._k1_ball_settled_or_past().float()
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3)* behind

    def _reward_postlinvel(self):
        if self.cfg.asset.name == "k1":
            return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:2]), dim=1) * -3) * self._k1_both_feet_support() * self._k1_ball_settled_or_past().float()
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1) * -3)* behind

    def _reward_postupperdofpos(self):
        sq_err = torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.default_dof_pos[:, self.upper_body_joint_indices])
        if self.cfg.asset.name != "k1":
            behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
            reward = torch.exp(-1.0 * torch.sum(sq_err, dim=-1))
            return reward * behind
        else:
            behind = ((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0) | (self.stop_flag > 0)
            gate = 0.15 + 0.85 * self.success_flag * (1.0 - self.goal_conceded_flag)
            reward = torch.exp(-2.0 * torch.mean(sq_err, dim=-1))
            return reward * behind.float() * gate

    def _reward_head_track(self):
        """ Menghadiahi robot jika sendi kepala (yaw & pitch) mengarah mengikuti bola saat bola datang. """
        if len(self.head_joint_indices) < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.asset.name != "k1":
            behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:, 7] - self.ball_vel > 2.0)
        else:
            behind = self._k1_ball_safe() | (self.goal_conceded_flag > 0)
        ball_local = quat_rotate_inverse(self.base_quat, self.ball_states[:, :3] - self.torso_pos)
        dx = torch.clamp(ball_local[:, 0], min=0.1)
        dy = ball_local[:, 1]
        dz = ball_local[:, 2] - 0.345  # posisi kamera stereo/RealSense K1 ~34.5 cm di atas titik pusat trunk
        target_yaw = torch.clamp(torch.atan2(dy, dx), -0.95, 0.95)
        horiz_dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        target_pitch = torch.clamp(torch.atan2(-dz, horiz_dist), -0.28, 0.75)
        yaw_err = torch.square(self.dof_pos[:, self.head_joint_indices[0]] - target_yaw)
        pitch_err = torch.square(self.dof_pos[:, self.head_joint_indices[1]] - target_pitch)
        return torch.exp(-4.0 * (yaw_err + pitch_err)) * (~behind)

    def _reward_postwaistdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.waist_joint_indices] - self.default_dof_pos[:, self.waist_joint_indices]), dim=-1)
        reward = torch.exp(-3 * mse) 
        return reward * behind

    def _reward_stayonline(self):
        distance = torch.clip(torch.abs(self.torso_pos[:,0] - self.env_origins[:, 0]), 0.2, 1.2) - 0.2   
        if getattr(self.cfg.asset, "name", None) == "k1":
            return distance * (0.30 + 0.70 * self._k1_ball_safe(include_behind=True).float())
        return distance

    def _reward_noretreat(self):
        pen = -1 * torch.clip(self.base_lin_vel[:, 0], -1.0, 0.0)
        if getattr(self.cfg.asset, "name", None) == "k1":
            return pen * (0.30 + 0.70 * self._k1_ball_safe(include_behind=True).float())
        return pen
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations lightly while live for agility, strongly post-shot to settle cleanly.
        sq_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        if getattr(self.cfg.asset, "name", None) == "k1" and hasattr(self, "_k1_ball_settled_or_past"):
            return sq_acc * (0.55 + 2.45 * self._k1_ball_settled_or_past().float())
        return sq_acc

    
    def _reward_smoothness(self):
        # Second order action smoothness: light while live for fast arm/jump whip, 3.0x post-shot to kill kejang2.
        sq_d2a = torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
        if getattr(self.cfg.asset, "name", None) == "k1" and hasattr(self, "_k1_ball_settled_or_past"):
            return sq_d2a * (0.40 + 2.60 * self._k1_ball_settled_or_past().float())
        return sq_d2a
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    
    def _reward_termination(self):
        # Terminal penalty for non-timeout falls
        if self.cfg.asset.name == "k1":
            return (self.reset_buf & ~self.time_out_buf).float() / self.dt
        return (self.reset_buf & ~self.time_out_buf).float()

    def _reward_deviation_waist_pitch_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices[2]], dim=-1)
