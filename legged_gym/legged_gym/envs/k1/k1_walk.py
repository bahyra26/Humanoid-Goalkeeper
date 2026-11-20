import copy
import math
import os
import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict


def build_walk_one_step_obs(
    commands,
    gait_sin_cos,
    base_ang_vel,
    projected_gravity,
    dof_pos_err,
    dof_vel,
    actions,
    cmd_scales=(2.0, 2.0, 0.25),
    ang_vel_scale=0.25,
    dof_pos_scale=1.0,
    dof_vel_scale=0.05,
):
    """Single source of truth for Policy 2 (k1_walk) 77-dim one-step actor observation."""
    scaled_cmd = torch.stack(
        (
            commands[:, 0] * cmd_scales[0],
            commands[:, 1] * cmd_scales[1],
            commands[:, 2] * cmd_scales[2],
        ),
        dim=-1,
    )
    return torch.cat(
        (
            scaled_cmd,
            gait_sin_cos,
            base_ang_vel * ang_vel_scale,
            projected_gravity,
            dof_pos_err * dof_pos_scale,
            dof_vel * dof_vel_scale,
            actions,
        ),
        dim=-1,
    )


class K1WalkEnv(BaseTask):
    """Omnidirectional velocity-tracking + upright-recovery locomotion env for Booster K1."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_amp_obs = 1
        self.motions = None
        self.actor_history_length = self.cfg.env.num_actor_history

        if not self.headless and self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(2.4, 2.0, 1.2), gymapi.Vec3(0.0, 0.0, 0.5)
            )
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def _parse_cfg(self, cfg):
        self.dt = cfg.control.decimation * self.sim_params.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        self.play = getattr(cfg.env, 'play', False)
        self.cfg.domain_rand.push_interval = max(1, math.ceil(cfg.domain_rand.push_interval_s / self.dt))
        self.resample_steps = max(1, math.ceil(cfg.commands.resampling_time / self.dt))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = int(np.floor(np.sqrt(self.num_envs)))
        num_rows = int(np.ceil(self.num_envs / num_cols))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            for i in range(len(props)):
                lo, hi = props["lower"][i].item(), props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = lo
                self.hard_dof_pos_limits[i, 1] = hi
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = 0.5 * (lo + hi)
                r = hi - lo
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                fr = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
                friction_buckets = torch_rand_float(fr[0], fr[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids].to(self.device)
            for s in range(len(props)):
                props[s].friction = float(self.friction_coeffs[env_id, 0].item())
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = float(self.default_rigid_body_mass[0].item() + self.payload[env_id, 0].item())
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(
                float(self.com_displacement[env_id, 0].item()),
                float(self.com_displacement[env_id, 1].item()),
                float(self.com_displacement[env_id, 2].item()),
            )
        if self.cfg.domain_rand.randomize_link_mass:
            lo, hi = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = float(np.random.uniform(lo, hi) * self.default_rigid_body_mass[i].item())
        return props

    def _create_envs(self):
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
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        base_init_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_list, device=self.device, requires_grad=False)
        self._get_env_origins()

        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            rng = self.cfg.domain_rand.payload_mass_range
            self.payload = torch_rand_float(rng[0], rng[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            rng = self.cfg.domain_rand.com_displacement_range
            self.com_displacement = torch_rand_float(rng[0], rng[1], (self.num_envs, 3), device=self.device)

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device)
        self.envs = []
        self.actor_handles = []
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[2] = self.base_init_state[2]
            start_pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        feet_names = [s for s in body_names if self.cfg.asset.contact_foot_names in s]
        self.contact_feet_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in feet_names],
            dtype=torch.long,
            device=self.device,
        )
        self.knee_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in self.cfg.asset.knee_names],
            dtype=torch.long,
            device=self.device,
        )
        pen_names = []
        for pat in self.cfg.asset.penalize_contacts_on:
            pen_names.extend([s for s in body_names if pat in s])
        self.penalised_contact_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in pen_names],
            dtype=torch.long,
            device=self.device,
        )

        self.arm_joint_indices = torch.tensor(
            [self.dof_names.index(n) for n in (self.cfg.control.left_arm_joints + self.cfg.control.right_arm_joints)],
            dtype=torch.long,
            device=self.device,
        )
        self.head_joint_indices = torch.tensor(
            [self.dof_names.index(n) for n in self.cfg.control.head_joints],
            dtype=torch.long,
            device=self.device,
        )
        self.upper_body_joint_indices = torch.cat((self.head_joint_indices, self.arm_joint_indices))
        self.leg_joint_indices = torch.tensor(
            [self.dof_names.index(n) for n in (self.cfg.control.left_leg_joints + self.cfg.control.right_leg_joints)],
            dtype=torch.long,
            device=self.device,
        )
        self.hip_yaw_roll_indices = torch.tensor(
            [
                self.dof_names.index(n)
                for n in (
                    'left_hip_roll_joint',
                    'left_hip_yaw_joint',
                    'right_hip_roll_joint',
                    'right_hip_yaw_joint',
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7].clone()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            for key, kp in self.cfg.control.stiffness.items():
                if key in name:
                    self.p_gains[i] = kp
                    self.d_gains[i] = self.cfg.control.damping[key]
                    break
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.action_scale_vec = torch.full((1, self.num_dof), float(self.cfg.control.action_scale), device=self.device)
        self.action_scale_vec[0, self.arm_joint_indices] = float(self.cfg.control.arm_action_scale)
        self.action_scale_vec[0, self.leg_joint_indices] = float(self.cfg.control.leg_action_scale)
        self.action_scale_vec[0, self.head_joint_indices] = float(self.cfg.control.head_action_scale)

        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)

        self.add_noise = self.cfg.noise.add_noise
        self.noise_scale_vec = self._get_noise_scale_vec()

        self.post_parry_data = None
        ds_tmpl = getattr(self.cfg.domain_rand, 'post_parry_dataset_path', None)
        if ds_tmpl:
            ds_path = ds_tmpl.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            if os.path.isfile(ds_path):
                raw = torch.load(ds_path, map_location=self.device)
                self.post_parry_data = {
                    k: raw[k].to(self.device)
                    for k in ('base_z', 'base_quat', 'lin_vel', 'ang_vel', 'dof_pos', 'dof_vel', 'last_action')
                }
                print(f"[K1WalkEnv] Loaded {self.post_parry_data['base_z'].shape[0]} real post-parry states from {ds_path}")

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.num_one_step_obs, dtype=torch.float, device=self.device)
        ns = self.cfg.noise.noise_scales
        nl = self.cfg.noise.noise_level
        # [0:3] commands, [3:5] gait_sin_cos: noiseless
        noise_vec[5:8] = ns.ang_vel * nl * self.obs_scales.ang_vel
        noise_vec[8:11] = ns.gravity * nl
        noise_vec[11:33] = ns.dof_pos * nl * self.obs_scales.dof_pos
        noise_vec[33:55] = ns.dof_vel * nl * self.obs_scales.dof_vel
        return noise_vec

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            if self.reward_scales[key] == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_scales.keys()
        }

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        rng = self.cfg.commands.ranges
        n = len(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(rng.lin_vel_x[0], rng.lin_vel_x[1], (n, 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(rng.lin_vel_y[0], rng.lin_vel_y[1], (n, 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(rng.ang_vel_yaw[0], rng.ang_vel_yaw[1], (n, 1), device=self.device).squeeze(1)
        # Structure command mix: standing + pure yaw + pure strafe + pure fwd/back + combined omni.
        u = torch.rand(n, device=self.device)
        p_stand = float(self.cfg.commands.standing_prob)
        stand_mask = u < p_stand
        pure_yaw = (u >= p_stand) & (u < p_stand + 0.18)
        pure_lat = (u >= p_stand + 0.18) & (u < p_stand + 0.34)
        pure_fwd = (u >= p_stand + 0.34) & (u < p_stand + 0.50)
        if stand_mask.any():
            self.commands[env_ids[stand_mask]] = 0.0
        if pure_yaw.any():
            self.commands[env_ids[pure_yaw], 0:2] = 0.0
        if pure_lat.any():
            self.commands[env_ids[pure_lat], 0] = 0.0
            self.commands[env_ids[pure_lat], 2] = 0.0
        if pure_fwd.any():
            self.commands[env_ids[pure_fwd], 1:3] = 0.0
        small = torch.norm(self.commands[env_ids], dim=-1) < float(self.cfg.commands.move_threshold)
        if small.any():
            self.commands[env_ids[small]] = 0.0

    def _is_moving(self):
        return torch.norm(self.commands, dim=-1) >= float(self.cfg.commands.move_threshold)

    def _gait_sin_cos(self):
        angle = 2.0 * math.pi * self.gait_phase
        moving = self._is_moving().float().unsqueeze(-1)
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1) * moving

    def _compute_torques(self, actions):
        actions_scaled = actions * self.action_scale_vec
        target = self.default_dof_pos + actions_scaled
        if getattr(self.cfg.control, "limit_joint_targets", False):
            target = torch.maximum(
                torch.minimum(target, self.hard_dof_pos_limits[:, 1]),
                self.hard_dof_pos_limits[:, 0],
            )
        torques = self.p_gains * self.Kp_factors * (target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        delayed = self.actions.view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        if self.cfg.domain_rand.delay:
            delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
            for i in range(self.cfg.control.decimation):
                delayed[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        if self.cfg.domain_rand.randomize_joint_injection:
            rng = self.cfg.domain_rand.joint_injection_range
            self.joint_injection = (
                torch_rand_float(rng[0], rng[1], (self.num_envs, self.num_dof), device=self.device)
                * self.torque_limits.unsqueeze(0)
            )

        self.render()
        for i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(delayed[i])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        env_ids, term_priv_obs = self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, env_ids, term_priv_obs

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.extras.pop("episode", None)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        moving = self._is_moving()
        self.gait_phase = torch.where(
            moving,
            torch.remainder(self.gait_phase + self.dt / float(self.cfg.commands.gait_period), 1.0),
            torch.zeros_like(self.gait_phase),
        )

        # Periodic command resampling and random velocity pushes
        if not self.play:
            resample_ids = (self.episode_length_buf % self.resample_steps == 0).nonzero(as_tuple=False).flatten()
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids)
            if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
                self._push_robots()

        self.check_termination()
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        term_priv_obs = self._build_privileged_obs(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return env_ids, term_priv_obs

    def _push_robots(self):
        max_vel = float(self.cfg.domain_rand.max_push_vel_xy)
        self.root_states[:, 7:9] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        min_knee = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values
        base_z = self.root_states[:, 2] - self.env_origins[:, 2]
        fell = (min_knee < self.cfg.rewards.min_knee_height) | (self.projected_gravity[:, 2] > -0.50) | (base_z < 0.25)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = fell | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.extras["episode"] = {}
        ep_len_s = torch.clip(self.episode_length_buf[env_ids].float(), min=1.0) * self.dt
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / ep_len_s)
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["fall_rate"] = (~self.time_out_buf[env_ids]).float().mean()

        n = len(env_ids)
        use_real = torch.zeros(n, dtype=torch.bool, device=self.device)
        real_pick = None
        if (not self.play) and (self.post_parry_data is not None):
            real_prob = float(getattr(self.cfg.domain_rand, 'real_post_parry_prob', 0.0))
            if real_prob > 0.0:
                use_real = torch.rand(n, device=self.device) < real_prob
                if use_real.any():
                    n_ds = self.post_parry_data['base_z'].shape[0]
                    real_pick = torch.randint(0, n_ds, (int(use_real.sum()),), device=self.device)

        # Decide which remaining envs get synthetic post-parry crooked-landing recovery starts
        if (not self.play) and getattr(self.cfg.domain_rand, 'post_parry_reset_prob', 0.0) > 0.0:
            crooked = (~use_real) & (torch.rand(n, device=self.device) < float(self.cfg.domain_rand.post_parry_reset_prob))
        else:
            crooked = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Reset DOFs
        dof_pos = self.default_dof_pos.repeat(n, 1)
        dof_vel = torch.zeros(n, self.num_dof, dtype=torch.float, device=self.device)
        if not self.play:
            if self.cfg.domain_rand.randomize_initial_joint_pos:
                s_lo, s_hi = self.cfg.domain_rand.initial_joint_pos_scale
                o_lo, o_hi = self.cfg.domain_rand.initial_joint_pos_offset
                dof_pos = dof_pos * torch_rand_float(s_lo, s_hi, (n, self.num_dof), device=self.device)
                dof_pos += torch_rand_float(o_lo, o_hi, (n, self.num_dof), device=self.device)
            if crooked.any():
                c_lo, c_hi = self.cfg.domain_rand.post_parry_dof_offset
                dof_pos[crooked] += torch_rand_float(c_lo, c_hi, (int(crooked.sum()), self.num_dof), device=self.device)
            if use_real.any():
                dof_pos[use_real] = self.post_parry_data['dof_pos'][real_pick]
                dof_vel[use_real] = self.post_parry_data['dof_vel'][real_pick]
        dof_pos = torch.clip(dof_pos, self.hard_dof_pos_limits[:, 0].unsqueeze(0), self.hard_dof_pos_limits[:, 1].unsqueeze(0))
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # Reset root states
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if not self.play:
            roll = torch_rand_float(-0.05, 0.05, (n, 1), device=self.device).squeeze(1)
            pitch = torch_rand_float(-0.05, 0.05, (n, 1), device=self.device).squeeze(1)
            yaw = torch_rand_float(-math.pi, math.pi, (n, 1), device=self.device).squeeze(1)
            if crooked.any():
                nc = int(crooked.sum())
                t_lo, t_hi = self.cfg.domain_rand.post_parry_tilt_range
                roll[crooked] = torch_rand_float(t_lo, t_hi, (nc, 1), device=self.device).squeeze(1)
                pitch[crooked] = torch_rand_float(t_lo, t_hi, (nc, 1), device=self.device).squeeze(1)
                self.root_states[env_ids[crooked], 2] += 0.025
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

            v_lo, v_hi = self.cfg.domain_rand.init_base_vel_range
            self.root_states[env_ids, 7:13] = torch_rand_float(v_lo, v_hi, (n, 6), device=self.device)
            if crooked.any():
                nc = int(crooked.sum())
                lv_lo, lv_hi = self.cfg.domain_rand.post_parry_lin_vel
                av_lo, av_hi = self.cfg.domain_rand.post_parry_ang_vel
                self.root_states[env_ids[crooked], 7:9] = torch_rand_float(lv_lo, lv_hi, (nc, 2), device=self.device)
                self.root_states[env_ids[crooked], 10:13] = torch_rand_float(av_lo, av_hi, (nc, 3), device=self.device)
            if use_real.any():
                r_ids = env_ids[use_real]
                self.root_states[r_ids, 2] = self.env_origins[r_ids, 2] + self.post_parry_data['base_z'][real_pick]
                self.root_states[r_ids, 3:7] = self.post_parry_data['base_quat'][real_pick]
                self.root_states[r_ids, 7:10] = self.post_parry_data['lin_vel'][real_pick]
                self.root_states[r_ids, 10:13] = self.post_parry_data['ang_vel'][real_pick]

        env_ids_i32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_i32), len(env_ids_i32)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_i32), len(env_ids_i32)
        )

        # Randomize PD gains and actuation offsets per episode
        if not self.play:
            if self.cfg.domain_rand.randomize_kp:
                lo, hi = self.cfg.domain_rand.kp_range
                self.Kp_factors[env_ids] = torch_rand_float(lo, hi, (n, self.num_dof), device=self.device)
            if self.cfg.domain_rand.randomize_kd:
                lo, hi = self.cfg.domain_rand.kd_range
                self.Kd_factors[env_ids] = torch_rand_float(lo, hi, (n, self.num_dof), device=self.device)
            if self.cfg.domain_rand.randomize_actuation_offset:
                lo, hi = self.cfg.domain_rand.actuation_offset_range
                self.actuation_offset[env_ids] = (
                    torch_rand_float(lo, hi, (n, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
                )

        self._resample_commands(env_ids)
        if use_real.any():
            # Half of real post-parry resets command immediate STAND (0,0,0) balance catch; half command walking.
            r_ids = env_ids[use_real]
            catch_stand = torch.rand(len(r_ids), device=self.device) < 0.50
            if catch_stand.any():
                self.commands[r_ids[catch_stand]] = 0.0
        self.gait_phase[env_ids] = torch.rand(n, device=self.device)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        if use_real.any():
            r_ids = env_ids[use_real]
            prev_a = self.post_parry_data['last_action'][real_pick]
            self.actions[r_ids] = prev_a
            self.last_actions[r_ids] = prev_a
            self.last_last_actions[r_ids] = prev_a
        self.last_dof_vel[env_ids] = dof_vel
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Refresh post-reset kinematics before computing initial obs
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])

        # Seed history buffer cleanly for reset envs so frame 0 has no zero-filled jump
        one_step = build_walk_one_step_obs(
            self.commands[env_ids],
            self._gait_sin_cos()[env_ids],
            self.base_ang_vel[env_ids],
            self.projected_gravity[env_ids],
            self.dof_pos[env_ids] - self.default_dof_pos,
            self.dof_vel[env_ids],
            self.actions[env_ids],
        )
        self.obs_buf[env_ids] = one_step.repeat(1, self.actor_history_length)

    def _build_privileged_obs(self, env_ids=None):
        if env_ids is not None and len(env_ids) == 0:
            return torch.zeros(0, self.num_privileged_obs, dtype=torch.float, device=self.device)
        sl = slice(None) if env_ids is None else env_ids
        one_step = build_walk_one_step_obs(
            self.commands[sl],
            self._gait_sin_cos()[sl],
            self.base_ang_vel[sl],
            self.projected_gravity[sl],
            self.dof_pos[sl] - self.default_dof_pos,
            self.dof_vel[sl],
            self.actions[sl],
        )
        foot_contacts = (self.contact_forces[sl][:, self.contact_feet_indices, 2] > 1.0).float()
        foot_heights = self.rigid_body_states[sl][:, self.contact_feet_indices, 2] - self.env_origins[sl, 2:3]
        return torch.cat(
            (
                one_step,
                self.base_lin_vel[sl] * self.obs_scales.lin_vel,
                foot_contacts,
                foot_heights,
                self.friction_coeffs[sl],
            ),
            dim=-1,
        )

    def compute_observations(self):
        one_step = build_walk_one_step_obs(
            self.commands,
            self._gait_sin_cos(),
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos - self.default_dof_pos,
            self.dof_vel,
            self.actions,
        )
        actor_step = one_step.clone()
        if self.add_noise:
            actor_step += (2.0 * torch.rand_like(actor_step) - 1.0) * self.noise_scale_vec
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs :], actor_step), dim=-1)
        self.privileged_obs_buf = self._build_privileged_obs()

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for name, fn in zip(self.reward_names, self.reward_functions):
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if 'termination' in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales['termination']
            self.rew_buf += rew
            self.episode_sums['termination'] += rew

    # --- Rewards ---
    def _reward_alive(self):
        return (~(self.reset_buf & ~self.time_out_buf)).float()

    def _reward_tracking_lin_vel(self):
        err_sq = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=-1)
        return torch.exp(-err_sq / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        err_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err_sq / self.cfg.rewards.tracking_sigma)

    def _reward_gait_contact(self):
        contacts = self.contact_forces[:, self.contact_feet_indices, 2] > 1.0
        c_l, c_r = contacts[:, 0], contacts[:, 1]
        sin_p = torch.sin(2.0 * math.pi * self.gait_phase)
        # Strict alternating single-support reward during swing phase (zero free credit for standing glued to floor).
        left_stance_only = (sin_p > 0.15) & c_l & (~c_r)
        right_stance_only = (sin_p < -0.15) & c_r & (~c_l)
        double_support_window = (torch.abs(sin_p) <= 0.15) & c_l & c_r
        walk_rew = (left_stance_only | right_stance_only).float() + 0.35 * double_support_window.float()
        stand_rew = (c_l & c_r).float()
        return torch.where(self._is_moving(), walk_rew, stand_rew)

    def _reward_feet_clearance(self):
        contacts = self.contact_forces[:, self.contact_feet_indices, 2] > 1.0
        sin_p = torch.sin(2.0 * math.pi * self.gait_phase)
        swing_l = ((sin_p < -0.15) & (~contacts[:, 0])).float()
        swing_r = ((sin_p > 0.15) & (~contacts[:, 1])).float()
        fz = self.rigid_body_states[:, self.contact_feet_indices, 2] - self.env_origins[:, 2:3]
        target_z = float(self.cfg.rewards.rest_foot_z + self.cfg.rewards.swing_clearance)
        cl_l = torch.exp(-torch.square((fz[:, 0] - target_z) / 0.035)) * swing_l
        cl_r = torch.exp(-torch.square((fz[:, 1] - target_z) / 0.035)) * swing_r
        return (cl_l + cl_r) * self._is_moving().float()

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_base_height(self):
        base_z = self.root_states[:, 2] - self.env_origins[:, 2]
        return torch.square(base_z - float(self.cfg.rewards.target_base_height))

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_feet_orientation(self):
        l_grav = quat_rotate_inverse(self.rigid_body_states[:, self.contact_feet_indices[0], 3:7], self.gravity_vec)
        r_grav = quat_rotate_inverse(self.rigid_body_states[:, self.contact_feet_indices[1], 3:7], self.gravity_vec)
        return torch.sum(torch.square(l_grav[:, :2]), dim=-1) + torch.sum(torch.square(r_grav[:, :2]), dim=-1)

    def _reward_feet_distance(self):
        fpos = self.rigid_body_states[:, self.contact_feet_indices, :2]
        dist = torch.norm(fpos[:, 0, :] - fpos[:, 1, :], dim=-1)
        too_close = torch.clamp(float(self.cfg.rewards.min_foot_dist) - dist, min=0.0)
        too_far = torch.clamp(dist - float(self.cfg.rewards.max_foot_dist), min=0.0)
        return too_close + too_far

    def _reward_feet_slip(self):
        contacts = (self.contact_forces[:, self.contact_feet_indices, 2] > 1.0).float()
        foot_vel_xy = torch.norm(self.rigid_body_states[:, self.contact_feet_indices, 7:9], dim=-1)
        return torch.sum(foot_vel_xy * contacts, dim=-1)

    def _reward_upper_body_pose(self):
        err = self.dof_pos[:, self.upper_body_joint_indices] - self.default_dof_pos[:, self.upper_body_joint_indices]
        return torch.mean(torch.square(err), dim=-1)

    def _reward_stand_still(self):
        still_mask = (~self._is_moving()).float()
        dof_err = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        dof_v = torch.mean(torch.square(self.dof_vel), dim=-1)
        lin_v = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=-1)
        pose_ok = torch.exp(-6.0 * dof_err)
        calm_ok = torch.exp(-0.25 * dof_v) * torch.exp(-6.0 * lin_v)
        return (0.55 * pose_ok + 0.45 * calm_ok) * still_mask

    def _reward_hip_yaw_roll(self):
        err = self.dof_pos[:, self.hip_yaw_roll_indices] - self.default_dof_pos[:, self.hip_yaw_roll_indices]
        no_turn = (torch.abs(self.commands[:, 2]) < 0.15).float()
        return torch.sum(torch.square(err), dim=-1) * (0.25 + 0.75 * no_turn)

    def _reward_action_rate(self):
        mult = torch.where(self._is_moving(), 1.0, 3.0)
        return torch.sum(torch.square(self.actions - self.last_actions), dim=-1) * mult

    def _reward_smoothness(self):
        mult = torch.where(self._is_moving(), 1.0, 3.0)
        d2 = self.actions - 2.0 * self.last_actions + self.last_last_actions
        return torch.sum(torch.square(d2), dim=-1) * mult

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=-1)

    def _reward_dof_pos_limits(self):
        out = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out, dim=-1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=-1)

    def _reward_collision(self):
        if len(self.penalised_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        return torch.sum((forces > 2.0).float(), dim=-1)

    def _reward_termination(self):
        return (self.reset_buf & ~self.time_out_buf).float() / self.dt
