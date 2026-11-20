"""2-Policy state machine coordinating Booster K1 Goalkeeper (Policy 1) and Walk-Back (Policy 2).

Flow per environment:
  1. STATE_GOALKEEPER (0): Policy 1 active during incoming shot & wild rebounds.
  2. Once the shot is 100% finished/safe for >= 0.4 s (_k1_play_fully_done) and K1 is
     grounded & upright enough to recover for >= 0.10 s, transition smoothly (0.16 s
     linear action blend) into STATE_WALK_BACK (1).
  3. STATE_WALK_BACK (1): Policy 2 stabilizes upright first, then walks K1 back to
     goal center (x = 0, y = 0, yaw = 0) using body-frame PD velocity commands.
  4. STATE_READY_HOME (2): Once arrived and calm at (0, 0, yaw = 0) for >= 0.30 s,
     Policy 2 holds a calm upright ready stance (cmd = 0, zero twitching).
  5. Emergency preemption: any genuine incoming ball immediately restores STATE_GOALKEEPER
     with zero blend delay.
"""
import math
import torch

STATE_GOALKEEPER = 0
STATE_WALK_BACK = 1
STATE_READY_HOME = 2


def quat_to_yaw(quat_xyzw):
    """Extract yaw angle in [-pi, pi] from Isaac Gym xyzw quaternions."""
    qx, qy, qz, qw = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def build_walk_obs_step(
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
    """77-dim one-step observation matching K1WalkEnv.build_walk_one_step_obs."""
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


def compute_home_velocity_command(
    local_xy,
    yaw,
    projected_gravity_z,
    kp_xy=1.0,
    kp_yaw=1.2,
    max_vx_fwd=0.65,
    max_vx_back=0.50,
    max_vy=0.40,
    max_vyaw=0.70,
    pos_deadzone=0.08,
    yaw_deadzone=0.10,
):
    """Body-frame PD velocity command driving (local_xy, yaw) back to (0, 0, 0).

    Scales down linear/yaw motion while trunk is still leaning so Policy 2 stabilizes
    upright first before walking back.
    """
    ex = -local_xy[:, 0]
    ey = -local_xy[:, 1]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    ex_b = c * ex + s * ey
    ey_b = -s * ex + c * ey

    vx = torch.clamp(kp_xy * ex_b, -max_vx_back, max_vx_fwd)
    vy = torch.clamp(kp_xy * ey_b, -max_vy, max_vy)
    vyaw = torch.clamp(-kp_yaw * yaw, -max_vyaw, max_vyaw)

    dist = torch.norm(local_xy, dim=-1)
    at_home = (dist < pos_deadzone) & (torch.abs(yaw) < yaw_deadzone)

    # Upright-first scaling: catch balance in STAND mode first while leaning, then walk home smoothly.
    tilt_scale = torch.where(
        projected_gravity_z > -0.93,
        torch.zeros_like(projected_gravity_z),
        torch.where(projected_gravity_z > -0.96, torch.full_like(projected_gravity_z, 0.40), torch.ones_like(projected_gravity_z)),
    )
    scale = torch.where(at_home, torch.zeros_like(tilt_scale), tilt_scale)
    cmd = torch.stack((vx, vy, vyaw), dim=-1) * scale.unsqueeze(-1)
    small = torch.norm(cmd, dim=-1) < 0.05
    cmd[small] = 0.0
    return cmd


class K1TwoPolicyController:
    """Runtime coordinator between Policy 1 (Goalkeeper) and Policy 2 (k1_walk)."""

    def __init__(
        self,
        num_envs,
        num_actions=22,
        dt=0.02,
        device='cpu',
        walk_history_len=5,
        walk_one_step_dim=77,
        gait_period=0.70,
        handover_hold_steps=2,
        arrive_hold_steps=12,
        blend_steps=6,
    ):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.dt = float(dt)
        self.device = device
        self.walk_history_len = walk_history_len
        self.walk_one_step_dim = walk_one_step_dim
        self.gait_period = float(gait_period)
        self.handover_hold_steps = int(handover_hold_steps)
        self.arrive_hold_steps = int(arrive_hold_steps)
        self.blend_steps = int(blend_steps)

        self.state = torch.full((num_envs,), STATE_GOALKEEPER, dtype=torch.long, device=device)
        self.handover_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.arrive_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.walk_phase = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.walk_commands = torch.zeros(num_envs, 3, dtype=torch.float, device=device)
        self.walk_obs_buf = torch.zeros(num_envs, walk_history_len * walk_one_step_dim, dtype=torch.float, device=device)
        self.blend_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.blend_start_action = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)
        self.prev_action = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.state.fill_(STATE_GOALKEEPER)
            self.handover_steps.zero_()
            self.arrive_steps.zero_()
            self.walk_phase.zero_()
            self.walk_commands.zero_()
            self.walk_obs_buf.zero_()
            self.blend_remaining.zero_()
            self.blend_start_action.zero_()
            self.prev_action.zero_()
            return
        if len(env_ids) == 0:
            return
        self.state[env_ids] = STATE_GOALKEEPER
        self.handover_steps[env_ids] = 0
        self.arrive_steps[env_ids] = 0
        self.walk_phase[env_ids] = 0.0
        self.walk_commands[env_ids] = 0.0
        self.walk_obs_buf[env_ids] = 0.0
        self.blend_remaining[env_ids] = 0
        self.blend_start_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0

    def _ball_re_approaching(self, env):
        local = env.ball_states[:, :3] - env.env_origins
        goal = env.cfg.goal
        half_w = getattr(goal, 'half_width', 1.30)
        return (
            (local[:, 0] > goal.line_x + goal.ball_radius)
            & (local[:, 0] < 6.0)
            & (torch.abs(local[:, 1]) < half_w + 0.5)
            & (env.ball_states[:, 7] < -0.5)
            & (env.goal_conceded_flag == 0)
        )

    def step_action(self, env, gk_obs, gk_policy, walk_policy):
        """Compute blended 2-policy actions and update state machine for all envs."""
        local_xy = env.root_states[:, :2] - env.env_origins[:, :2]
        base_z = env.root_states[:, 2] - env.env_origins[:, 2]
        yaw = quat_to_yaw(env.base_quat)
        pgz = env.projected_gravity[:, 2]
        both_feet = torch.all(env.contact_forces[:, env.contact_feet_indices, 2] > 1.0, dim=-1)
        play_done = env._k1_play_fully_done()
        re_approach = self._ball_re_approaching(env)

        # 1. Emergency preemption: incoming ball immediately restores Policy 1 (no blend delay).
        preempt = (self.state != STATE_GOALKEEPER) & re_approach
        if preempt.any():
            self.state[preempt] = STATE_GOALKEEPER
            self.handover_steps[preempt] = 0
            self.arrive_steps[preempt] = 0
            self.blend_remaining[preempt] = 0
            self.walk_phase[preempt] = 0.0

        # 2. Transition GOALKEEPER -> WALK_BACK once shot is 100% finished; allow catching single-foot/tilted landings.
        in_gk = self.state == STATE_GOALKEEPER
        any_foot = torch.any(env.contact_forces[:, env.contact_feet_indices, 2] > 1.0, dim=-1)
        can_handover = in_gk & play_done & (~re_approach) & any_foot & (base_z >= 0.34) & (pgz <= -0.68)
        self.handover_steps = torch.where(can_handover, self.handover_steps + 1, torch.zeros_like(self.handover_steps))
        to_walk = in_gk & (self.handover_steps >= self.handover_hold_steps)
        if to_walk.any():
            self.state[to_walk] = STATE_WALK_BACK
            self.handover_steps[to_walk] = 0
            self.arrive_steps[to_walk] = 0
            self.walk_phase[to_walk] = 0.0
            self.blend_start_action[to_walk] = self.prev_action[to_walk].clone()
            self.blend_remaining[to_walk] = self.blend_steps

        # 3. Compute walk commands & check arrival (WALK_BACK -> READY_HOME).
        in_walk = self.state == STATE_WALK_BACK
        in_home = self.state == STATE_READY_HOME
        home_cmd = compute_home_velocity_command(local_xy, yaw, pgz)
        self.walk_commands = torch.where(in_walk.unsqueeze(-1), home_cmd, torch.zeros_like(home_cmd))

        dist = torch.norm(local_xy, dim=-1)
        vlin = torch.norm(env.base_lin_vel[:, :2], dim=-1)
        dvel = torch.mean(torch.abs(env.dof_vel), dim=-1)
        arrived_now = (
            in_walk
            & (dist < 0.10)
            & (torch.abs(yaw) < 0.12)
            & (pgz <= -0.94)
            & both_feet
            & (vlin < 0.15)
            & (dvel < 0.80)
        )
        self.arrive_steps = torch.where(arrived_now, self.arrive_steps + 1, torch.zeros_like(self.arrive_steps))
        to_home = in_walk & (self.arrive_steps >= self.arrive_hold_steps)
        if to_home.any():
            self.state[to_home] = STATE_READY_HOME
            self.arrive_steps[to_home] = 0
            self.walk_commands[to_home] = 0.0

        # If robot drifts away while READY_HOME (e.g., push), re-enter WALK_BACK cleanly.
        drifted = in_home & ((dist > 0.18) | (torch.abs(yaw) > 0.22))
        if drifted.any():
            self.state[drifted] = STATE_WALK_BACK

        # 4. Advance gait phase only while actively walking; zero when stationary or in GK mode.
        moving = (self.state == STATE_WALK_BACK) & (torch.norm(self.walk_commands, dim=-1) >= 0.05)
        self.walk_phase = torch.where(
            moving,
            torch.remainder(self.walk_phase + self.dt / self.gait_period, 1.0),
            torch.zeros_like(self.walk_phase),
        )
        angle = 2.0 * math.pi * self.walk_phase
        gait_sc = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1) * moving.float().unsqueeze(-1)

        # 5. Update Policy 2 observation history passively on EVERY step (no cold-start jump at handover).
        step_obs = build_walk_obs_step(
            self.walk_commands,
            gait_sc,
            env.base_ang_vel,
            env.projected_gravity,
            env.dof_pos - env.default_dof_pos,
            env.dof_vel,
            self.prev_action,
        )
        self.walk_obs_buf = torch.cat((self.walk_obs_buf[:, self.walk_one_step_dim :], step_obs), dim=-1)
        if to_walk.any():
            # Ensure clean history right at transition when command switches from 0 to home_cmd.
            self.walk_obs_buf[to_walk] = step_obs[to_walk].repeat(1, self.walk_history_len)

        # 6. Query policies and apply linear blend during handover window.
        with torch.no_grad():
            gk_act = gk_policy(gk_obs.detach())
            walk_act = walk_policy(self.walk_obs_buf.detach())

        use_walk = self.state != STATE_GOALKEEPER
        act = torch.where(use_walk.unsqueeze(-1), walk_act, gk_act)

        blending = use_walk & (self.blend_remaining > 0)
        if blending.any():
            alpha = 1.0 - (self.blend_remaining[blending].float() - 1.0) / max(1, self.blend_steps)
            alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(-1)
            act[blending] = (1.0 - alpha) * self.blend_start_action[blending] + alpha * walk_act[blending]
            self.blend_remaining[blending] -= 1

        self.prev_action = act.detach().clone()
        return act


def load_walk_policy(ckpt_path, device='cuda:0'):
    """Load a trained Policy 2 (k1_walk WalkActorCritic) checkpoint for inference."""
    from rsl_rl.modules.actor_critic import WalkActorCritic

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    hist_in = state['history_encoder.0.weight'].shape[1]
    latent_dim = state['history_encoder.4.weight'].shape[0]
    actor_in = state['actor.0.weight'].shape[1]
    one_step_dim = actor_in - latent_dim
    history_len = hist_in // one_step_dim
    num_actions = state['std'].shape[0]
    critic_in = state['critic.0.weight'].shape[1]
    actor_hidden = [state[f'actor.{2 * i}.weight'].shape[0] for i in range(3)]
    critic_hidden = [state[f'critic.{2 * i}.weight'].shape[0] for i in range(3)]

    model = WalkActorCritic(
        num_actor_obs=hist_in,
        num_critic_obs=critic_in,
        num_one_step_obs=one_step_dim,
        actor_history_length=history_len,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        history_latent_dim=latent_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model.act_inference, {'history_len': history_len, 'one_step_dim': one_step_dim, 'num_actions': num_actions}

