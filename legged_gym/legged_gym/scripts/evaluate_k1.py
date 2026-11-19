"""Evaluate every CSV shot once with deterministic policy actions and fixed launch states."""
import csv
import json
import hashlib
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import isaacgym
from isaacgym import gymtorch
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, set_seed
from legged_gym.utils.recovery_diagnostics import summarize_recovery
from legged_gym.utils.shot_outcome import shot_outcome_label


def reset_env_clean(env, shot, seed=0):
    """Flush residual GPU PhysX forces and articulation cache before a deterministic shot."""
    set_seed(seed)
    env.evaluation_shot = shot
    if hasattr(env, 'ball_force_tensor'):
        env.ball_force_tensor.zero_()
        env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(env.ball_force_tensor))
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.startstep = 49
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.step(zero_actions)
    if hasattr(env, 'ball_force_tensor'):
        env.ball_force_tensor.zero_()
        env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(env.ball_force_tensor))
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.startstep = 49
    return env.get_observations()


def replay_without_keeper(env, shot, seed=0):
    """Replay the identical shot with the keeper shifted aside to check if it would score."""
    obs = reset_env_clean(env, shot, seed=seed)
    zeros = torch.zeros(1, env.num_actions, device=env.device)
    trajectory = []
    cross_yz = (None, None)
    would_score = False
    for _ in range(int(env.max_episode_length) + 2):
        env.root_states[0, 1] = env.env_origins[0, 1] - 4.0
        env.root_states[0, 7:13] = 0.0
        all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        ids = torch.tensor([0], dtype=torch.int32, device=env.device)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(all_states), gymtorch.unwrap_tensor(ids), 1
        )
        obs, _, _, dones, infos, _, _ = env.step(zeros)
        if dones[0].item():
            sr = infos['shot_results']
            would_score = bool(sr['goal'][0].item())
            pos = torch.tensor([sr['final_ball_x_m'][0].item(), sr['final_ball_y_m'][0].item(), sr['final_ball_z_m'][0].item()], device=env.device)
            trajectory.append(pos)
            if cross_yz[0] is None and float(pos[0]) <= env.cfg.goal.line_x - env.cfg.goal.ball_radius:
                cross_yz = (float(pos[1]), float(pos[2]))
            break
        pos = (env.ball_states[0, :3] - env.env_origins[0]).clone()
        trajectory.append(pos)
        if cross_yz[0] is None and float(pos[0]) <= env.cfg.goal.line_x - env.cfg.goal.ball_radius:
            cross_yz = (float(pos[1]), float(pos[2]))
    return would_score, torch.stack(trajectory, dim=0), cross_yz


def select_shots(rows, shot_id):
    if shot_id is None:
        return rows
    selected = [row for row in rows if int(row['shot_id']) == shot_id]
    if not selected:
        raise ValueError(f'Shot ID {shot_id} not found in shot CSV')
    return selected


def main():
    args = get_args([{'name': '--shot_file', 'type': str, 'default': None,
                      'help': 'CSV with the same columns as k1_sprint1_shots.csv.'},
                     {'name': '--shot_id', 'type': int, 'default': None,
                      'help': 'Run only this CSV shot ID; use --gui to watch.'},
                     {'name': '--trace_motion', 'action': 'store_true',
                      'help': 'Record policy input, actions, joints and torso before each step.'},
                     {'name': '--limit_joint_targets', 'action': 'store_true',
                      'help': 'Test the existing URDF joint-target guard; simulation only.'},
                     {'name': '--observe_recovery', 'action': 'store_true',
                      'help': 'Observe K1 for 8 seconds despite posture thresholds; save per-step trace.'},
                     {'name': '--no_paired', 'action': 'store_true',
                      'help': 'Skip headless no-keeper baseline replay used for on-target outcome split.'}])
    if args.ball_speed is not None:
        raise ValueError('CSV evaluation uses vx/vy/vz from the shot file; use play_k1.py for --ball_speed')
    args.task = 'k1'
    if args.exptid is None:
        args.exptid = 'k1_v9_s1_landfix'
    args.num_envs = 1
    root = Path(LEGGED_GYM_ROOT_DIR).parent
    shot_file = Path(args.shot_file) if args.shot_file else root / 'evaluation/k1_sprint1_shots.csv'
    with shot_file.open() as shot_stream:
        rows = list(csv.DictReader(shot_stream))
    rows = select_shots(rows, args.shot_id)
    output = root / 'evaluation/k1_diagnostic_results.jsonl'
    if args.shot_id is not None or args.gui or args.observe_recovery or args.trace_motion or args.shot_file:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        mode = 'recovery' if args.observe_recovery else 'review'
        output = root / f'evaluation/k1_{mode}_{stamp}.jsonl'
    if output.exists():
        raise FileExistsError(f'Results already exist: {output}. Archive them before rerunning.')
    cfg, train_cfg = task_registry.get_cfgs(name='k1')
    cfg.env.num_envs = 1
    cfg.env.play = True
    cfg.env.observe_recovery = args.observe_recovery
    cfg.env.episode_length_s = 8.0 if args.observe_recovery else 5.0
    cfg.noise.add_noise = False
    cfg.control.limit_joint_targets = args.limit_joint_targets
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_') or name in ('push_robots', 'delay', 'continue_keep'):
            setattr(cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(name='k1', args=args, env_cfg=cfg)
    if env.viewer is not None:
        origin = env.env_origins[0].cpu().tolist()
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            isaacgym.gymapi.Vec3(origin[0] + 3.5, origin[1] + 2.5, origin[2] + 1.5),
            isaacgym.gymapi.Vec3(origin[0] + 0.5, origin[1], origin[2] + 0.5))
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name='k1', args=args, train_cfg=train_cfg)
    checkpoint_path = Path(runner.loaded_checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    policy = runner.get_inference_policy(device=env.device)
    # Exclusive creation protects previous measured results.
    with ExitStack() as stack:
        stream = stack.enter_context(output.open('x'))
        if args.trace_motion:
            motion_path = output.with_suffix('.motion.jsonl')
            motion_stream = stack.enter_context(motion_path.open('x'))
            print(f'Motion trace: {motion_path}', flush=True)
        if args.observe_recovery:
            trace_path = output.with_suffix('.trace.jsonl')
            trace_stream = stack.enter_context(trace_path.open('x'))
            print(f'Recovery observation: 8 s; posture thresholds logged, not used to reset. Trace: {trace_path}')
        for row in rows:
            frames = []
            print(f"Starting shot ID {row['shot_id']}: target y={row['target_y_m']}, z={row['target_z_m']} m", flush=True)
            if args.gui:
                print('Starting automatically. Use SPACE in the viewer to pause/resume.', flush=True)
            shot = {key: float(value) for key, value in row.items()}
            shot_seed = int(shot['repeat_seed'])
            would_score = None
            nk_traj = None
            nk_cross_yz = (None, None)
            if (not args.gui) and (not args.no_paired):
                would_score, nk_traj, nk_cross_yz = replay_without_keeper(env, shot, seed=shot_seed)
            obs = reset_env_clean(env, shot, seed=shot_seed)
            keeper_traj = []
            for _ in range(int(env.max_episode_length) + 2):
                with torch.no_grad():
                    actions = policy(obs)
                    if args.trace_motion:
                        # Capture before step: terminal step automatically resets robot state.
                        history = obs.reshape(1, cfg.env.num_actor_history, env.num_one_step_obs)
                        motion = {
                            'shot_id': row['shot_id'], 'checkpoint': checkpoint_path.name,
                                   'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
                                   'limit_joint_targets': cfg.control.limit_joint_targets,
                            'time_s': env.episode_length_buf[0].item() * env.dt,
                            'joint_names': env.dof_names,
                            'ball_observation_history': history[0, :, :3].tolist(),
                            'estimated_region': runner.alg.actor_critic.region_estimator(obs).argmax(-1).item(),
                            'action': actions[0].tolist(),
                            'position': env.dof_pos[0].tolist(),
                            'velocity': env.dof_vel[0].tolist(),
                            'torso_quaternion': env.rigid_body_states[0, env.torso_index, 3:7].tolist(),
                            'hand_position': (env.rigid_body_states[0, env.hand_indices, :3] - env.env_origins[0]).tolist(),
                            'ball_position': (env.ball_states[0, :3] - env.env_origins[0]).tolist(),
                            'contact_detected': bool(env.stop_flag[0].item()),
                        }
                    obs, _, _, dones, infos, _, _ = env.step(actions)
                    if dones[0].item():
                        sr = infos['shot_results']
                        keeper_traj.append(torch.tensor(
                            [sr['final_ball_x_m'][0].item(), sr['final_ball_y_m'][0].item(), sr['final_ball_z_m'][0].item()],
                            device=env.device,
                        ))
                    else:
                        keeper_traj.append((env.ball_states[0, :3] - env.env_origins[0]).clone())
                    if args.trace_motion:
                        motion['applied_target'] = env.joint_pos_target[0].tolist()
                        motion['torque_last_substep'] = env.torques[0].tolist()
                        motion['terminal_step'] = bool(dones[0].item())
                        motion_stream.write(json.dumps(motion, allow_nan=False) + '\n')
                if args.observe_recovery:
                    frame = {key: value[0].item() for key, value in infos['evaluation_frame'].items()}
                    frame['shot_id'] = row['shot_id']
                    frame['time_s'] = frame['episode_steps'] * env.dt
                    trace_stream.write(json.dumps(frame, allow_nan=False) + '\n')
                    trace_stream.flush()
                    frames.append(frame)
                if dones[0].item():
                    result = infos['shot_results']
                    record = dict(row)
                    record.update({'experiment': args.exptid, 'checkpoint': checkpoint_path.name,
                                   'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
                                   'limit_joint_targets': cfg.control.limit_joint_targets,
                                   'selected_shot_id': args.shot_id})
                    record.update({key: value[0].item() for key, value in result.items() if key != 'env_ids'})
                    record['episode_time_s'] = record['episode_steps'] * env.dt
                    record['episode_rewards'] = {
                        key[4:]: value.item() * record['episode_time_s']
                        for key, value in infos['episode'].items() if key.startswith('rew_')
                    }
                    record['termination_reasons'] = [key for key in ('timeout', 'low_knee', 'excessive_tilt', 'excessive_foot_force') if record[key]]
                    record['ball_fully_behind_goal_line'] = record['final_ball_x_m'] < cfg.goal.line_x - cfg.goal.ball_radius
                    record['ball_returning_to_field'] = record['final_ball_x_m'] > 2.0 and record['final_ball_vx_m_s'] > 0.0
                    record['observe_recovery'] = args.observe_recovery
                    # Legacy fall was only a non-timeout termination, not an observed fall.
                    record['non_timeout_termination'] = record.pop('fall')
                    max_traj_diff = None
                    diverged = False
                    if nk_traj is not None:
                        steps_cmp = min(len(keeper_traj), len(nk_traj))
                        k_stack = torch.stack(keeper_traj[:steps_cmp], dim=0)
                        max_traj_diff = float(torch.norm(k_stack - nk_traj[:steps_cmp], dim=-1).max().item())
                        diverged = max_traj_diff > 0.005
                    touched = bool(record['contact_flag_at_end']) or diverged
                    ball_spd = float((record['final_ball_vx_m_s'] ** 2 + record['final_ball_vy_m_s'] ** 2 + record['final_ball_vz_m_s'] ** 2) ** 0.5)
                    outcome = shot_outcome_label(
                        bool(record['goal']), touched, record['final_ball_x_m'], record['final_ball_vx_m_s'],
                        cfg.goal.line_x, cfg.goal.ball_radius, y=record['final_ball_y_m'],
                        half_width=cfg.goal.half_width, speed=ball_spd, would_score=would_score,
                    )
                    record.update({
                        'outcome': outcome,
                        'would_score': would_score,
                        'touched': touched,
                        'trajectory_diverged': diverged,
                        'max_ball_divergence_m': max_traj_diff,
                        'no_keeper_cross_y_m': nk_cross_yz[0],
                        'no_keeper_cross_z_m': nk_cross_yz[1],
                    })
                    if args.observe_recovery:
                        record.update(summarize_recovery(frames, env.dt, cfg.init_state.pos[2]))
                        record['trace_file'] = str(trace_path)
                        print(f"Stability: {record['stability_status']}; first violation={record['first_violation_time_s']}; "
                              f"first recovery={record['first_recovery_time_s']}")
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    print(f"Shot ID {row['shot_id']}: {outcome} (would_score={would_score}); stop={','.join(record['termination_reasons'])}; "
                          f"ball=({record['final_ball_x_m']:.2f}, {record['final_ball_y_m']:.2f}, {record['final_ball_z_m']:.2f}) m; "
                          f"touched={touched} (ct={record['contact_flag_at_end']}, div={diverged})")
                    break
            else:
                raise RuntimeError('Episode exceeded evaluation horizon without terminal record')
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
