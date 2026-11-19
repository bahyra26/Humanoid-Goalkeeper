"""Paired deterministic & 360-env GPU benchmark for Booster K1 checkpoints.

Separates every shot using an identical no-keeper baseline replay into:
- goal: ball crossed the goal opening
- blocked_save: on-target shot (would score without keeper) deflected/stopped cleanly
- off_target_touched: off-target shot touched by keeper
- out_no_touch: missed/passed without keeper touch (never credited as save)
- unresolved: ball neither conceded, exited, nor clearly stopped safe
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

from legged_gym.envs import *  # noqa: F403,F401
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed
from legged_gym.utils.shot_outcome import shot_outcome_masks


ROOT = Path(__file__).resolve().parents[3]


def _make_kolong_shots():
    rows = []
    sid = 901
    for y in (-0.22, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.22, -0.15, 0.15, 0.02):
        z = 0.16 if abs(y) < 0.15 else 0.14
        t = 0.68
        rows.append({
            'shot_id': sid, 'start_x_m': 3.5, 'start_y_m': 0.0, 'start_z_m': 0.18,
            'target_x_m': 0.08, 'target_y_m': y, 'target_z_m': z,
            'vx_m_s': ((0.08 - 3.5) / t) * 1.04, 'vy_m_s': (y / t) * 1.04,
            'vz_m_s': (z - 0.18 + 0.5 * 9.81 * t * t) / t,
        })
        sid += 1
    return rows


def _make_high_grid_shots():
    rows = []
    sid = 1001
    for spd, sx in [(7.5, 3.7), (10.0, 4.5), (10.5, 4.5), (12.0, 4.5), (14.0, 4.5)]:
        for tz in (0.85, 0.98, 1.12):
            for ty in (0.65, -0.65):
                dx, dy, dz = 0.08 - sx, ty, tz - 0.18
                t = math.hypot(dx, dy) * 1.04 / spd
                rows.append({
                    'shot_id': sid, 'speed_tag': spd, 'start_x_m': sx, 'start_y_m': 0.0, 'start_z_m': 0.18,
                    'target_x_m': 0.08, 'target_y_m': ty, 'target_z_m': tz,
                    'vx_m_s': (dx / t) * 1.04, 'vy_m_s': (dy / t) * 1.04,
                    'vz_m_s': (dz + 0.5 * 9.81 * t * t) / t,
                })
                sid += 1
    return rows


def _settle_reset(env, seed=None, custom_shots=None):
    """Flush GPU PhysX forces and articulation cache before spawning identical test shots."""
    num_envs = env.num_envs
    all_ids = torch.arange(num_envs, device=env.device)
    if hasattr(env, 'ball_force_tensor'):
        env.ball_force_tensor.zero_()
        env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(env.ball_force_tensor))
    env.reset_idx(all_ids)
    env.startstep = 49
    env.step(torch.zeros(num_envs, env.num_actions, device=env.device))
    if hasattr(env, 'ball_force_tensor'):
        env.ball_force_tensor.zero_()
        env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(env.ball_force_tensor))
    if seed is not None:
        set_seed(seed)
    env.reset_idx(all_ids)
    env.startstep = 49
    if custom_shots is not None:
        for i, shot in enumerate(custom_shots):
            start = torch.tensor([shot['start_x_m'], shot['start_y_m'], shot['start_z_m']], device=env.device) + env.env_origins[i]
            target = torch.tensor([shot['target_x_m'], shot['target_y_m'], shot['target_z_m']], device=env.device) + env.env_origins[i]
            env.ball_start[i] = start
            env.ball_end[i] = target
            env.end_target[i] = target
            right = float(shot['target_y_m']) < 0.0
            tz = float(shot['target_z_m'])
            env.end_regions[i] = ((3 if right else 2) if tz > .72 else (5 if right else 4) if tz <= .30 else (1 if right else 0))
            env.ball_states[i, :3] = start
            env.prev_ball_pos[i] = start
            vel = torch.tensor([shot['vx_m_s'], shot['vy_m_s'], shot['vz_m_s']], device=env.device)
            env.ball_states[i, 7:10] = vel
            env.prev_ball_lin_vel[i] = vel
            env.ball_vel[i] = vel[0]
        all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_states))
    init_ball = torch.cat([env.ball_states[:, :3] - env.env_origins, env.ball_states[:, 7:10]], dim=-1).clone()
    return env.get_observations(), init_ball


def _run_batch(env, policy, n_active, seed=None, custom_shots=None, steps=150, nk_traj=None, nk_goals=None, expected_init=None):
    """Run up to `steps` on active envs `[0:n_active]`, capturing trajectories, outcomes, and stability."""
    no_keeper = policy is None
    obs, init_ball = _settle_reset(env, seed=seed, custom_shots=custom_shots)
    if expected_init is not None:
        torch.testing.assert_close(init_ball[:n_active], expected_init[:n_active], atol=1e-5, rtol=1e-5)

    num_envs = env.num_envs
    traj = torch.zeros(steps, num_envs, 3, device=env.device)
    done_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    ever_gl = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    ever_ct = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    ever_fl = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    ever_ov2 = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    term_x = torch.zeros(num_envs, device=env.device)
    term_y = torch.zeros(num_envs, device=env.device)
    term_vx = torch.zeros(num_envs, device=env.device)
    term_spd = torch.zeros(num_envs, device=env.device)
    end_2ft = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    prev_a = torch.zeros(num_envs, env.num_actions, device=env.device)
    daL, daP, lim_v = [], [], []
    max_steps = torch.full((num_envs,), steps, dtype=torch.long, device=env.device)

    cross_goal_yz = torch.full((num_envs, 2), float('nan'), device=env.device)
    cross_step_seen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    # Detailed contact/reach trace for deterministic slots when needed
    kol_cross_info = {}
    high_trace = {i: {'pk_hz': 0.0, 't_pk': -1, 'at_cross': None, 'min_d': 9.0, 't_min': -1} for i in range(n_active)}
    zeros = torch.zeros(num_envs, env.num_actions, device=env.device)

    for s in range(steps):
        prev_ball = (env.ball_states[:, :3] - env.env_origins).clone()
        prev_vel = env.ball_states[:, 7:10].clone()
        prev_ct = env.success_flag > 0
        if no_keeper:
            env.root_states[:, 1] = env.env_origins[:, 1] - 4.0
            env.root_states[:, 7:13] = 0.0
            all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_states))
        with torch.no_grad():
            act = policy(obs.detach()) if not no_keeper else zeros
        obs, _, _, dones, infos = env.step(act.detach())[:5]
        active_step = ~done_mask

        # Snapshot pre-reset terminal results on any env that reset on this step
        if dones.any() and 'shot_results' in infos:
            sr = infos['shot_results']
            eids = sr['env_ids']
            mask_new = active_step[eids]
            if mask_new.any():
                u_ids = eids[mask_new]
                ever_gl[u_ids] |= sr['goal'][mask_new]
                ever_ct[u_ids] |= prev_ct[u_ids] | sr['save'][mask_new]
                ever_fl[u_ids] |= sr['fall'][mask_new]
                term_x[u_ids] = prev_ball[u_ids, 0]
                term_y[u_ids] = prev_ball[u_ids, 1]
                term_vx[u_ids] = prev_vel[u_ids, 0]
                term_spd[u_ids] = torch.norm(prev_vel[u_ids], dim=-1)
                max_steps[u_ids] = s
                done_mask[u_ids] = True

        # Wait: reset_idx already ran for `dones` inside step(), so check live envs (`~done_mask`) directly.
        if active_step.any():
            hl = env.rigid_body_states[:, env.hand_indices[0], :3] - env.env_origins
            hr = env.rigid_body_states[:, env.hand_indices[1], :3] - env.env_origins
            hzh = env.rigid_body_states[:, env.head_body_index, 2] - env.env_origins[:, 2]
            b_local = env.ball_states[:, :3] - env.env_origins
            bx = b_local[:, 0]

            still = ~done_mask
            traj[s, still] = b_local[still]
            ever_gl[still] |= env.goal_conceded_flag[still] > 0
            ever_ct[still] |= env.success_flag[still] > 0
            if s > 5:
                tilt = torch.norm(env.projected_gravity[:, :2], dim=-1) > 0.8
                lowk = torch.min(env.rigid_body_states[:, env.knee_indices, 2], dim=-1).values < 0.08
                ever_fl[still] |= (tilt | lowk)[still]
                over = (env.dof_pos[still] < env.dof_pos_limits[:, 0]) | (env.dof_pos[still] > env.dof_pos_limits[:, 1])
                lim_v.append(float(over.float().mean()))
                da = torch.abs(env.actions - prev_a).mean(dim=-1)
                live = (bx > -0.02) & (~env._k1_ball_safe()) & (env.goal_conceded_flag == 0) & still
                safe = env._k1_ball_safe(include_behind=True) & still
                if live.any():
                    daL.append(float(da[live].mean()))
                if safe.any():
                    daP.append(float(da[safe].mean()))
            prev_a = env.actions.clone()

            live_all = (bx > -0.02) & (~env._k1_ball_safe()) & (env.goal_conceded_flag == 0) & still
            ever_ov2[still] |= (live_all & (hl[:, 2] > hzh - 0.08) & (hr[:, 2] > hzh - 0.08))[still]

            just_crossed = (~cross_step_seen) & (bx <= env.cfg.goal.line_x - env.cfg.goal.ball_radius) & still
            if just_crossed.any():
                cross_goal_yz[just_crossed, 0] = b_local[just_crossed, 1]
                cross_goal_yz[just_crossed, 1] = b_local[just_crossed, 2]
                cross_step_seen |= just_crossed

            if s == steps - 1:
                term_x[still] = b_local[still, 0]
                term_y[still] = b_local[still, 1]
                term_vx[still] = env.ball_states[still, 7]
                term_spd[still] = torch.norm(env.ball_states[still, 7:10], dim=-1)
                end_2ft[still] = torch.all(env.contact_forces[still][:, env.contact_feet_indices, 2] > 1.0, dim=-1)

            if custom_shots is not None and not no_keeper:
                fl = env.rigid_body_states[:, env.contact_feet_indices[0], :3] - env.env_origins
                fr = env.rigid_body_states[:, env.contact_feet_indices[1], :3] - env.env_origins
                for i in range(n_active):
                    if not still[i]:
                        continue
                    shot = custom_shots[i]
                    b = b_local[i]
                    if 901 <= shot['shot_id'] <= 912:
                        fx_front = max(float(fl[i, 0]), float(fr[i, 0]))
                        if i not in kol_cross_info and float(b[0]) <= fx_front:
                            by, bz = float(b[1]), float(b[2])
                            dyl = float(fl[i, 1]) - by
                            dyr = float(fr[i, 1]) - by
                            yz = min(math.hypot(dyl, float(fl[i, 2]) - bz), math.hypot(dyr, float(fr[i, 2]) - bz))
                            kol_cross_info[i] = {'s': s, 'surf_m': yz - 0.145, 'fy': (round(float(fl[i, 1]), 2), round(float(fr[i, 1]), 2)), 'ty': round(float(env.torso_pos[i, 1] - env.env_origins[i, 1]), 2)}
                    if shot['shot_id'] >= 1001:
                        ah = hl[i] if float(shot['target_y_m']) >= 0.0 else hr[i]
                        if float(ah[2]) > high_trace[i]['pk_hz']:
                            high_trace[i]['pk_hz'] = float(ah[2])
                            high_trace[i]['t_pk'] = s
                        d_any = min(float(torch.norm(b - hl[i])), float(torch.norm(b - hr[i])))
                        if d_any < high_trace[i]['min_d']:
                            high_trace[i]['min_d'] = d_any
                            high_trace[i]['t_min'] = s
                        if high_trace[i]['at_cross'] is None and float(b[0]) <= 0.12:
                            high_trace[i]['at_cross'] = {
                                's': s, 'by': round(float(b[1]), 2), 'bz': round(float(b[2]), 2),
                                'hy': round(float(ah[1]), 2), 'hz': round(float(ah[2]), 2),
                                'dy': round(float(ah[1] - b[1]), 2), 'dz': round(float(ah[2] - b[2]), 2),
                            }

    # Physical trajectory divergence vs no-keeper catches high-speed goal-line grazes without false ground bounces
    diverged = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    if nk_traj is not None:
        for i in range(n_active):
            ms = int(max_steps[i].item())
            diff = torch.norm(traj[:ms, i] - nk_traj[:ms, i], dim=-1).max()
            diverged[i] = diff > 0.005

    touched = diverged if nk_traj is not None else ever_ct
    ws = nk_goals if nk_goals is not None else None
    blk, gl_m, out_m, off_t, unr = shot_outcome_masks(
        ever_gl[:n_active], touched[:n_active], term_x[:n_active], term_vx[:n_active],
        env.cfg.goal.line_x, env.cfg.goal.ball_radius,
        y=term_y[:n_active], half_width=env.cfg.goal.half_width, speed=term_spd[:n_active],
        would_score=ws[:n_active] if ws is not None else None,
    )
    return {
        'init_ball': init_ball, 'traj': traj, 'ever_gl': ever_gl, 'touched': touched,
        'blocked_save': blk, 'goal': gl_m, 'out_no_touch': out_m, 'off_target_touched': off_t,
        'unresolved': unr, 'fall': ever_fl[:n_active], 'end_2ft': end_2ft[:n_active],
        'ov2': ever_ov2[:n_active], 'cross_goal_yz': cross_goal_yz[:n_active],
        'daL': sum(daL) / max(len(daL), 1), 'daP': sum(daP) / max(len(daP), 1),
        'lim_pct': 100.0 * sum(lim_v) / max(len(lim_v), 1),
        'kol_cross_info': kol_cross_info, 'high_trace': high_trace,
    }


def _summarize_slice(res, idx_slice, nk_goals_slice, regions=None):
    blk = res['blocked_save'][idx_slice]
    gl = res['goal'][idx_slice]
    out_m = res['out_no_touch'][idx_slice]
    off_t = res['off_target_touched'][idx_slice]
    unr = res['unresolved'][idx_slice]
    fl = res['fall'][idx_slice]
    f2 = res['end_2ft'][idx_slice]
    ov2 = res['ov2'][idx_slice]
    on_t = nk_goals_slice[idx_slice]
    n_on = int(on_t.sum())
    n_blk_on = int((blk & on_t).sum())
    n_sv = int(blk.sum())
    summary = {
        'total': int(blk.numel()),
        'on_target_shots': n_on,
        'off_target_shots': int((~on_t).sum()),
        'blocked_save': n_blk_on,
        'on_target_save_pct': round(100.0 * n_blk_on / max(n_on, 1), 2),
        'goal': int(gl.sum()),
        'out_no_touch': int(out_m.sum()),
        'off_target_touched': int(off_t.sum()),
        'unresolved': int(unr.sum()),
        'fall': int(fl.sum()),
        '2ft_pct_on_blocked': round(100.0 * int((f2 & blk).sum()) / max(n_sv, 1), 1),
        'ov2_sv_pct': round(100.0 * int((ov2 & blk).sum()) / max(n_sv, 1), 1),
        'ov2_fail_pct': round(100.0 * int((ov2 & gl).sum()) / max(int(gl.sum()), 1), 1),
        'daL': round(res['daL'], 4),
        'daP': round(res['daP'], 4),
        'lim_pct': round(res['lim_pct'], 2),
    }
    if regions is not None:
        reg_s = {}
        for r in (4, 0, 2, 5, 1, 3):
            rm = (regions[idx_slice] == r) & on_t
            reg_s[f'R{r}'] = f"{int((blk & rm).sum())}/{int(rm.sum())}"
        summary['per_region_on_target'] = reg_s
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='tag=checkpoint_path entries')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 777])
    parser.add_argument('--output', type=str, default=str(ROOT / 'evaluation/k1_paired_benchmark.json'))
    cli, _ = parser.parse_known_args()

    sys.argv = ['benchmark_k1', '--task=k1', '--headless', '--num_envs=360']
    args = get_args()
    ec, tc = task_registry.get_cfgs('k1')
    ec.env.num_envs = 360
    ec.env.play = True
    ec.env.episode_length_s = 5.0
    ec.terrain.mesh_type = 'plane'
    ec.noise.add_noise = False
    for k in dir(ec.domain_rand):
        if k.startswith('randomize_') or k in ('push_robots', 'delay', 'continue_keep'):
            setattr(ec.domain_rand, k, False)
    env, _ = task_registry.make_env('k1', args=args, env_cfg=ec)
    tc.runner.resume = False
    ppo, _ = task_registry.make_alg_runner(env=env, name='k1', args=args, train_cfg=tc, log_root=None)

    eval_dir = ROOT / 'evaluation'
    s1 = [{k: (int(v) if k == 'shot_id' else float(v)) for k, v in r.items()}
          for r in csv.DictReader((eval_dir / 'k1_sprint1_shots.csv').open()) if r['repeat_seed'] == '0']
    kol = _make_kolong_shots()
    mot = [{k: (int(v) if k == 'shot_id' else float(v)) for k, v in r.items()}
           for r in csv.DictReader((eval_dir / 'k1_motion_shots.csv').open())]
    hg = _make_high_grid_shots()
    det_shots = s1 + kol + mot + hg
    n_det = len(det_shots)
    n1, n2, n3 = len(s1), len(s1) + len(kol), len(s1) + len(kol) + len(mot)

    print(f'=== RUNNING NO-KEEPER BASELINE ({n_det} det + {len(cli.seeds)}x1440 random) ===', flush=True)
    nk_det = _run_batch(env, None, n_det, seed=0, custom_shots=det_shots, steps=150)
    nk_rand = {}
    for seed in cli.seeds:
        nk_rand[seed] = {}
        for mname, spd, sx in [('STD', None, None), ('10ms', 10.0, None), ('12ms_5m', 12.0, 5.0), ('14ms_6m', 14.0, 6.0)]:
            env.cfg.commands.custom_ball_speed = spd
            env.cfg.commands.custom_start_x = sx
            mode_seed = seed * 100 + {'STD': 0, '10ms': 1, '12ms_5m': 2, '14ms_6m': 3}[mname]
            nk_rand[seed][mname] = _run_batch(env, None, 360, seed=mode_seed, steps=150)

    report = {
        'no_keeper': {
            'det': {
                'SPRINT1_18': _summarize_slice(nk_det, slice(0, n1), nk_det['ever_gl'][:n_det]),
                'KOLONG_12': _summarize_slice(nk_det, slice(n1, n2), nk_det['ever_gl'][:n_det]),
                'MOTION_9': _summarize_slice(nk_det, slice(n2, n3), nk_det['ever_gl'][:n_det]),
                'HIGH_GRID_30': _summarize_slice(nk_det, slice(n3, n_det), nk_det['ever_gl'][:n_det]),
            },
            'random': {
                str(seed): {m: {'on_target_goals': int(r['ever_gl'].sum()), 'off_target_out': int(r['out_no_touch'].sum())}
                            for m, r in modes.items()}
                for seed, modes in nk_rand.items()
            },
        },
        'models': {},
    }

    print(f"  NO-KEEPER DET on-target goals: SPRINT1={int(nk_det['ever_gl'][:n1].sum())}/18 | KOLONG={int(nk_det['ever_gl'][n1:n2].sum())}/12 | MOTION={int(nk_det['ever_gl'][n2:n3].sum())}/9 | HIGH_GRID={int(nk_det['ever_gl'][n3:n_det].sum())}/30", flush=True)
    for i in range(n3, n_det):
        sh = det_shots[i]
        if not bool(nk_det['ever_gl'][i]):
            yz = nk_det['cross_goal_yz'][i]
            print(f"    [NK OFF-TARGET] spd={sh['speed_tag']} Z={sh['target_z_m']} y={sh['target_y_m']:+.2f} cross=(y={float(yz[0]):+.2f}, z={float(yz[1]):.3f})", flush=True)

    for entry in cli.models:
        tag, path = entry.split('=', 1)
        ppo.load(path)
        pol = ppo.get_inference_policy(device=env.device)
        print(f'\n=== MODEL {tag} ===', flush=True)
        k_det = _run_batch(env, pol, n_det, seed=0, custom_shots=det_shots, steps=150,
                           nk_traj=nk_det['traj'], nk_goals=nk_det['ever_gl'], expected_init=nk_det['init_ball'])
        det_summary = {}
        for name, sl in [('SPRINT1_18', slice(0, n1)), ('KOLONG_12', slice(n1, n2)), ('MOTION_9', slice(n2, n3)), ('HIGH_GRID_30', slice(n3, n_det))]:
            s = _summarize_slice(k_det, sl, nk_det['ever_gl'][:n_det])
            det_summary[name] = s
            print(f"  {name:12s}: blk={s['blocked_save']}/{s['on_target_shots']} ({s['on_target_save_pct']:.1f}%) goal={s['goal']} out_no_touch={s['out_no_touch']} off_tgt_tch={s['off_target_touched']} unres={s['unresolved']} fl={s['fall']}", flush=True)

        # Print kolong failures & high corner timing diagnostics
        kol_fails = []
        for idx in range(n1, n2):
            if bool(k_det['goal'][idx]) or bool(k_det['unresolved'][idx]):
                sh = det_shots[idx]
                kol_fails.append({'id': sh['shot_id'], 'y': sh['target_y_m'], 'goal': bool(k_det['goal'][idx]), 'diag': k_det['kol_cross_info'].get(idx)})
        print(f"  KOLONG details: {kol_fails}", flush=True)

        high_details = []
        for idx in range(n3, n_det):
            sh = det_shots[idx]
            if float(sh['target_z_m']) == 1.12 and float(sh['speed_tag']) in (10.0, 12.0, 14.0):
                ht = k_det['high_trace'][idx]
                lbl = 'blocked_save' if k_det['blocked_save'][idx] else 'goal' if k_det['goal'][idx] else 'out_no_touch' if k_det['out_no_touch'][idx] else 'off_tgt_tch' if k_det['off_target_touched'][idx] else 'unres'
                high_details.append({'spd': sh['speed_tag'], 'y': sh['target_y_m'], 'nk_goal': bool(nk_det['ever_gl'][idx]), 'outcome': lbl, 'pk_hz': round(ht['pk_hz'], 2), 't_pk': ht['t_pk'], 'min_ctr': round(ht['min_d'], 2), 't_min': ht['t_min'], 'at_cross': ht['at_cross']})
                print(f"  HIGH Z=1.12 spd={sh['speed_tag']:4.1f} y={sh['target_y_m']:+.2f} nk_goal={str(bool(nk_det['ever_gl'][idx])):5s} -> {lbl:12s} | at_cross={ht['at_cross']} | pk_hz={ht['pk_hz']:.2f}@s={ht['t_pk']} min_d={ht['min_d']:.2f}@s={ht['t_min']}", flush=True)

        rand_summary = {}
        for seed in cli.seeds:
            rand_summary[str(seed)] = {}
            for mname, spd, sx in [('STD', None, None), ('10ms', 10.0, None), ('12ms_5m', 12.0, 5.0), ('14ms_6m', 14.0, 6.0)]:
                env.cfg.commands.custom_ball_speed = spd
                env.cfg.commands.custom_start_x = sx
                mode_seed = seed * 100 + {'STD': 0, '10ms': 1, '12ms_5m': 2, '14ms_6m': 3}[mname]
                nk_r = nk_rand[seed][mname]
                k_r = _run_batch(env, pol, 360, seed=mode_seed, steps=150,
                                 nk_traj=nk_r['traj'], nk_goals=nk_r['ever_gl'], expected_init=nk_r['init_ball'])
                ms = _summarize_slice(k_r, slice(0, 360), nk_r['ever_gl'], regions=env.end_regions)
                rand_summary[str(seed)][mname] = ms
                print(f"  [seed={seed}] {mname:7s}: blk={ms['blocked_save']:3d}/{ms['on_target_shots']} ({ms['on_target_save_pct']:5.1f}%) | goal={ms['goal']:3d} out_no_touch={ms['out_no_touch']:2d} off_tgt_tch={ms['off_target_touched']:2d} unres={ms['unresolved']:2d} | fl={ms['fall']} 2ft={ms['2ft_pct_on_blocked']}% lim={ms['lim_pct']}% daP={ms['daP']} ov2_sv={ms['ov2_sv_pct']}% ov2_fl={ms['ov2_fail_pct']}%", flush=True)

        try:
            rel_ckpt = str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            rel_ckpt = path
        report['models'][tag] = {
            'checkpoint_path': rel_ckpt,
            'det': det_summary,
            'kolong_failures': kol_fails,
            'high_corner_timing': high_details,
            'random': rand_summary,
        }

    out_path = Path(cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f'\nSaved report -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
