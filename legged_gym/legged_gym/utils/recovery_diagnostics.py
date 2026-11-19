"""Summarize bounded evaluation traces, without changing reward or save criteria."""
import math


VIOLATIONS = ('low_knee', 'excessive_tilt', 'excessive_foot_force')


def summarize_recovery(frames, dt, standing_height):
    if not frames or dt <= 0 or standing_height <= 0:
        raise ValueError('Recovery summary requires frames and positive timing/height')
    first = next((f for f in frames if any(f[k] for k in VIOLATIONS)), None)
    stable_steps = 0
    first_recovery = None
    needed = math.ceil(0.5 / dt)
    for frame in frames:
        # shortcut: posture sustained for 0.5 s is a recovery proxy, not proof of dynamic stability.
        stable = (not any(frame[k] for k in VIOLATIONS)
                  and frame['upper_gravity_z'] <= -math.cos(math.radians(30))
                  and frame['base_height_m'] >= 0.7 * standing_height)
        stable_steps = stable_steps + 1 if stable else 0
        if first is not None and frame['episode_steps'] > first['episode_steps'] and stable_steps >= needed and first_recovery is None:
            first_recovery = frame['episode_steps'] * dt
    status = 'no_violation_recorded' if first is None else (
        'recovered_proxy' if stable_steps >= needed else 'not_recovered_by_horizon')
    return {
        'stability_status': status,
        'first_violation_time_s': None if first is None else first['episode_steps'] * dt,
        'first_violation_reasons': [] if first is None else [k for k in VIOLATIONS if first[k]],
        'violation_counts': {k: sum(bool(f[k]) for f in frames) for k in VIOLATIONS},
        'first_recovery_time_s': first_recovery,
        'stable_tail_s': stable_steps * dt,
        'contact_observed_any_step': any(f['contact_flag_at_end'] for f in frames),
        'first_violation_frame': first,
        'frame_at_original_horizon': next((f for f in frames if f['episode_steps'] * dt > 5.0), None),
        'recovery_proxy': {'stable_duration_s': 0.5, 'max_upper_tilt_deg': 30,
                           'min_base_height_m': 0.7 * standing_height},
    }
