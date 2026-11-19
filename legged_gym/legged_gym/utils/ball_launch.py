"""Ballistic launch at a requested initial speed (before drag and contacts)."""
import math
import torch


def launch_at_speed(delta, speed, gravity=9.81):
    """Return low-arc velocity for an (N, 3) displacement tensor, in SI units."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('ball_speed must be finite and greater than zero (m/s)')
    distance_sq = torch.sum(delta.square(), dim=-1, keepdim=True)
    if not torch.isfinite(delta).all() or (distance_sq <= 0).any():
        raise ValueError('Ball start and target must be finite and distinct')
    height = delta[:, 2:3]
    minimum = torch.sqrt(gravity * (height + torch.sqrt(distance_sq)))
    if (speed < minimum).any():
        raise ValueError(f'ball_speed={speed:g} m/s cannot reach this target; '
                         f'need at least {minimum.max().item():.3f} m/s without drag. '
                         'Increase speed or move the launch point closer.')
    a = speed * speed - gravity * height
    discriminant = torch.clamp(a.square() - gravity * gravity * distance_sq, min=0)
    time = torch.sqrt(2 * distance_sq / (a + torch.sqrt(discriminant)))
    velocity = delta / time
    velocity[:, 2:3] += .5 * gravity * time
    return velocity
