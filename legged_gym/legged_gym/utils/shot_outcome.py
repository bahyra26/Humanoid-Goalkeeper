"""Conservative terminal labels; accepts Python scalars or tensor masks."""


def resolved_save(contact, conceded, x, vx, line_x, radius, y=None, half_width=None, speed=None):
    # A contact alone, or a ball still on/approaching the goal line, remains unresolved.
    escaped = (x < line_x - radius) | ((x > line_x + radius) & (vx >= 0.0))
    if y is not None and half_width is not None:
        escaped = escaped | ((abs(y) > half_width + radius) & (vx >= -0.25))
    if speed is not None:
        escaped = escaped | ((x > line_x + radius + 0.15) & (speed < 0.15))
    return contact & (conceded == False) & escaped


def contact_candidate(z, distance, dv_x, dv_y, dv_xy, force, ball_radius=0.10):
    """Conservative force/proximity/impulse heuristic scaled to ball radius."""
    r = float(ball_radius)
    air = (z > r + 0.025) & (distance < r + 0.14) & ((dv_x > 0.9) | (dv_xy > 1.2))
    low = (z <= r + 0.025) & (distance < r + 0.075) & ((dv_x > 1.8) | (dv_y > 1.0))
    return (force > 1.0) & (air | low)


def shot_outcome_masks(conceded, touched, x, vx, line_x, radius, y=None, half_width=None, speed=None, would_score=None):
    """Partition outcomes into (blocked_save, goal, out_no_touch, off_target_touched, unresolved)."""
    not_conceded = conceded == False
    not_touched = touched == False
    any_save = resolved_save(touched, conceded, x, vx, line_x, radius, y=y, half_width=half_width, speed=speed)
    exited = x < line_x - radius
    if y is not None and half_width is not None:
        exited = exited | (abs(y) > half_width + radius)
    if speed is not None:
        exited = exited | ((x > line_x + radius + 0.15) & (speed < 0.15))
    out_no_touch = not_conceded & not_touched & exited
    if would_score is None:
        blocked_save = any_save
        off_target_touched = any_save & False
    else:
        blocked_save = any_save & (would_score != False)
        off_target_touched = any_save & (would_score == False)
    unresolved = not_conceded & (blocked_save == False) & (off_target_touched == False) & (out_no_touch == False)
    return blocked_save, conceded, out_no_touch, off_target_touched, unresolved


def shot_outcome_label(conceded, touched, x, vx, line_x, radius, y=None, half_width=None, speed=None, would_score=None):
    """Return single-shot string outcome label."""
    blocked_save, goal, out_no_touch, off_target_touched, _ = shot_outcome_masks(
        conceded, touched, x, vx, line_x, radius, y=y, half_width=half_width, speed=speed, would_score=would_score
    )
    if goal:
        return 'goal'
    if blocked_save:
        return 'blocked_save'
    if off_target_touched:
        return 'off_target_touched'
    if out_no_touch:
        return 'out_no_touch'
    return 'unresolved'
