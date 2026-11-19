def ball_crossed_goal(previous, current, line_x, half_width, height, radius, ground_tol=0.015):
    """Detect a full-ball crossing toward negative X in goal-local coordinates.

    Positions are coordinate-first (3, N) tensors, or scalar (x, y, z) tuples.
    line_x is the rear edge of the goal line; width/height are the clear opening.
    ground_tol allows float32 ground contact penetration on rolling balls.
    shortcut: interpolate linearly between samples; sample physics substeps if
    collisions within a control step need to be resolved individually.
    """
    threshold = line_x - radius
    dx = previous[0] - current[0]
    crossed = (previous[0] >= threshold) & (current[0] < threshold)
    # A stationary ball cannot cross; avoid division by zero for those entries.
    fraction = (previous[0] - threshold) / (dx + (dx == 0))
    crossing_y = previous[1] + fraction * (current[1] - previous[1])
    crossing_z = previous[2] + fraction * (current[2] - previous[2])
    inside = (abs(crossing_y) + radius <= half_width)
    inside &= (crossing_z + radius <= height) & (crossing_z >= radius - ground_tol)
    return crossed & inside
