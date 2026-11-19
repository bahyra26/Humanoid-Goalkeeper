import numpy as np


_FRAME_COLORS = {
    "live": (0.96, 0.96, 0.98),
    "goal": (1.00, 0.15, 0.15),
    "save": (0.15, 0.95, 0.35),
    "out": (0.30, 0.82, 0.95),
}

_NET_COLORS = {
    "live": (0.60, 0.74, 0.90),
    "goal": (0.95, 0.22, 0.22),
    "save": (0.28, 0.80, 0.45),
    "out": (0.38, 0.72, 0.84),
}


def _box_post(segs, x0, x1, y0, y1, z0, z1):
    """Append 12 edges of an axis-aligned rectangular bar (thicker visual post)."""
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    for a, b in (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ):
        segs.append((*corners[a], *corners[b]))


def build_goal_geometry(
    line_x=-0.05,
    half_width=1.30,
    height=1.20,
    ball_radius=0.10,
    net_depth=0.65,
    post_size=0.045,
    net_step=0.13,
):
    """Build static 3D goal frame + net mesh segments matching ball_crossed_goal geometry.

    Inner clear opening sits at x=line_x, y in [-half_width, +half_width], z in [0, height].
    Returns (frame_verts, net_verts) as float32 arrays shaped (N, 6) in env-local coords.
    """
    lx = float(line_x)
    hw = float(half_width)
    hz = float(height)
    bx = lx - float(net_depth)
    ps = float(post_size)
    z_ground = 0.004

    frame = []
    # Left (+Y) and right (-Y) vertical posts drawn outside the clear opening.
    _box_post(frame, lx, lx + ps, hw, hw + ps, z_ground, hz + ps)
    _box_post(frame, lx, lx + ps, -(hw + ps), -hw, z_ground, hz + ps)
    # Crossbar above clear height.
    _box_post(frame, lx, lx + ps, -hw, hw, hz, hz + ps)
    # Ground goal line and full-ball crossing threshold line (line_x - ball_radius).
    frame.append((lx, -hw, z_ground, lx, hw, z_ground))
    thresh_x = lx - float(ball_radius)
    frame.append((thresh_x, -hw, z_ground, thresh_x, hw, z_ground))
    # Rear frame & ground base bars.
    for sy in (-hw, hw):
        frame.append((lx, sy, hz, bx, sy, hz))
        frame.append((lx, sy, z_ground, bx, sy, z_ground))
        frame.append((bx, sy, z_ground, bx, sy, hz))
    frame.append((bx, -hw, z_ground, bx, hw, z_ground))
    frame.append((bx, -hw, hz, bx, hw, hz))

    net = []
    n_y = max(4, int(round((2.0 * hw) / net_step)))
    n_z = max(3, int(round(hz / net_step)))
    n_x = max(2, int(round(float(net_depth) / net_step)))
    ys = np.linspace(-hw, hw, n_y + 1)
    zs = np.linspace(z_ground, hz, n_z + 1)
    xs = np.linspace(bx, lx, n_x + 1)

    # Back net panel (x = bx)
    for y in ys[1:-1]:
        net.append((bx, y, z_ground, bx, y, hz))
    for z in zs[1:-1]:
        net.append((bx, -hw, z, bx, hw, z))

    # Left and right side net panels (y = ±hw)
    for sy in (-hw, hw):
        for x in xs[1:-1]:
            net.append((x, sy, z_ground, x, sy, hz))
        for z in zs[1:-1]:
            net.append((bx, sy, z, lx, sy, z))

    # Top roof net panel (z = hz)
    for x in xs[1:-1]:
        net.append((x, -hw, hz, x, hw, hz))
    for y in ys[1:-1]:
        net.append((bx, y, hz, lx, y, hz))

    return np.asarray(frame, dtype=np.float32), np.asarray(net, dtype=np.float32)


def make_goal_lines(frame_verts, net_verts, status="live", line_x=-0.05, target_yz=None, offset_xyz=None):
    """Combine goal frame, net, and optional live target crosshair into (verts, colors)."""
    st = status if status in _FRAME_COLORS else "live"
    fc = _FRAME_COLORS[st]
    nc = _NET_COLORS[st]

    n_frame = len(frame_verts)
    n_net = len(net_verts)
    verts_list = [frame_verts, net_verts]
    colors_list = [
        np.broadcast_to(np.asarray(fc, dtype=np.float32), (n_frame, 3)),
        np.broadcast_to(np.asarray(nc, dtype=np.float32), (n_net, 3)),
    ]

    if target_yz is not None and st == "live":
        ty, tz = float(target_yz[0]), max(0.05, float(target_yz[1]))
        r = 0.06
        xhair = np.asarray(
            [
                [line_x, ty - r, tz, line_x, ty + r, tz],
                [line_x, ty, max(0.005, tz - r), line_x, ty, tz + r],
            ],
            dtype=np.float32,
        )
        verts_list.append(xhair)
        colors_list.append(np.tile(np.asarray((1.0, 0.82, 0.12), dtype=np.float32), (2, 1)))

    verts = np.ascontiguousarray(np.vstack(verts_list), dtype=np.float32)
    if offset_xyz is not None:
        ox, oy, oz = float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2])
        if ox != 0.0 or oy != 0.0 or oz != 0.0:
            verts = verts + np.asarray([ox, oy, oz, ox, oy, oz], dtype=np.float32)
    colors = np.ascontiguousarray(np.vstack(colors_list), dtype=np.float32)
    return verts, colors
