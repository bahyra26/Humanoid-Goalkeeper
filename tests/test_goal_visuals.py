import importlib.util
from pathlib import Path
import unittest
import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "legged_gym/legged_gym/utils/goal_visuals.py"
_SPEC = importlib.util.spec_from_file_location("goal_visuals", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class GoalVisualsTests(unittest.TestCase):
    def test_geometry_matches_goal_opening_and_net_bounds(self):
        frame, net = _MOD.build_goal_geometry(
            line_x=-0.05,
            half_width=1.30,
            height=1.20,
            ball_radius=0.10,
            net_depth=0.65,
        )
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(net.dtype, np.float32)
        self.assertEqual(frame.ndim, 2)
        self.assertEqual(frame.shape[1], 6)
        self.assertEqual(net.shape[1], 6)
        self.assertGreater(len(frame), 30)
        self.assertGreater(len(net), 40)

        all_pts = np.vstack([frame.reshape(-1, 3), net.reshape(-1, 3)])
        self.assertTrue(np.isfinite(all_pts).all())
        # Net rear sits at -0.70 m, goal line at -0.05 m, threshold line at -0.15 m.
        self.assertAlmostEqual(float(all_pts[:, 0].min()), -0.70, places=4)
        self.assertLessEqual(float(all_pts[:, 0].max()), 0.01)
        self.assertGreaterEqual(float(all_pts[:, 2].max()), 1.20)
        # Clear opening threshold line (-0.15, +/-1.30, ground) is included.
        has_thresh = any(
            abs(seg[0] + 0.15) < 1e-5 and abs(seg[3] + 0.15) < 1e-5 and abs(seg[1] + 1.30) < 1e-5 and abs(seg[4] - 1.30) < 1e-5
            for seg in frame
        )
        self.assertTrue(has_thresh)

    def test_status_colors_and_live_crosshair(self):
        frame, net = _MOD.build_goal_geometry()
        live_v, live_c = _MOD.make_goal_lines(frame, net, status="live", line_x=-0.05, target_yz=(0.40, 0.85))
        self.assertEqual(len(live_v), len(frame) + len(net) + 2)
        self.assertEqual(live_v.shape, (len(live_v), 6))
        self.assertEqual(live_c.shape, (len(live_v), 3))
        self.assertTrue(live_v.flags["C_CONTIGUOUS"])
        self.assertTrue(live_c.flags["C_CONTIGUOUS"])

        _, goal_c = _MOD.make_goal_lines(frame, net, status="goal", target_yz=(0.40, 0.85))
        _, save_c = _MOD.make_goal_lines(frame, net, status="save", target_yz=(0.40, 0.85))
        _, out_c = _MOD.make_goal_lines(frame, net, status="out")
        self.assertEqual(len(goal_c), len(frame) + len(net))
        # Goal is distinctly red-dominant; save is green-dominant; out is cyan/blue-dominant.
        self.assertGreater(goal_c[0, 0], goal_c[0, 1] * 3.0)
        self.assertGreater(save_c[0, 1], save_c[0, 0] * 3.0)
        self.assertGreater(out_c[0, 2], out_c[0, 0] * 2.0)

        off_v, _ = _MOD.make_goal_lines(frame, net, status="live", offset_xyz=(-4.0, -2.0, 0.0))
        self.assertAlmostEqual(float(off_v[0, 0] - live_v[0, 0]), -4.0, places=5)
        self.assertAlmostEqual(float(off_v[0, 1] - live_v[0, 1]), -2.0, places=5)


if __name__ == "__main__":
    unittest.main()
