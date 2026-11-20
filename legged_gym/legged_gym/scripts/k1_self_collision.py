"""Offline collision correction against the K1 URDF used by the simulator."""
import numpy as np
import pybullet as pb


class K1CollisionCheck:
    def __init__(self, urdf, names):
        self.client = pb.connect(pb.DIRECT)
        self.robot = pb.loadURDF(str(urdf), useFixedBase=True, physicsClientId=self.client,
                                 flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        joints = [pb.getJointInfo(self.robot, i, physicsClientId=self.client)
                  for i in range(pb.getNumJoints(self.robot, physicsClientId=self.client))]
        indices = {joint[1].decode(): joint[0] for joint in joints}
        self.active = [indices[name] for name in names]
        self.roll = [names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
        self.links = {joint[0]: joint[12].decode() for joint in joints}
        self.links[-1] = "trunk"
        pb.resetJointState(self.robot, indices["aahead_pitch_joint"], 0.15, physicsClientId=self.client)

    def close(self):
        pb.disconnect(self.client)

    def collisions(self, pose):
        for joint, angle in zip(self.active, pose):
            pb.resetJointState(self.robot, joint, angle, physicsClientId=self.client)
        pb.performCollisionDetection(physicsClientId=self.client)
        return [(self.links[c[3]], self.links[c[4]]) for c in
                pb.getContactPoints(self.robot, self.robot, physicsClientId=self.client) if c[8] < 0]

    def correct(self, positions, lower, upper, max_step):
        # Work on the exported precision; AMP interpolates between the 30 Hz samples.
        original = np.asarray(positions, dtype=np.float32)
        result = original.copy()
        frames = np.arange(len(result))
        signs = np.array([1., -1.])
        offset = np.zeros((len(result), 2))
        for _ in range(32):
            changed = False
            for index in range(4 * (len(result) - 1) + 1):
                frame, substep = divmod(index, 4)
                t = substep / 4
                pose = result[frame] * (1 - t) + result[min(frame + 1, len(result) - 1)] * t
                pairs = self.collisions(pose)
                if not pairs:
                    continue
                changed = True
                for pair in pairs:
                    arms = [side for side, prefix in enumerate(("left_", "right_"))
                            if any(link.startswith(prefix + "elbow") or link == prefix + "hand_link"
                                   for link in pair)]
                    if not arms:
                        raise ValueError(f"Unresolved self-collision at frame {frame}: {pair}")
                    # Spread small outward arm corrections over neighbouring frames.
                    taper = np.maximum(0., 1 - abs(frames - (frame + t)) / 8)
                    for arm in arms:
                        current = (pose[self.roll[arm]] - np.interp(frame + t, frames, original[:, self.roll[arm]])) * signs[arm]
                        offset[:, arm] = np.maximum(offset[:, arm], (current + 0.045) * taper)
            if not changed:
                return result
            angles = original[:, self.roll] * signs + offset
            # Propagate outward corrections to neighbours to respect the velocity bounds.
            for i in range(1, len(angles)):
                angles[i] = np.maximum(angles[i], angles[i-1] - max_step[self.roll])
            for i in range(len(angles)-2, -1, -1):
                angles[i] = np.maximum(angles[i], angles[i+1] - max_step[self.roll])
            result[:, self.roll] = np.clip(angles * signs, lower[self.roll], upper[self.roll])
        raise ValueError("K1 clip still has self-collisions after shoulder correction")
