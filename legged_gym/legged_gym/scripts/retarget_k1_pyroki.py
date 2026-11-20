"""Retarget the six G1 goalkeeper clips to K1 for DOF AMP training.

Run offline with Python 3.12, PyTorch CPU, PyBullet and PyRoki installed. The training
environment only needs the generated .pt files and joint_id.txt.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki
from scipy.optimize import least_squares
import torch
from yourdfpy import URDF

from k1_self_collision import K1CollisionCheck


ROOT = Path(__file__).resolve().parents[2] / "resources"
FPS = 30
CLIPS = ("lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep")


def load_model(robot, urdf):
    model = pyroki.Robot.from_urdf(URDF.load(
        str(ROOT / "robots" / robot / "urdf" / urdf), build_scene_graph=True,
        load_meshes=False))
    indices = []
    feet = []
    for side in ("left", "right"):
        shoulder = f"{side}_shoulder_pitch_link"
        arm = ([shoulder, f"{side}_elbow_link", f"{side}_rubber_hand"] if robot == "g1"
               else ["aa" + shoulder, f"{side}_elbow_yaw_link", f"{side}_hand_link"])
        knee = "knee" if robot == "g1" else "knee_pitch"
        leg = [f"{side}_hip_pitch_link", f"{side}_{knee}_link", f"{side}_ankle_roll_link"]
        indices.extend(model.links.names.index(n) for n in arm + leg)
        feet.append(model.links.names.index(leg[-1]))

    def features(q):
        pose = model.forward_kinematics(q)
        points = pose[jnp.array(indices), 4:].reshape(4, 3, 3)
        bones = jnp.diff(points, axis=1)
        foot_axes = jaxlie.SO3(pose[jnp.array(feet), :4]).as_matrix()[..., :2]
        return bones, foot_axes

    return model, features


def main():
    jax.config.update("jax_enable_x64", True)
    source, source_features = load_model("g1", "g1_23.urdf")
    target, target_features = load_model("k1", "k1.urdf")
    source_dir = ROOT / "datasets/goalkeeper"
    output = ROOT / "datasets/g1_to_k1_retargeted"
    columns = {name: int(i) for i, name in (line.split() for line in
               (source_dir / "joint_id.txt").read_text().splitlines())}
    source_indices = [source.joints.actuated_names.index(name) for name in columns]
    names = [n for n in target.joints.actuated_names if "head" not in n]
    active = np.array([target.joints.actuated_names.index(n) for n in names])
    neutral = jnp.zeros(target.joints.num_actuated_joints)
    target_lengths = np.linalg.norm(target_features(neutral)[0], axis=-1, keepdims=True)
    lower = np.asarray(target.joints.lower_limits)[active]
    upper = np.asarray(target.joints.upper_limits)[active]
    max_step = np.asarray(target.joints.velocity_limits)[active] / FPS
    source_batch = jax.jit(jax.vmap(source_features))

    def residual(q, bones, feet, previous):
        actual, axes = target_features(neutral.at[active].set(q))
        return jnp.concatenate(((actual - bones).ravel(),
                                0.08 * (axes - feet).ravel(),
                                0.015 * (q - previous)))

    evaluate = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    measure = jax.jit(jax.vmap(lambda q: target_features(neutral.at[active].set(q))[0]))
    clips, report = {}, {}
    collision = K1CollisionCheck(ROOT / "robots/k1/urdf/k1.urdf", names)
    for clip in CLIPS:
        data = torch.load(source_dir / (clip + ".pt"), map_location="cpu", weights_only=True)
        raw = data["joint_position"].numpy()
        if raw.shape != (len(raw), len(columns)) or len(raw) < 4 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid source positions in {clip}")
        full = np.zeros((len(raw), source.joints.num_actuated_joints))
        full[:, source_indices] = raw[:, list(columns.values())]
        bones, feet = map(np.asarray, source_batch(full))
        # Match segment directions using K1 limb lengths, in the pelvis frame.
        goals = bones * target_lengths / np.linalg.norm(bones, axis=-1, keepdims=True)
        q = np.zeros(len(names))
        for side, sign in (("left", -1), ("right", 1)):
            q[names.index(f"{side}_elbow_yaw_joint")] = sign * 0.65
            q[names.index(f"{side}_shoulder_roll_joint")] = sign * 0.35
        positions = []
        for frame, goal in enumerate(goals):
            lo = np.maximum(lower, q - max_step) if frame else lower
            hi = np.minimum(upper, q + max_step) if frame else upper
            fit = least_squares(evaluate, q, jac=jacobian, bounds=(lo, hi),
                                args=(goal, feet[frame], q), ftol=1e-5)
            if not fit.success:
                raise ValueError(f"{clip} frame {frame}: {fit.message}")
            q = fit.x
            positions.append(q)
        positions = collision.correct(positions, lower, upper, max_step)
        velocity = np.diff(positions, axis=0) * FPS
        errors = np.linalg.norm((measure(positions) - goals).cumsum(axis=2), axis=-1)
        if (not np.isfinite(errors).all() or np.any(positions < lower - 1e-6)
                or np.any(positions > upper + 1e-6) or np.any(abs(velocity) > FPS * max_step + 1e-4)):
            raise ValueError(f"{clip} failed trajectory validation")
        report[clip] = {"frames": len(positions), "mean_error_m": float(errors.mean()),
                        "max_error_m": float(errors.max()),
                        "self_collision_samples": 4 * (len(positions) - 1) + 1,
                        "penetrating_samples": 0}
        print(clip, report[clip], flush=True)
        clips[clip] = {"joint_position": torch.from_numpy(positions),
                       "joint_velocity": torch.from_numpy(np.concatenate([velocity, velocity[-1:]]))}
    collision.close()
    output.mkdir(exist_ok=True)
    for clip, tensors in clips.items():
        torch.save(tensors, output / (clip + ".pt"))
    (output / "joint_id.txt").write_text("".join(f"{i} {name}\n" for i, name in enumerate(names)))
    (output / "retarget_report.json").write_text(json.dumps({
        "method": "PyRoki FK and bounded trajectory IK, limb vectors and foot axes in pelvis frame",
        "error": "Distance to scaled elbow, hand, knee and ankle targets relative to shoulder or hip",
        "collision_check": "PyBullet K1 URDF, direct-parent pairs excluded, head yaw 0 and pitch 0.15 rad, linear joint interpolation at 120 Hz",
        "fps": FPS, "clips": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
