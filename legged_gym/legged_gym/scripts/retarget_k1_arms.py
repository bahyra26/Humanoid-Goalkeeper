"""Retarget shoulder-relative G1 elbow/hand paths to the eight K1 arm joints."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


def rotation(axis, angle):
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + angle.sin()[..., None, None] * skew + (1 - angle.cos())[..., None, None] * (skew @ skew)


class ArmChain:
    def __init__(self, urdf, side, robot):
        root = ET.parse(urdf).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        link = f'{side}_hand_link' if robot == 'k1' else f'{side}_rubber_hand'
        base = 'trunk' if robot == 'k1' else 'torso_link'
        chain = []
        while link != base:
            joint = by_child[link]
            chain.insert(0, joint)
            link = joint.find('parent').get('link')
        self.joints, self.names, self.limits, self.velocity_limits = [], [], [], []
        self.elbow = f'{side}_elbow_yaw_link' if robot == 'k1' else f'{side}_elbow_link'
        self.length = 0.0
        for joint in chain:
            origin = joint.find('origin')
            xyz = torch.tensor([float(v) for v in origin.get('xyz', '0 0 0').split()])
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
            orient = torch.eye(3)
            for axis, angle in reversed(list(zip(torch.eye(3), rpy))):
                orient = orient @ rotation(axis, torch.tensor(angle))
            axis_node = joint.find('axis')
            axis = torch.tensor([float(v) for v in axis_node.get('xyz').split()]) if axis_node is not None else torch.tensor([1., 0., 0.])
            moving = joint.get('type') != 'fixed'
            index = len(self.names) if moving else None
            if moving:
                self.names.append(joint.get('name'))
                limit = joint.find('limit')
                self.limits.append((float(limit.get('lower')), float(limit.get('upper'))))
                self.velocity_limits.append(float(limit.get('velocity')))
            self.joints.append((xyz, orient, axis, index, joint.find('child').get('link')))
            if len(self.joints) > 1:
                self.length += xyz.norm().item()
        self.shoulder = self.joints[0][0]

    def points(self, angles):
        position = torch.zeros(angles.shape[0], 3, dtype=angles.dtype)
        orient = torch.eye(3, dtype=angles.dtype).expand(angles.shape[0], 3, 3)
        elbow = None
        for xyz, fixed, axis, index, child in self.joints:
            position = position + (orient @ xyz.view(3, 1)).squeeze(-1)
            orient = orient @ fixed
            if index is not None:
                orient = orient @ rotation(axis, angles[:, index])
            if child == self.elbow:
                elbow = position
        return torch.stack((elbow, position), dim=1) - self.shoulder


def retarget(source, target, source_angles, initial, iterations=600):
    goals = source.points(source_angles).detach() * (target.length / source.length)
    angles = torch.tensor(initial).repeat(source_angles.shape[0], 1).requires_grad_()
    bounds = torch.tensor(target.limits)
    optimizer = torch.optim.Adam([angles], lr=0.04)
    for _ in range(iterations):
        error = target.points(angles) - goals
        loss = error.square().mean() + 0.01 * (angles[1:] - angles[:-1]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            angles.clamp_(bounds[:, 0], bounds[:, 1])
    angles = angles.detach()
    max_step = torch.tensor(target.velocity_limits) * 0.8 / 30
    for indices in (range(1, len(angles)), range(len(angles) - 2, -1, -1)):
        for index in indices:
            neighbor = index - 1 if indices.step > 0 else index + 1
            angles[index] = angles[neighbor] + (angles[index] - angles[neighbor]).clamp(-max_step, max_step)
    errors = torch.linalg.vector_norm(target.points(angles) - goals, dim=-1)
    return angles, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2] / 'resources'
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    dataset = root / 'datasets/goalkeeper'
    mapping = {name: int(index) for index, name in (line.split() for line in (dataset / 'joint_id.txt').read_text().splitlines())}
    chains = [(ArmChain(root / 'robots/g1/urdf/g1_23.urdf', side, 'g1'),
               ArmChain(root / 'robots/k1/urdf/k1.urdf', side, 'k1')) for side in ('left', 'right')]
    names = [name for _, target in chains for name in target.names]
    report, clips = {}, {}
    for path in sorted(dataset.glob('*.pt')):
        data = torch.load(path, map_location='cpu')
        output, errors = [], []
        for side, (source, target) in enumerate(chains):
            source_angles = torch.zeros(len(data['joint_position']), len(source.names))
            for index, name in enumerate(source.names):
                if name in mapping:
                    source_angles[:, index] = data['joint_position'][:, mapping[name]]
                elif 'wrist' not in name:
                    raise ValueError(f'Missing source joint: {name}')
            # shortcut: source wrist roll is absent in the 21-DOF clips; use zero until wrist data exists.
            initial = [-.25, -.35, 0., -.65] if side == 0 else [-.25, .35, 0., .65]
            angles, error = retarget(source, target, source_angles, initial)
            output.append(angles)
            errors.append(error)
        positions = torch.cat(output, dim=1)
        speeds = torch.diff(positions, dim=0) * 30
        error = torch.cat(errors, dim=1)
        limits = torch.tensor([limit for _, target in chains for limit in target.velocity_limits])
        metrics = {'frames': len(positions), 'mean_position_error_m': error.mean().item(),
                   'max_position_error_m': error.max().item(), 'peak_joint_speed_rad_s': speeds.abs().max().item(),
                   'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report[path.stem] = metrics
        if not torch.isfinite(positions).all() or error.max() > .12 or error.mean() > .04 or (speeds.abs() > limits).any():
            raise ValueError(f'Retarget validation failed for {path.name}: {metrics}')
        clips[path.name] = {'joint_position': positions, 'joint_velocity': torch.cat((speeds, speeds[-1:]), dim=0)}
    if len(clips) != 6:
        raise ValueError('Expected six source skills')
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, clip in clips.items():
        torch.save(clip, args.output / filename)
    (args.output / 'joint_id.txt').write_text(''.join(f'{index} {name}\n' for index, name in enumerate(names)))
    (args.output / 'retarget_report.json').write_text(json.dumps({'joint_names': names, 'fps': 30,
        'method': 'URDF elbow/hand paths relative to shoulder, scaled by arm length; bounded IK',
        'limits': 'Arm kinematics only; collision, whole-body balance and human style not certified.',
        'clips': report}, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
