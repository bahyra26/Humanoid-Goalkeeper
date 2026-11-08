import isaacgym
from isaacgym import gymapi
import numpy as np
import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_29_config import G129Cfg

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Create viewer window
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Load G1 29-DOF URDF Asset
asset_root = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/g1")
asset_file = "urdf/g1_29.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = False
asset_options.fix_base_link = True # Fix pelvis so robot stays standing in place
asset_options.use_mesh_materials = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Load Ball Asset
ball_root = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/gymassets/urdf")
ball_file = "ball.urdf"
ball_options = gymapi.AssetOptions()
ball_options.fix_base_link = True
ball_asset = gym.load_asset(sim, ball_root, ball_file, ball_options)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0.0), gymapi.Vec3(2.5, 2.5, 2.5), 1)

# Spawn Robot G1 at its configured standing height (Z = 0.80 m)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, G129Cfg.init_state.pos[2]) # 0.80m
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
actor = gym.create_actor(env, asset, pose, "g1", 0, 1)

# Spawn Ball in front of G1 (2.0 m in front, height 0.65 m)
ball_pose = gymapi.Transform()
ball_pose.p = gymapi.Vec3(2.0, 0.0, 0.65)
ball_actor = gym.create_actor(env, ball_asset, ball_pose, "ball", 0, 2)

# Apply Ready Stance (default_joint_angles from g1_29_config.py)
dof_names = gym.get_asset_dof_names(asset)
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

default_angles = G129Cfg.init_state.default_joint_angles
for i, name in enumerate(dof_names):
    if name in default_angles:
        dof_states['pos'][i] = default_angles[name]

gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)

# Set camera close-up looking at G1 in Ready Stance
cam_pos = gymapi.Vec3(2.5, 1.8, 1.1)
cam_target = gymapi.Vec3(0.3, 0.0, 0.65)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n" + "="*60)
print("  ROBOT UNITREE G1 (29 DOF) 3D VIEWER: POSE SIAGA (READY STANCE)")
print("  - Robot berdiri di Z = 0.80m dengan kuda-kuda kiper siap tangkap bola")
print("  - Bola simulasi berada di depan robot pada jarak 2.0 meter")
print("  ----------------------------------------------------------")
print("  Kontrol Kamera:")
print("  - Klik Kiri + Geser       : Putar sudut pandang (orbit)")
print("  - Scroll Mouse            : Zoom in / Zoom out")
print("  - Tahan Klik Kanan + WASD : Terbang keliling")
print("  - Tekan ESC               : Tutup jendela")
print("="*60 + "\n")

while not gym.query_viewer_has_closed(viewer):
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
