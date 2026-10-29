import isaacgym
from isaacgym import gymapi
import numpy as np

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

# Load K1 URDF Asset
asset_root = "/mnt/data/repos/Humanoid-Goalkeeper/legged_gym/resources/robots/k1"
asset_file = "urdf/k1.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = False
asset_options.fix_base_link = True # Fix trunk so robot stays standing in place for 3D viewing

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.55)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
actor = gym.create_actor(env, asset, pose, "k1", 0, 1)

# Set camera close-up looking at K1
cam_pos = gymapi.Vec3(1.6, 1.2, 0.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.45)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n" + "="*50)
print("  ROBOT BOOSTER K1 3D VIEWER AKTIF!")
print("  - Klik Kiri + Geser : Putar sudut pandang")
print("  - Scroll Mouse      : Zoom in / Zoom out")
print("  - Tahan Klik Kanan + WASD : Terbang keliling")
print("  - Tekan ESC atau tutup jendela untuk keluar")
print("="*50 + "\n")

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
