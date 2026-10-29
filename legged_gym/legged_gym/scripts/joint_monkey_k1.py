import math
import numpy as np
from isaacgym import gymapi

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

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

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Load K1 URDF
asset_root = "/mnt/data/repos/Humanoid-Goalkeeper/legged_gym/resources/robots/k1"
asset_file = "urdf/k1.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True

print("Loading K1 asset from '%s'..." % asset_file)
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_positions = dof_states['pos']

has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
        upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    dof_positions[i] = defaults[i]
    speeds[i] = 1.5 * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 2.0 * math.pi)

print("\n--- 22 SENDI BOOSTER K1 YANG DIANIMASIKAN ---")
for i in range(num_dofs):
    print("Sendi %2d: %-30s [Range: %.2f rad s/d %.2f rad]" % (i, dof_names[i], lower_limits[i], upper_limits[i]))
print("-----------------------------------------------\n")

# Create single env centered
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.55)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
actor = gym.create_actor(env, asset, pose, "k1", 0, 1)

# Camera looking at K1
cam_pos = gymapi.Vec3(1.8, 1.3, 0.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

ANIM_DURATION = 2.0 # seconds per joint
current_dof = 0
anim_time = 0.0
dt = 1.0 / 60.0

print("Tekan ESC atau tutup jendela untuk keluar.")
while not gym.query_viewer_has_closed(viewer):
    # Animate active joint with sine wave
    anim_time += dt
    if anim_time > ANIM_DURATION:
        anim_time = 0.0
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        print("Menguji Sendi [%2d / 22]: %s" % (current_dof + 1, dof_names[current_dof]))

    phase = 0.5 * (1.0 + math.sin(anim_time * (2.0 * math.pi / ANIM_DURATION) - 0.5 * math.pi))
    dof_positions[current_dof] = lower_limits[current_dof] + phase * (upper_limits[current_dof] - lower_limits[current_dof])

    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
