from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class K1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024

        num_actor_history = 10
        
        num_actions = 22 # 22 actuators on Booster K1
        num_dofs = 22
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions + 3 + 1 + 6 + 6 + 1

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 4.
        send_timeouts = True
        episode_length_s = 5
        ball_gravity = True
        play = False

    class commands:
        class ranges_0:
            height = [0.3, 0.8] 
            width =  [0.2, 0.9]
            maxh = [0.2, 1.0]
            maxw = [0.0, 1.2]
            evalh = [0.2, 1.0]
            evalw = [0.0, 1.1]

        class ranges_1:
            height = [0.3, 0.8] 
            width = [-0.9, -0.2]
            maxh = [0.2, 1.0] 
            maxw = [-1.2, -0.0]
            evalh = [0.2, 1.0]
            evalw = [-1.1, 0.0]

        class ranges_2:
            height = [0.8, 1.1] 
            width = [0, 0.8]
            maxh = [0.8, 1.2] 
            maxw = [0, 1.1]
            evalh = [0.8, 1.2] 
            evalw = [0, 1.1]

        class ranges_3:
            height = [0.8, 1.1] 
            width = [-0.8, 0.0]
            maxh = [0.8, 1.2] 
            maxw = [-1.1, 0.0]
            evalh = [0.8, 1.2] 
            evalw = [-1.1, 0.0]

        class ranges_4:
            height = [0.08, 0.3] 
            width = [0.2, 0.9]
            maxh = [0.05, 0.3]
            maxw = [0.0, 1.2]
            evalh = [0.05, 0.3] 
            evalw = [0.0, 1.1]

        class ranges_5:
            height = [0.08, 0.3] 
            width = [-0.9, -0.2]
            maxh = [0.05, 0.3]
            maxw = [-1.2, -0.0]
            evalh = [0.05, 0.3] 
            evalw = [-1.1, -0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55] # x,y,z [m] for Booster K1 (~95cm tall)
        default_joint_angles = {
            'aahead_yaw_joint': 0.0,
            'aahead_pitch_joint': 0.0,

            'aaleft_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.35,
            'left_elbow_pitch_joint': 0.7,
            'left_elbow_yaw_joint': 0.0,

            'aaright_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.35,
            'right_elbow_pitch_joint': 0.7,
            'right_elbow_yaw_joint': 0.0,

            'left_hip_pitch_joint': -0.15,
            'left_hip_roll_joint': 0.08,
            'left_hip_yaw_joint': 0.0,
            'left_knee_pitch_joint': 0.30,
            'left_ankle_pitch_joint': -0.15,
            'left_ankle_roll_joint': -0.08,

            'right_hip_pitch_joint': -0.15,
            'right_hip_roll_joint': -0.08,
            'right_hip_yaw_joint': 0.0,
            'right_knee_pitch_joint': 0.30,
            'right_ankle_pitch_joint': -0.15,
            'right_ankle_roll_joint': 0.08,
        }
        init_pos = [0.0, 0.0, 0.0, 0.35, 0.7, 0.0, 0.0, -0.35, 0.7, 0.0, -0.15, 0.08, 0.0, 0.30, -0.15, -0.08, -0.15, -0.08, 0.0, 0.30, -0.15, 0.08]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'head': 25.0,
            'shoulder': 40.0,
            'elbow': 35.0,
            'hip_pitch': 70.0,
            'hip_roll': 55.0,
            'hip_yaw': 50.0,
            'knee': 100.0,
            'ankle': 35.0,
        }
        damping = {
            'head': 1.0,
            'shoulder': 2.0,
            'elbow': 1.5,
            'hip_pitch': 3.0,
            'hip_roll': 2.5,
            'hip_yaw': 2.0,
            'knee': 4.0,
            'ankle': 2.0,
        }
        action_scale = 0.25
        decimation = 4

        left_leg_joints = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_pitch_joint', 'right_knee_pitch_joint']
        left_arm_joints = ['aaleft_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint', 'left_elbow_yaw_joint']
        right_arm_joints = ['aaright_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint', 'right_elbow_yaw_joint']
        elbow_joints = ['left_elbow_pitch_joint', 'right_elbow_pitch_joint']
        wrist_joints = []
        curriculum_joints = ['left_shoulder_roll_joint', 'right_shoulder_roll_joint']

        upper_body_link = "trunk"
        torso_link = "trunk"

        left_hip_joints = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint']
        right_hip_joints = ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']

    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            ball_vel = 0.2
            ball_pos = 0.3
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            ball = 0.08
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_joint_injection = True
        joint_injection_range = [-0.01, 0.01]
        randomize_actuation_offset = True
        actuation_offset_range = [-0.01, 0.01]
        randomize_payload_mass = True
        payload_mass_range = [-2, 3]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        randomize_kd = True
        kd_range = [0.8, 1.2]
        randomize_initial_joint_pos = True
        continue_keep = True
        initial_joint_pos_scale = [0.5, 1.5]
        initial_joint_pos_offset = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        ball_interval_s = 0.5
        max_ball_vel = 0.5
        delay = True

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/k1/urdf/k1.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball.urdf'
        name = "k1"

        foot_name = "ankle_pitch"
        contact_foot_names = "ankle_roll"

        hand_name = "elbow_yaw"
        penalize_contacts_on = ["hip", "knee", "trunk", "shoulder", "elbow", "head"]
        terminate_after_contacts_on = []

        waist_joints = []
        ankle_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        imu_link = "trunk"
        knee_names = ["left_knee_pitch_link", "right_knee_pitch_link"]
        keyframe_name = "keyframe"

        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        class scales:
            eereach = 10.0
            success = 5.0
            stopball = 100.0
            stayonline = -2.0
            noretreat = -2.0
            successland = 4.0
            feetorientaion = 3.0
            penalize_sharpcontact = -100.
            penalize_kneeheight = -100.
            feet_slippage = 3.0
            postorientation = 3.0
            postangvel = 3.0
            postupperdofpos = 1.0
            postlinvel = 1.0
            ang_vel_xy = -0.1
            dof_acc = -2.5e-7
            smoothness = -0.1
            torques = -1e-5
            dof_vel = -5e-4
            dof_pos_limits = -3.0
            dof_vel_limits = -2.0
            torque_limits = -3.0

        catch_th = 0.35
        handheight_th = 0.7
        reach_th = 0.15
        strict_th = 0.10

        target_dof_pos_sigma = -20
        tracking_sigma = 0.25
        catch_sigma = 5.0

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        max_contact_force = 500.

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30
        min_time = 0.1

    class amp:
        obs_type = 'dof'
        num_obs = 22 * 2
        amp_coef = 0.0
        num_steps = 2

class K1CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100
        max_iterations = 20000
        save_interval = 200
        run_name = 'goalkeeper'
        experiment_name = 'k1'
        wandb_project = "k1_goalkeeper"
        logger = 'tensorboard'

        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    amp = K1Cfg.amp
