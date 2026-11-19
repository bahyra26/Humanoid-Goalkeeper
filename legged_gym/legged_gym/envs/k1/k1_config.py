from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class K1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1536 # Multiple of 6 (256 envs/region), optimal for 80-90W GPU

        num_actor_history = 10
        
        num_actions = 22 # 22 actuators on Booster K1
        num_dofs = 22
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions + 3 + 1 + 6 + 6 + 1

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 4.
        send_timeouts = True
        episode_length_s = 3.2
        ball_gravity = True
        play = False

    class goal:
        # Local X decreases into the goal; this is the rear edge of the line.
        line_x = -0.05
        half_width = 1.30
        height = 1.20
        ball_radius = 0.075  # Matches ball_k1.urdf (Size 1 baseline: 0.200 kg, r=0.075 m).

    class commands:
        train_speed_range = [9.0, 14.5]  # Campuran cepat (55%): ~9.5-15 m/s dgn flight 0.42-0.56 s; 45% standar
        train_fast_ratio = 0.55
        train_kolong_ratio = 0.25  # 25% tembakan rendah standar difokuskan ke celah kolong (|y| <= 0.24 m)
        # Gawang RoboCup KidSize: Lebar 2.6 m (-1.3 m s/d +1.3 m), Tinggi Mistar 1.2 m (0.075 m s/d 1.2 m; radius bola = 0.075 m)
        # Konvensi koordinat robot: +X depan, +Y sisi kiri robot (kanan penendang), -Y sisi kanan robot (kiri penendang)
        class ranges_0: # Samping Kiri Robot / Kanan Penendang (Tinggi Sedang)
            height = [0.35, 0.85] 
            width =  [0.05, 0.75]
            maxh = [0.30, 0.90]
            maxw = [0.0, 1.25]
            evalh = [0.35, 0.85]
            evalw = [0.05, 0.75]

        class ranges_1: # Samping Kanan Robot / Kiri Penendang (Tinggi Sedang)
            height = [0.35, 0.85] 
            width = [-0.75, -0.05]
            maxh = [0.30, 0.90] 
            maxw = [-1.25, 0.0]
            evalh = [0.35, 0.85]
            evalw = [-0.75, -0.05]

        class ranges_2: # Atas Kiri Robot / Kanan Penendang (Area Lompat - Menuju Mistar 1.2 m)
            height = [0.80, 1.15] 
            width = [0.05, 0.70]
            maxh = [0.80, 1.18] 
            maxw = [0.0, 1.20]
            evalh = [0.80, 1.15] 
            evalw = [0.05, 0.70]

        class ranges_3: # Atas Kanan Robot / Kiri Penendang (Area Lompat - Menuju Mistar 1.2 m)
            height = [0.80, 1.15] 
            width = [-0.70, -0.05]
            maxh = [0.80, 1.18] 
            maxw = [-1.20, 0.0]
            evalh = [0.80, 1.15] 
            evalw = [-0.70, -0.05]

        class ranges_4: # Bawah Kiri Robot / Kanan Penendang (Area Rendah/Jongkok)
            height = [0.11, 0.35] 
            width = [0.0, 0.75]
            maxh = [0.11, 0.35]
            maxw = [0.0, 1.25]
            evalh = [0.11, 0.35] 
            evalw = [0.0, 0.75]

        class ranges_5: # Bawah Kanan Robot / Kiri Penendang (Area Rendah/Jongkok)
            height = [0.11, 0.35] 
            width = [-0.75, 0.0]
            maxh = [0.11, 0.35]
            maxw = [-1.25, 0.0]
            evalh = [0.11, 0.35] 
            evalw = [-0.75, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        # POSE SIAP KIPER ATLETIS: kuda-kuda rendah seimbang, lengan terangkat di depan-samping siap menepis ke atas/samping/bawah.
        # Konvensi sendi K1 (dicek dari URDF): pose nol lengan = T-pose (lengan lurus ke samping).
        #   shoulder_roll kiri negatif = lengan turun; elbow_pitch = putar lengan atas (twist);
        #   elbow_yaw = tekuk siku (kiri negatif / kanan positif = ke depan).
        #   Sendi pitch bertanda sama kiri-kanan, sendi roll & yaw tandanya dibalik.
        pos = [0.0, 0.0, 0.54] # x,y,z [m] tinggi trunk kuda-kuda kiper siap (bebas penetrasi telapak kaki saat reset)
        default_joint_angles = {
            'aahead_yaw_joint': 0.0,
            'aahead_pitch_joint': 0.15,          # sedikit menunduk mengawasi bola (~9 derajat)

            'aaleft_shoulder_pitch_joint': -0.25, # lengan siap di depan dada
            'left_shoulder_roll_joint': -0.35,    # tangan di z~0.65m (titik tengah gawang 1.2m: mudah ke atas & ke bawah)
            'left_elbow_pitch_joint': 0.0,
            'left_elbow_yaw_joint': -0.65,        # siku menekuk atletis di depan-samping badan

            'aaright_shoulder_pitch_joint': -0.25,
            'right_shoulder_roll_joint': 0.35,
            'right_elbow_pitch_joint': 0.0,
            'right_elbow_yaw_joint': 0.65,

            'left_hip_pitch_joint': -0.28,
            'left_hip_roll_joint': 0.10,
            'left_hip_yaw_joint': 0.0,
            'left_knee_pitch_joint': 0.58,        # kuda-kuda atletis stabil & siap melompat/menekuk
            'left_ankle_pitch_joint': -0.28,
            'left_ankle_roll_joint': -0.10,

            'right_hip_pitch_joint': -0.28,
            'right_hip_roll_joint': -0.10,
            'right_hip_yaw_joint': 0.0,
            'right_knee_pitch_joint': 0.58,
            'right_ankle_pitch_joint': -0.28,
            'right_ankle_roll_joint': 0.10,
        }
        # Urutan DOF sesuai URDF (head 2, lengan kiri 4, lengan kanan 4, kaki kiri 6, kaki kanan 6)
        init_pos = [0.0, 0.15,
                    -0.25, -0.35, 0.0, -0.65,
                    -0.25, 0.35, 0.0, 0.65,
                    -0.28, 0.10, 0.0, 0.58, -0.28, -0.10,
                    -0.28, -0.10, 0.0, 0.58, -0.28, 0.10]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        limit_joint_targets = False  # Enable only for paired simulation trials first.
        stiffness = {
            'head': 12.0,
            'shoulder': 42.0,
            'elbow': 34.0,
            'hip_pitch': 80.0,
            'hip_roll': 65.0,
            'hip_yaw': 50.0,
            'knee': 115.0,
            'ankle': 40.0,
        }
        damping = {
            'head': 0.5,
            'shoulder': 1.2,
            'elbow': 1.0,
            'hip_pitch': 3.0,
            'hip_roll': 2.5,
            'hip_yaw': 2.0,
            'knee': 4.0,
            'ankle': 2.0,
        }
        action_scale = 0.35
        arm_action_scale = 0.55
        leg_action_scale = 0.50
        head_action_scale = 0.40
        decimation = 4

        left_leg_joints = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_pitch_joint', 'right_knee_pitch_joint']
        left_arm_joints = ['aaleft_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint', 'left_elbow_yaw_joint']
        right_arm_joints = ['aaright_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint', 'right_elbow_yaw_joint']
        elbow_joints = ['left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_yaw_joint']
        wrist_joints = []
        head_joints = ['aahead_yaw_joint', 'aahead_pitch_joint']
        curriculum_joints = ['left_shoulder_roll_joint', 'left_elbow_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint']

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
        payload_mass_range = [-1.0, 1.5]
        randomize_com_displacement = True
        com_displacement_range = [-0.015, 0.015]
        randomize_link_mass = True
        link_mass_range = [0.85, 1.15]
        randomize_friction = True
        friction_range = [0.4, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 0.6]
        randomize_kp = True
        kp_range = [0.85, 1.15]
        randomize_kd = True
        kd_range = [0.85, 1.15]
        randomize_initial_joint_pos = True
        continue_keep = False              # selalu mulai dari kuda-kuda siap kiper yang konsisten
        initial_joint_pos_scale = [0.95, 1.05] # proporsional untuk telapak kaki K1 (18.3 x 7.9 cm)
        initial_joint_pos_offset = [-0.03, 0.03]
        init_base_vel_range = [-0.08, 0.08]
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 0.35
        ball_interval_s = 0.5
        max_ball_vel = 0.05
        delay = True

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/k1/urdf/k1.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball_k1.urdf'
        name = "k1"

        foot_name = "ankle_roll"
        contact_foot_names = "ankle_roll"

        hand_name = "hand_link"
        # K1 URDF has ~4.8-6.5 kN internal link-adjacency contact forces on trunk/hip_roll at rest; keep real collision links only.
        penalize_contacts_on = ["elbow", "shoulder", "head"]
        terminate_after_contacts_on = []

        waist_joints = []
        ankle_joints = ["left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
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

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        class scales:
            eereach = 26.0
            success = 40.0  # Terminal safe-ball bonus; independent of control timestep.
            recovery = 25.0  # Terminal bonus after 0.5 s of safe, stable recovery.
            goal_conceded = -220.0  # Terminal penalty, including goals after a touch.
            stopball = 450.0
            termination = -220.0
            stayonline = -1.0
            noretreat = -1.0
            successland = 14.0
            feetorientaion = 3.0
            airfeetorientation = 2.0
            penalize_sharpcontact = -15.
            penalize_kneeheight = -100.
            collision = -4.0
            arm_overhead = -22.0
            kolong_gap = -120.0
            feet_slippage = 2.5
            postorientation = 10.0
            postangvel = 3.0
            postupperdofpos = 0.0
            post_hold = 12.0
            postlinvel = 2.5
            head_track = 1.0
            ang_vel_xy = -0.07
            dof_acc = -2.0e-7
            smoothness = -0.035
            torques = -1e-5
            dof_vel = -2e-4
            dof_pos_limits = -4.5
            dof_vel_limits = -1.0
            torque_limits = -1.0

        catch_th = 0.35
        handheight_th = 0.7
        reach_th = 0.15
        strict_th = 0.12

        target_dof_pos_sigma = -20
        tracking_sigma = 0.25
        catch_sigma = 5.0

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        max_contact_force = 1100.
        min_knee_height = 0.08
        penalize_knee_height = 0.12

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/k1_arms"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/k1_arms/joint_id.txt"
        frame_rate = 30
        min_time = 0.1

    class amp:
        obs_type = 'dof'
        joint_names = ['aaleft_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                       'left_elbow_pitch_joint', 'left_elbow_yaw_joint',
                       'aaright_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                       'right_elbow_pitch_joint', 'right_elbow_yaw_joint']
        num_obs = 8 * 2
        amp_coef = 0.0
        num_steps = 2

class K1CfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.75

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.004
        num_learning_epochs = 4
        num_mini_batches = 8
        smoothness_lower_bound = 0.02
        smoothness_upper_bound = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 64
        max_iterations = 20000
        save_interval = 50
        run_name = 'goalkeeper'
        experiment_name = 'k1'
        wandb_project = "k1_goalkeeper"
        logger = 'tensorboard'

        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    amp = K1Cfg.amp
