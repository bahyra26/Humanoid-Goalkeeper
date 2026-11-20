from legged_gym.envs.k1.k1_config import K1Cfg, K1CfgPPO


class K1WalkCfg(K1Cfg):
    """Omnidirectional walking + upright recovery config for Booster K1 (Policy 2).

    Inherits exact K1 URDF, PD stiffness/damping, action scales, default ready-stance
    joint angles, and domain randomization from K1Cfg for seamless handover with Policy 1.
    """

    class env(K1Cfg.env):
        num_envs = 2048
        num_actor_history = 5
        num_actions = 22
        num_dofs = 22
        # 3 (cmd) + 2 (gait sin/cos) + 3 (ang_vel) + 3 (gravity) + 22 (dof_pos) + 22 (dof_vel) + 22 (actions)
        num_one_step_observations = 77
        num_observations = num_actor_history * num_one_step_observations
        # 77 (one-step) + 3 (base_lin_vel) + 2 (foot_contacts) + 2 (foot_heights) + 1 (friction)
        num_privileged_obs = 85
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 16.0
        play = False

    class commands:
        resampling_time = 4.0
        standing_prob = 0.25
        move_threshold = 0.05
        gait_period = 0.70

        class ranges:
            lin_vel_x = [-0.60, 0.80]
            lin_vel_y = [-0.50, 0.50]
            ang_vel_yaw = [-0.90, 0.90]

    class domain_rand(K1Cfg.domain_rand):
        post_parry_dataset_path = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/k1_post_parry_states.pt"
        real_post_parry_prob = 0.60
        post_parry_reset_prob = 0.30
        post_parry_tilt_range = [-0.28, 0.28]
        post_parry_dof_offset = [-0.20, 0.20]
        post_parry_lin_vel = [-0.25, 0.25]
        post_parry_ang_vel = [-0.35, 0.35]
        push_robots = True
        push_interval_s = 7.0
        max_push_vel_xy = 0.40

    class rewards(K1Cfg.rewards):
        only_positive_rewards = True
        target_base_height = 0.51
        rest_foot_z = 0.038
        swing_clearance = 0.065
        min_foot_dist = 0.14
        max_foot_dist = 0.42
        tracking_sigma = 0.15

        class scales:
            alive = 0.30
            tracking_lin_vel = 5.00
            tracking_ang_vel = 4.00
            gait_contact = 1.80
            feet_clearance = 1.50
            stand_still = 2.00
            orientation = -2.50
            base_height = -5.00
            lin_vel_z = -1.00
            ang_vel_xy = -0.05
            feet_orientation = -0.50
            feet_distance = -2.00
            feet_slip = -0.25
            upper_body_pose = -0.60
            hip_yaw_roll = -0.15
            action_rate = -0.005
            smoothness = -0.003
            dof_vel = -1.0e-4
            dof_acc = -5.0e-8
            torques = -1.0e-5
            dof_pos_limits = -3.0
            torque_limits = -0.05
            collision = -1.5
            termination = -40.0

    class amp(K1Cfg.amp):
        amp_coef = 0.0


class K1WalkCfgPPO(K1CfgPPO):
    class policy(K1CfgPPO.policy):
        init_noise_std = 0.80
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(K1CfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        num_learning_epochs = 4
        num_mini_batches = 4
        learning_rate = 1.5e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.0
        smoothness_lower_bound = 0.005
        smoothness_upper_bound = 1.0

    class runner(K1CfgPPO.runner):
        policy_class_name = 'WalkActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 24
        max_iterations = 1000
        save_interval = 100
        run_name = 'walk'
        experiment_name = 'k1_walk'
        logger = 'tensorboard'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    amp = K1WalkCfg.amp

