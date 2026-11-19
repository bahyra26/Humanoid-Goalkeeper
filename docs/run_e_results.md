# Run E Evaluation, Tables, Plots, and IEEE Report Draft

## Selected checkpoint for the report and demonstration

Across evaluation seeds 1, 777 and 2026, `landfix_5350` saved 2,076 of 4,312 on-target shots (48.14%). Its parent, `nocollide_5200`, saved 2,008 (46.57%). The paired gain was 1.58 percentage points (95% CI 0.40–2.75). Falls decreased from 153 to 116 across 4,320 trials. Two-foot support at the final observation declined from 87.6% to 73.8%. [Benchmark](../evaluation/k1_g_allseeds_benchmark.json) · [Paired comparison](../evaluation/k1_g_paired_ci.json)

The terminal recovery bonus requires 25 consecutive stable steps through timeout. In the reported training-condition replay, `landfix_5350` earned the bonus in 115 of 766 shared saves reaching timeout with deterministic actions. With PPO action sampling it earned the bonus in 0 of 548 shared saves reaching timeout. In each of those 548 episodes, both foot support and angular speed violated their respective criteria at least once during the final 0.5 s. The audit found no discrepancy between bonus eligibility and payment. The sampled-action comparison had a 0.028 m initial ball-state difference between checkpoints. These results explain how a successful save can earn zero recovery bonus under the training conditions tested.

## 1. Run E (`k1_v9_s1_fast5050`) vs `v9_5000` and Run A (`ctrl`)

### A. Benchmark Summary (`Seeds 1 + 777` = 2,876 On-Target Shots + 69 Deterministic Shots, Size 1 Ball: 200 g, $r = 0.075\text{ m}$)

| Checkpoint | Random Saves (`2,876`) | $\Delta$ vs `v9_5000` | Falls (`2,880`) | `STD` (`720`) | `FAST` Combined (`2,156`) | `10ms` / `12ms` / `14ms` | `KOLONG_12` | `HIGH_GRID_30` | `SPRINT1_18` | `MOTION_9` |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **`v9_5000` *(Main)*** | `1,315` (`45.72%`) | `+0` | **`40`** | `554` (`76.94%`) | `761` (`35.30%`) | `245 / 267 / 249` | `4/12` | `3/30` | `15/18` | `6/9` |
| **Run A `ctrl_5050`** | `1,270` (`44.16%`) | `-45` | `54` | `549` (`76.25%`) | `721` (`33.44%`) | `227 / 258 / 236` | `1/12` | `3/30` | `16/18` | `9/9` |
| **Run A `ctrl_5100`** | `1,288` (`44.78%`) | `-27` | `108` | `537` (`74.58%`) | `751` (`34.83%`) | `249 / 267 / 235` | `1/12` | `1/30` | `16/18` | `8/9` |
| **Run A `ctrl_5150`** | `1,346` (`46.80%`) | `+31` | `72` | `550` (`76.39%`) | `796` (`36.92%`) | `258 / 279 / 259` | `2/12` | **`4/30`** | `15/18` | `8/9` |
| **Run A `ctrl_5200`** | `1,312` (`45.62%`) | `-3` | `72` | `547` (`75.97%`) | `765` (`35.48%`) | `256 / 283 / 226` | `4/12` | `3/30` | `15/18` | `7/9` |
| **Run A `ctrl_5250`** | `1,292` (`44.92%`) | `-23` | `74` | `557` (`77.36%`) | `735` (`34.09%`) | `249 / 269 / 217` | **`7/12`** | `2/30` | `15/18` | **`9/9`** |
| **Run E `fast5050_5050`** | `1,299` (`45.17%`) | `-16` | **`33`** | `569` (`79.03%`) | `730` (`33.86%`) | `236 / 244 / 250` | `0/12` | `2/30` | `15/18` | `9/9` |
| **Run E `fast5050_5100`** | `1,338` (`46.52%`) | `+23` | `115` | `552` (`76.67%`) | `786` (`36.46%`) | `261 / 271 / 254` | `0/12` | `2/30` | **`17/18`** | `9/9` |
| **Run E `fast5050_5150`** | `1,344` (`46.73%`) | `+29` | `99` | `557` (`77.36%`) | `787` (`36.50%`) | `262 / 285 / 240` | `1/12` | `3/30` | **`17/18`** | `8/9` |
| **Run E `fast5050_5200`** | **`1,372` (`47.71%`)** | **`+57`** | `102` | **`576` (`80.00%`)** | **`796` (`36.92%`)** | `267 / 281 / 248` | `0/12` | `1/30` | `16/18` | `8/9` |
| **Run E `fast5050_5250`** | **`1,365` (`47.46%`)** | **`+50`** | `93` | **`574` (`79.72%`)** | `791` (`36.69%`) | `259 / 283 / 249` | `2/12` | `3/30` | `16/18` | `7/9` |

### B. Paired Per-Shot Comparisons & New Evaluation Seed (`seed = 2026`, `1,436` On-Target Shots)

| Pair (`Model A vs Model B`) | `Seeds 1 + 777` (`2,876` shots) | `New Seed 2026` (`1,436` shots) | `All 3 Seeds` (`4,312` shots) |
|---|---|---|---|
| **`fast5050_5200` vs `v9_5000`** | `+57` (`+1.98%`), 95% CI `[+0.59%, +3.38%]` (`+239/-182`) | `+18` (`+1.25%`), 95% CI `[-0.85%, +3.36%]` (`+128/-110`) | **`+75` (`+1.74%`), 95% CI `[+0.57%, +2.91%]`** (`+367/-292`) |
| **`fast5050_5200` vs `ctrl_5200`** | `+60` (`+2.09%`), 95% CI `[+0.66%, +3.51%]` (`+250/-190`) | `+32` (`+2.23%`), 95% CI `[+0.16%, +4.30%]` (`+131/-99`) | **`+92` (`+2.13%`), 95% CI `[+0.96%, +3.31%]`** (`+381/-289`) |
| **`fast5050_5250` vs `v9_5000`** | `+51` (`+1.77%`), 95% CI `[+0.36%, +3.18%]` (`+240/-189`) | `-4` (`-0.28%`), 95% CI `[-2.29%, +1.74%]` (`+107/-111`) | `+47` (`+1.09%`), 95% CI `[-0.07%, +2.25%]` (`+347/-300`) |
| **`fast5050_5250` vs `ctrl_5250`** | `+74` (`+2.57%`), 95% CI `[+1.09%, +4.05%]` (`+274/-200`) | `-3` (`-0.21%`), 95% CI `[-2.34%, +1.92%]` (`+120/-123`) | **`+71` (`+1.65%`), 95% CI `[+0.43%, +2.86%]`** (`+394/-323`) |

### C. Generated Figures
- Save Rate by Shot Speed: `evaluation/k1_save_rate_by_speed.png`
- Save Rate by Target Region: `evaluation/k1_save_rate_by_region.png`

---

## 2. Historical Run E IEEE Report Draft (`v9_5000` Baseline + Runs A–E Controlled Ablations)

Below is the historical Run E report draft documenting the `v9_5000` baseline and Runs A–E ablations (prior to the final `landfix_5350` and `gema25_5400` runs summarized above and in [K1_REWARD_CALIBRATION.md](../evaluation/K1_REWARD_CALIBRATION.md)).

---

### Abstract
**Abstract—**This paper describes the design, physical calibration, and empirical evaluation of a reinforcement-learning humanoid goalkeeper policy for the 22-DOF Booster K1 robot in an NVIDIA Isaac Gym (PhysX GPU) robot soccer simulation aligned with RoboCup Humanoid League KidSize 2025 specifications. The system operates at a 50 Hz control frequency (200 Hz physics substep) using a 10-step proprioceptive and ball-position history encoded into a 98-dimensional actor input, coupled with learned ball-landing (`Ball MLP`) and 6-region classification (`Region MLP`) estimators. We calibrate the simulation environment to a KidSize goal opening ($2.60 \times 1.20\text{ m}$, goal-line rear edge $x = -0.05\text{ m}$) and a Size 1 soccer ball ($m = 0.200\text{ kg}$, $r = 0.075\text{ m}$, thin-shell inertia $I = 7.5 \times 10^{-4}\text{ kg}\cdot\text{m}^2$) with isolated rigid-body damping, and enforce strict paired evaluation against a no-keeper baseline so that off-target misses and rebounds that cross the goal plane are never misclassified as saves. Across 2,876 on-target random shots spanning standard ($4.5\text{--}8.5\text{ m/s}$) and high-speed ($10\text{--}14\text{ m/s}$) regimes, the primary policy (`k1_v9/model_5000.pt`) achieves a **76.94% save rate (554/720)** on standard shots and **45.72% overall (1,315/2,876)** while falling on only **1.39% (40/2,880)** of trials and recovering to stable double-foot support on **93.9%** of standard saves. Controlled one-factor-at-a-time (OFAT) fine-tuning experiments on the 200 g ball further demonstrate that (1) adapting to the smaller ball increases between-the-legs (`kolong`) blocks from **4/12 (33.3%) to 7/12 (58.3%)** and posture-audit saves from **6/9 to 9/9** (`ctrl_5250`), and (2) expanding training flight-time diversity to include short-arrival shots ($0.32\text{--}0.48\text{ s}$) increases overall random saves to **47.71% (1,372/2,876)** (`fast5050_5200`, $+2.13\text{ percentage points}$ over same-iteration control across 4,312 three-seed shots, 95% CI $[+0.96\%, +3.31\%]$), albeit with higher fall rates ($74\text{--}102$ falls).

**Index Terms—**humanoid robot, goalkeeper policy, robot soccer, ball trajectory, motion control, reinforcement learning, simulation.

---

### I. INTRODUCTION
A humanoid goalkeeper in robot soccer must intercept fast-moving balls within $300\text{--}800\text{ ms}$ while respecting actuator torque limits, kinematic reach, and whole-body balance. Unlike wheeled or fixed-base manipulators, a bipedal goalkeeper cannot rely on static anchoring: lunging laterally, squatting to close the between-the-legs (*five-hole* / *kolong*) gap, or jumping toward the crossbar induces angular momentum that can easily topple the robot either before ball arrival or after deflection. Furthermore, uncalibrated simulation evaluations frequently overestimate goalkeeper performance when oversized balls, inherited joint damping on the ball actor, or loose contact-based save metrics credit off-target shots and post-touch goals as successful blocks.

The objective of this project is to design, calibrate, and rigorously evaluate a closed-loop humanoid goalkeeper policy for the Booster K1 KidSize humanoid in simulation. The specific contributions supported by our implementation are:
1. **RoboCup KidSize 2025 Physical Calibration and Paired Outcome Verification:** We calibrate the goal geometry ($2.60 \times 1.20\text{ m}$) and Size 1 ball dynamics ($m = 0.200\text{ kg}$, $r = 0.075\text{ m}$, $I = 7.5 \times 10^{-4}\text{ kg}\cdot\text{m}^2$, zero artificial armature/damping) and implement a paired no-keeper trajectory verifier (`shot_outcome.py`, `goal_detection.py`) backed by 58 automated unit tests.
2. **Position-Conditioned Task-Motion Goalkeeper Policy:** We train an end-to-end actor-critic policy with auxiliary ball-landing and target-region estimators (`k1_v9/model_5000.pt`), combining three-phase interception rewards (preparation, active reach/block, and double-support post-save stabilization) with an optional substep PD torque safety clamp (`legged_robot.py:_compute_torques`, `limit_joint_targets`).
3. **Controlled One-Factor-at-a-Time (OFAT) Experimental Analysis:** Using identical random seeds and shot sets, we quantify the effects of ball-size migration ($400\text{ g}, r = 0.10\text{ m}$ vs. $200\text{ g}, r = 0.075\text{ m}$), between-the-legs reward shaping (Runs A, B, C), high-ball countermovement timing thresholds (Run D), and short-arrival training distributions (Run E).

---

### II. RELATED WORK
Research on humanoid goalkeeping and agile legged locomotion spans model-based control, imitation learning, and deep reinforcement learning (DRL). Early RoboCup Humanoid League goalkeepers relied on scripted dive keyframes triggered by geometric time-to-contact thresholds [1], which struggle to recover balance or adapt to variable bounce heights. In general legged locomotion, Rudin *et al.* [2] demonstrated that massively parallel GPU simulation in NVIDIA Isaac Gym enables rapid PPO training of robust bipedal and quadrupedal controllers, while Long *et al.* [3] introduced Hybrid Internal Model (HIM) estimation from short proprioceptive histories to stabilize locomotion under partial observability. Peng *et al.* [4] proposed Adversarial Motion Priors (AMP) to regularize RL policies toward natural human motion capture clips.

Specifically for legged robotic goalkeeping, Huang *et al.* [6] demonstrated a hierarchical RL goalkeeper on a quadrupedal robot (MIT Mini Cheetah) by combining low-level trajectory-tracking maneuver skills (jump, dive, sidestep) with a high-level interception planner. For bipedal humanoid goalkeeping, recent work on position-conditioned task-motion constraints [5] formulated shot blocking in a single unified policy by dividing the goal plane into spatial regions, predicting the ball's intersection point from observation history, and coupling region-specific reach rewards with post-impact recovery constraints. Our work builds directly upon this position-conditioned framework and adapts it from larger humanoid platforms to the compact 22-DOF Booster K1 KidSize morphology ($19.67\text{ kg}$, standing torso height $0.54\text{ m}$), introducing exact whole-ball goal-plane segment intersection, Size 1 ball physics isolation, between-the-legs (`kolong`) geometry shaping, and paired no-keeper evaluation.

---

### III. GOALKEEPER APPROACH AND METHODOLOGY

#### A. System Overview
The simulation environment is implemented in NVIDIA Isaac Gym using the GPU PhysX pipeline (`dt = 0.005 s`, control decimation $= 4$, yielding a $50\text{ Hz}$ policy loop). The goalkeeper is the **Booster K1** humanoid robot (`total mass = 19.67 kg`, standing torso height $z_0 = 0.54\text{ m}$) equipped with $22$ actuated degrees of freedom (DOFs):
- **Head (2 DOFs):** `aahead_yaw_joint`, `aahead_pitch_joint`
- **Left & Right Arms (8 DOFs):** shoulder pitch/roll and elbow pitch/yaw (`4 DOFs` per arm)
- **Left & Right Legs (12 DOFs):** hip pitch/roll/yaw, knee pitch, and ankle pitch/roll (`6 DOFs` per leg)

At each $20\text{ ms}$ control step $t$, the single-step observation vector $\mathbf{o}_t \in \mathbb{R}^{75}$ concatenates the relative ball position in the robot torso frame ($\mathbb{R}^3$), base angular velocity ($\mathbb{R}^3$), projected gravity vector ($\mathbb{R}^3$), joint positions relative to default stance ($\mathbb{R}^{22}$), joint velocities ($\mathbb{R}^{22}$), and previous actions ($\mathbb{R}^{22}$). A 10-step rolling history $\mathbf{H}_t \in \mathbb{R}^{750}$ is fed into three parallel encoders (`him_actor_critic.py`):
1. **History MLP** ($750 \rightarrow 128 \rightarrow 64 \rightarrow 16$) extracting latent robot dynamics,
2. **Ball MLP** ($750 \rightarrow 128 \rightarrow 32 \rightarrow 6$) predicting the ball's 3D start and goal-plane intersection coordinates $\hat{\mathbf{p}}_{\text{end}}$,
3. **Region MLP** ($750 \rightarrow 128 \rightarrow 32 \rightarrow 6$) predicting logits over the six goal target regions $R_0 \dots R_5$.

The concatenated 98-dimensional vector $[\mathbf{o}_t, \mathbf{z}_{\text{hist}}, \hat{\mathbf{p}}_{\text{ball}}, \hat{\mathbf{r}}_{\text{reg}}] \in \mathbb{R}^{98}$ drives the **Actor MLP** ($98 \rightarrow 512 \rightarrow 256 \rightarrow 256 \rightarrow 22$, ELU activations), outputting joint position targets $\mathbf{q}^{\text{des}}_t = \mathbf{q}_0 + 0.25\,\mathbf{a}_t$ tracked by a 200 Hz PD controller.

#### B. Task Constraints
Following the RoboCup Humanoid League KidSize 2025 rules [1], the goal opening has width $W = 2.60\text{ m}$ ($y \in [-1.30, +1.30]\text{ m}$) and crossbar height $H = 1.20\text{ m}$, with the rear edge of the goal line located at $x_{\text{line}} = -0.05\text{ m}$ (`k1_config.py`). The robot initializes upright at $(x, y, z) = (0.0, 0.0, 0.54)\text{ m}$ facing $+x$. The ball is a Size 1 mini soccer ball (`ball_k1.urdf`) with mass $m = 0.200\text{ kg}$, radius $r = 0.075\text{ m}$, and thin spherical shell inertia:
$$I_{xx} = I_{yy} = I_{zz} = \frac{2}{3} m r^2 = 7.5 \times 10^{-4}\text{ kg}\cdot\text{m}^2$$
To prevent artificial aerodynamic slowdown, the ball asset is loaded with dedicated `AssetOptions` (`armature = 0.0`, `linear_damping = 0.0`, `angular_damping = 0.0`). The goal plane is partitioned into six target regions (`R0` Mid-Left, `R1` Mid-Right, `R2` High-Left, `R3` High-Right, `R4` Low-Left, `R5` Low-Right). A goal is registered by `ball_crossed_goal()` (`goal_detection.py`) if and only if the ball's center crosses the full-ball threshold plane $x_{\text{thresh}} = x_{\text{line}} - r = -0.125\text{ m}$ with interpolated crossing coordinates satisfying $|y_{\text{cross}}| \le 1.30 - r = 1.225\text{ m}$ and $z_{\text{cross}} \le 1.20 - r = 1.125\text{ m}$.

#### C. Motion Constraints
To protect hardware-equivalent joint limits and prevent erratic arm flailing, three motion constraints are enforced:
1. **PD Torque Safety Clamp (`legged_robot.py:_compute_torques`):** When `limit_joint_targets` is enabled, desired PD joint targets are clamped at every physics substep so that implied torque does not exceed `torque_safety_scale = 1.0` of the URDF effort limits ($K_p(\mathbf{q}^{\text{des}} - \mathbf{q}) - K_d\dot{\mathbf{q}} \in [-\tau_{\max}, +\tau_{\max}]$).
2. **Overhead & Ready-Guard Arm Constraints (`_reward_arm_overhead`):** Off-side hands raised above `head_z - 0.09 m` and active hands raised excessively above the shot target height are penalized, while an early-phase guard (`ball_local_x >= 2.0 m`) encourages a ready stance with hand separation $\Delta y_{\text{hands}} \ge 0.72\text{ m}$ and height $z_{\text{hands}} \ge 0.46\text{ m}$.
3. **Between-the-Legs (`Kolong`) & Post-Save Double-Support Constraints:** For low central shots ($|y_{\text{end}}| < 0.30\text{ m}$, $z \le 0.36\text{ m}$), `_reward_kolong_gap` penalizes ankle separations exceeding $0.22\text{ m}$ and unblocked foot-to-ball lateral errors above $0.06\text{ m}$. After the ball is cleared (`_k1_ball_safe`), `_reward_postorientation`, `_reward_postlinvel`, and `_reward_postupperdofpos` require the robot to return to an upright, stationary, double-foot support posture.

#### D. Ball-Trajectory Handling
In the world/environment frame (origin at the center of the goal line on the ground, $+x$ toward the field, $+y$ toward the goalkeeper's left, $+z$ upward), shots are launched from $\mathbf{p}_0 = (x_0, y_0, z_0)$ toward target $\mathbf{p}_{\text{end}} = (0.08, y_{\text{goal}}, z_{\text{goal}})\text{ m}$ under gravity $g = 9.81\text{ m/s}^2$. For a prescribed flight time $t_{\text{flight}}$ or launch speed $v_0$ (`ball_launch.py`), initial velocities are computed via projectile kinematics:
$$v_x = 1.04\,\frac{x_{\text{end}} - x_0}{t_{\text{flight}}}, \quad v_y = 1.04\,\frac{y_{\text{end}} - y_0}{t_{\text{flight}}}, \quad v_z = \frac{z_{\text{end}} - z_0 + \frac{1}{2} g t_{\text{flight}}^2}{t_{\text{flight}}}$$
During the episode, the actor does not receive ground-truth $\mathbf{p}_{\text{end}}$ directly; instead, the `Ball MLP` and `Region MLP` infer $\hat{\mathbf{p}}_{\text{end}}$ and $\hat{\mathbf{r}}_{\text{reg}}$ online from the 10-step observation history of torso-relative ball positions $\mathbf{p}_{\text{ball}}^{\text{torso}} = R_{\text{torso}}^\top (\mathbf{p}_{\text{ball}} - \mathbf{p}_{\text{torso}})$. When the ball enters close range ($0.05 < x_{\text{ball}} < 0.80\text{ m}$) while still live, `end_target` switches to tracking the live 3D ball state directly (`legged_robot.py:284`).

#### E. Goalkeeper Policy and Control
The reward function in `legged_robot.py` coordinates three phases of each shot episode:
1. **Approach & Interception (`_reward_eereach`, weight $1.5$):** Combines lateral torso positioning (`lat_rew` + `lat_vel_bonus` with a $\pm 0.20\text{ m}$ `kolong_offset` on low-center shots so the near shin aligns with the ball path), region-conditioned vertical torso target ($z_{\text{target}} = 0.40\text{ m}$ for low shots, $0.72\text{ m}$ with countermovement dip allowance down to $0.42\text{ m}$ for high shots), active-hand distance shaping (`hand_reach` + `act_z_match`), close-range body/arm proximity (`live_track`), and low-ball foot/shin blocking (`fivehole_rew` gated by `center_gate`).
2. **Momentum-Stopping Impulse (`_reward_stopball`, weight $12.0$, and `_reward_success`, weight $15.0$):** Uses `contact_candidate()` (`shot_outcome.py`) scaled by `ball_radius = 0.075 m` to verify physical contact (contact force $> 1\text{ N}$ or proximity within $r + 0.14\text{ m}$ coupled with velocity change $\Delta v > 0.35\text{ m/s}$) and rewards forward velocity reversal $\Delta v_x = v_{x,t} - v_{x,0}$ plus a terminal safe-clearance bonus paid upon episode completion without conceding a goal.
3. **Post-Save Recovery (`_reward_successland`, `_reward_post_hold`, `_reward_termination`):** Rewards airborne vertical lift on high shots followed by two-foot landing, triples action-smoothness and joint-acceleration penalties once `_k1_ball_safe()` is true to eliminate post-save oscillation, and applies a terminal fall penalty ($-100$) and goal-conceded penalty ($-85$).

---

### IV. PROTOTYPE IMPLEMENTATION AND REPOSITORY

#### A. Policy Implementation
The goalkeeper prototype is implemented in Python using PyTorch and the Isaac Gym `legged_gym` / `rsl_rl` framework:
- `legged_gym/legged_gym/envs/base/legged_robot.py`: Core environment step, ball spawner (`assign_ball_states`), observation builder, optional substep PD torque safety clamp (`_compute_torques`), and K1 reward/termination functions.
- `legged_gym/legged_gym/envs/k1/k1_config.py`: K1 robot, goal, ball, PD gains, and reward weight configurations.
- `legged_gym/resources/gymassets/urdf/ball_k1.urdf`: Size 1 ball URDF ($m = 0.200\text{ kg}$, $r = 0.075\text{ m}$, $I = 0.00075\text{ kg}\cdot\text{m}^2$).
- `legged_gym/legged_gym/utils/goal_detection.py` & `shot_outcome.py`: Exact goal-plane segment crossing detector and paired save/goal/miss outcome classifier.
- `rsl_rl/rsl_rl/modules/him_actor_critic.py`: Actor, Critic, History, Ball, and Region MLP networks.

#### B. GitHub Repository and README
- **Repository structure:** `legged_gym/` (environments, configs, URDF assets, training and evaluation scripts), `rsl_rl/` (PPO runner and HIM networks), `tests/` (58 unit tests covering ball launch, goal detection, shot outcomes, reach/kolong rewards, safe recovery, and joint guards), and `evaluation/` (paired benchmark JSON logs and plots).
- **Execution:** `play_k1_size1.sh` verifies the Size 1 ball parameters and launches interactive 3D playback or deterministic scenario replays with live goal-line visual indicators.

#### C. Reproducibility and Artifacts
All experiments were executed on Linux with Python 3.8, PyTorch `2.1.2+cu118`, and NVIDIA Isaac Gym Preview 4 (`gym_38`) on `cuda:0`. Deterministic and multi-seed paired benchmarks are reproduced via:
```bash
# 1. Run the 58-test verification suite
python3 -m unittest discover -s tests -v

# 2. Reproduce the Size 1 paired benchmark (no_keeper, v8_4750, v9_5000)
cd legged_gym/legged_gym/scripts
python3 benchmark_k1.py \
  --models \
    v8_4750=../../logs/k1/k1_v8/model_4750.pt \
    v9_5000=../../logs/k1/k1_v9/model_5000.pt \
  --output ../../../evaluation/k1_paired_benchmark.json

# 3. Launch 3D visual demo on Size 1 ball
../../../play_k1_size1.sh
```

---

### V. EXPERIMENTAL SETUP

#### A. Simulation Environment
All evaluations run with domain randomization and observation noise disabled (`add_noise = False`, `randomize_friction = False`, `push_robots = False`), `360` parallel environments, episode horizon $150\text{ steps}$ ($3.0\text{ s}$ at $50\text{ Hz}$), and an explicit GPU PhysX force/articulation flush (`_settle_reset`) prior to each batch so that initial ball states match to within $10^{-5}$.

#### B. Evaluation Protocol
Every shot batch is first executed with **`no_keeper`** (robot translated out of the goal to $y = -4.0\text{ m}$) to record ground-truth unblocked trajectories and identify **on-target shots** (`nk_goals`). Each goalkeeper checkpoint is then tested on the exact same shots across two protocols:
1. **Deterministic Focused Suites (`69` shots, `seed = 0`):**
   - `SPRINT1_18` (`18` shots across speeds $6.0\text{--}10.0\text{ m/s}$ and all 6 regions),
   - `KOLONG_12` (`12` low between-the-legs shots at $y \in [-0.22, +0.22]\text{ m}$, $z \in [0.14, 0.16]\text{ m}$),
   - `MOTION_9` (`9` boundary posture-audit shots),
   - `HIGH_GRID_30` (`30` high-corner shots at $v \in \{7.5, 10.0, 10.5, 12.0, 14.0\}\text{ m/s}$, $z \in \{0.85, 0.98, 1.12\}\text{ m}$, $y = \pm 0.65\text{ m}$).
2. **Multi-Speed Random Suites (`2,880` shots across `seed = 1` and `seed = 777`, plus `1,440` held-out verification shots on `seed = 2026`):**
   - `STD` ($720$ shots, flight time $0.58\text{--}1.10\text{ s}$, $v \approx 4.5\text{--}8.5\text{ m/s}$, $x_0 \in [3.0, 4.4]\text{ m}$),
   - `10ms` ($720$ shots at $10.0\text{ m/s}$, $x_0 \in [3.0, 4.4]\text{ m}$),
   - `12ms_5m` ($720$ shots at $12.0\text{ m/s}$, $x_0 = 5.0\text{ m}$),
   - `14ms_6m` ($720$ shots at $14.0\text{ m/s}$, $x_0 = 6.0\text{ m}$).

#### C. Evaluation Metric
Using `shot_outcome_masks()` (`shot_outcome.py`), every trial is partitioned into four mutually exclusive outcomes on on-target shots (`would_score = True`):
1. **Blocked Save (`blocked_save`):** The shot would score without the keeper, the keeper made verified physical contact, no goal was conceded, and the ball either cleared the danger zone (`x > 0.45 m`, side exit $|y| > 1.375\text{ m}$, or outbound $v_x \ge 0.45\text{ m/s}$) or was stopped dead in front of the goal line ($|v| < 0.35\text{ m/s}$).
2. **Goal Conceded (`goal`):** The full ball crossed the goal opening ($x \le -0.125\text{ m}$, $|y| \le 1.225\text{ m}$, $z \le 1.125\text{ m}$), regardless of whether the keeper touched it first.
3. **Unresolved (`unresolved`):** The ball remained live in front of the line at timeout without meeting clearance criteria.
4. **Off-Target (`out_no_touch` / `off_target_touched`):** Shots that missed the goal in the `no_keeper` run (4 out of 2,880 random shots) are excluded from the denominator so that wide misses are never counted as saves:
$$\text{Save Rate (\%)} = \frac{N_{\text{blocked\_save}}}{N_{\text{on\_target}}} \times 100$$

---

### VI. RESULTS AND DISCUSSION

#### A. Save Rate by Shot Speed
Table I reports the measured performance of the primary goalkeeper policy (`v9_5000`) across the four shot-speed modes (`seeds 1 + 777`, $2,880$ total shots, $2,876$ on-target).

**TABLE I: Save Rate by Shot Speed (`v9_5000`, Size 1 Ball: 200 g, $r = 0.075\text{ m}$)**

| Shot Speed Condition | Total Shots | On-Target Shots | Blocked Saves | On-Target Save Rate (%) | Goals Conceded | Unresolved | Robot Falls | Double-Support Recovery (`2ft`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Standard (`4.5–8.5 m/s`, $x_0 = 3.0\text{--}4.4\text{ m}$)** | 720 | 720 | **554** | **76.94%** | 161 | 5 | **3** (0.4%) | **93.9%** |
| **`10.0 m/s` ($x_0 = 3.0\text{--}4.4\text{ m}$)** | 720 | 716 | **245** | **34.22%** | 467 | 4 | 15 (2.1%) | 84.5% |
| **`12.0 m/s` ($x_0 = 5.0\text{ m}$)** | 720 | 720 | **267** | **37.08%** | 450 | 3 | 9 (1.3%) | 87.6% |
| **`14.0 m/s` ($x_0 = 6.0\text{ m}$)** | 720 | 720 | **249** | **34.58%** | 466 | 5 | 13 (1.8%) | 88.4% |
| **Total / Overall** | **2,880** | **2,876** | **1,315** | **45.72%** | **1,544** | **17** | **40 (1.39%)** | **90.5%** |

*(See `evaluation/k1_save_rate_by_speed.png` for the comparative bar chart across `v9_5000`, `ctrl_5250`, and `fast5050_5200`.)*

#### B. Save Rate by Target Position
Table II reports the save rate of `v9_5000` across the six spatial regions of the goal opening, comparing standard-speed shots (`STD`, $720$ shots) against all speed modes combined ($2,876$ on-target shots), as well as deterministic suite performance.

**TABLE II: Save Rate by Target Position (`v9_5000`, Size 1 Ball: 200 g, $r = 0.075\text{ m}$)**

| Target Region / Position | Spatial Bounds ($y, z$) | Standard Speed Saves (`STD`) | Standard Save Rate (%) | All-Speed Saves (`2,876`) | All-Speed Save Rate (%) |
|---|---|---:|---:|---:|---:|
| **`R0` Mid-Left** | $y \in [0.05, 0.75]\text{ m},\; z \in [0.35, 0.85]\text{ m}$ | 110 / 120 | **91.67%** | 302 / 480 | **62.92%** |
| **`R1` Mid-Right** | $y \in [-0.75, -0.05]\text{ m},\; z \in [0.35, 0.85]\text{ m}$ | 102 / 120 | **85.00%** | 332 / 480 | **69.17%** |
| **`R4` Low-Left** | $y \in [0.02, 0.65]\text{ m},\; z \in [0.12, 0.30]\text{ m}$ | 96 / 120 | **80.00%** | 184 / 480 | **38.33%** |
| **`R5` Low-Right** | $y \in [-0.65, -0.02]\text{ m},\; z \in [0.12, 0.30]\text{ m}$ | 90 / 120 | **75.00%** | 219 / 480 | **45.62%** |
| **`R2` High-Left** | $y \in [0.05, 0.70]\text{ m},\; z \in [0.80, 1.15]\text{ m}$ | 64 / 120 | **53.33%** | 119 / 477 | **24.95%** |
| **`R3` High-Right** | $y \in [-0.70, -0.05]\text{ m},\; z \in [0.80, 1.15]\text{ m}$ | 92 / 120 | **76.67%** | 159 / 479 | **33.19%** |
| **Mid Height (`R0 + R1`)** | $z \in [0.35, 0.85]\text{ m}$ | 212 / 240 | **88.33%** | 634 / 960 | **66.04%** |
| **Low Height (`R4 + R5`)** | $z \in [0.12, 0.30]\text{ m}$ | 186 / 240 | **77.50%** | 403 / 960 | **41.98%** |
| **High Height (`R2 + R3`)** | $z \in [0.80, 1.15]\text{ m}$ | 156 / 240 | **65.00%** | 278 / 956 | **29.08%** |
| **Left Side (`R0 + R2 + R4`)** | $y > 0\text{ m}$ | 270 / 360 | **75.00%** | 605 / 1,437 | **42.10%** |
| **Right Side (`R1 + R3 + R5`)** | $y < 0\text{ m}$ | 284 / 360 | **78.89%** | 710 / 1,439 | **49.34%** |

*(See `evaluation/k1_save_rate_by_region.png` for the regional breakdown chart.)*

#### C. Analysis of Saves, Misses, and Controlled Fine-Tuning Experiments
1. **Impact of Ball Calibration ($400\text{ g}, r = 0.10\text{ m}$ vs. Size 1 $200\text{ g}, r = 0.075\text{ m}$):** Migrating `v9_5000` from the legacy $20\text{ cm}$ ball (`k1_paired_benchmark_r010.json`) to the RoboCup Size 1 $15\text{ cm}$ ball (`k1_paired_benchmark.json`) reduced the effective contact radius by $2.5\text{ cm}$ and widened the scoreable goal opening by $5.0\text{ cm}$ horizontally and $2.5\text{ cm}$ vertically. Consequently, `v9_5000` random saves shifted from $1,599/2,852$ ($56.07\%$) to $1,315/2,876$ ($45.72\%$), and deterministic `KOLONG_12` blocks dropped from $9/12$ ($75.0\%$) to $4/12$ ($33.3\%$).
2. **Between-the-Legs (`Kolong`) Interception Analysis (Runs A, B, C):** In **Run A (`ctrl_5250`)**, continuing training on the $200\text{ g}$ ball for $250$ iterations without altering reward weights improved `KOLONG_12` from **$4/12$ to $7/12$** (newly blocking shots `903, 907, 908` via `left_knee_pitch_link` at $z \approx 0.28\text{ m}$ and `left_hip_yaw_link` at $z \approx 0.34\text{ m}$ followed by `right_ankle_pitch_link` sweeps) and raised `MOTION_9` to **$9/9$ ($100\%$)**, while random falls rose from $40$ to $74$. Increasing the `_reward_kolong_gap` penalty cap to $-384/\text{s}$ (**Run B**) or removing the `compact_gap` multiplier from `foot_block` (**Run C**) destabilized the stance ($222\text{--}224$ falls and $0\text{--}1/12$ kolong blocks), confirming that excessive leg-closing penalties conflict with lateral balance.
3. **High-Corner Reaction Envelope (Run D):** Trajectory diagnostics on `HIGH_GRID_30` at $z = 1.12\text{ m}$ (`ID 1023, 1024, 1029, 1030`) show that at $12\text{--}14\text{ m/s}$ ($x_0 = 4.50\text{ m}$, arrival time $0.319\text{--}0.372\text{ s}$), the ball crosses the goal line at step $16\text{--}18$ ($320\text{--}360\text{ ms}$). On `v9_5000`, the active hand reaches $z = 0.77\text{--}0.95\text{ m}$ at crossing and peaks at $z = 1.03\text{--}1.07\text{ m}$ at step $20\text{--}22$ (**$80\text{--}120\text{ ms}$ after the ball passes**). Shifting the countermovement-to-takeoff reward threshold from $2.2\text{ m}$ to $3.2\text{ m}$ (**Run D `high32_5200/5250`**) advanced peak hand timing by $20\text{--}40\text{ ms}$, blocked $12\text{ m/s}$ shot `ID 1021` ($z = 0.98\text{ m}$), and increased random high-region (`R2+R3`) saves from $278/956$ ($29.1\%$) to $304/956$ ($31.8\%$), but increased falls to $120\text{--}141$.
4. **Arrival-Time Distribution Alignment (Run E):** Measuring arrival times at $x = 0.08\text{ m}$ revealed that the default fast training generator (`t_fast` $\in [0.42, 0.56]\text{ s}$) produced $0.0\%$ shots arriving in $< 0.404\text{ s}$, whereas $63.8\%$ ($457/716$) of `EVAL 10ms` shots arrive in $0.294\text{--}0.404\text{ s}$. Splitting the $55\%$ fast training portion $50:50$ between $[0.32, 0.48]\text{ s}$ and $[0.42, 0.56]\text{ s}$ (**Run E `fast5050_5200`**) increased `seeds 1+777` random saves to **$1,372/2,876$ ($47.71\%$)** ($+57$ saves over `v9_5000`, $+60$ over `ctrl_5200`) and held-out `seed = 2026` saves to **$634/1,436$ ($44.15\%$)** vs. $616/1,436$ ($42.90\%$) on `v9_5000` and $602/1,436$ ($41.92\%$) on `ctrl_5200`, at the cost of $102$ falls ($53$ on `seed = 2026`) and lower `KOLONG_12` saves. Thus, **`v9_5000` is retained as the primary balanced policy**, with `ctrl_5250` and `fast5050_5200` documenting the specific trade-offs of kolong adaptation and short-arrival training.

---

### VII. DEMONSTRATION VIDEO: SAVES AND MISSES

#### A. Video Evidence
The demonstration footage is generated directly from `play_k1_size1.sh` (`play.py`) using the primary checkpoint `k1_v9/model_5000.pt` on the Size 1 ball ($200\text{ g}$, $r = 0.075\text{ m}$). The visual overlay displays the 3D goal frame ($2.60 \times 1.20\text{ m}$), live target crosshair, and colour-coded goal-line status bar (amber while live, green upon a verified save, red upon a goal conceded).

#### B. Representative Cases (Linked to Evaluation Trial IDs)
1. **Representative Successful Save — Central Low Interception (`KOLONG_12` Trial `ID 905`, $y_{\text{target}} = 0.00\text{ m}, z_{\text{target}} = 0.16\text{ m}$):**
   - *Initial Conditions:* Ball launched from $(3.50, 0.00, 0.18)\text{ m}$ with $\mathbf{v}_0 = (-5.23, 0.00, 3.31)\text{ m/s}$.
   - *Policy Action & Result:* Guided by `kolong_offset`, `v9_5000` shifts its pelvis laterally to $y_{\text{torso}} = -0.125\text{ m}$ while placing its left knee and shin across the center line ($y_{\text{shin,L}} = +0.089\text{ m}$, $y_{\text{foot,L}} = +0.144\text{ m}$). At step $32$ ($t = 0.64\text{ s}$), the ball ($x = 0.230\text{ m}, y = -0.035\text{ m}, z = 0.282\text{ m}$) strikes the left upper shin/hip link (`min_d = 0.124 m`), rebounds forward away from the goal line, and the robot finishes in stable double-foot support (`2ft = 100%`).
2. **Representative Miss — High-Speed Upper-Corner Shot (`HIGH_GRID_30` Trial `ID 1030`, $14.0\text{ m/s}$, $y_{\text{target}} = -0.65\text{ m}, z_{\text{target}} = 1.12\text{ m}$):**
   - *Initial Conditions:* Ball launched from $(4.50, 0.00, 0.18)\text{ m}$ at $14.0\text{ m/s}$ toward the top-right corner ($z = 1.12\text{ m}$, just $5\text{ mm}$ below the full-ball crossbar threshold $1.125\text{ m}$), arriving at the goal line in $0.334\text{ s}$ (step $16$).
   - *Policy Action & Result:* The robot executes a countermovement squat during steps $1\text{--}8$ ($z_{\text{torso}}: 0.540 \rightarrow 0.500\text{ m}$) and initiates upward takeoff at step $9$ ($v_z = +0.18 \rightarrow +0.90\text{ m/s}$). When the ball crosses the goal plane at step $16$ ($z_{\text{ball}} = 1.11\text{ m}$), the right hand is still rising at $z_{\text{hand}} = 0.77\text{ m}$ ($\Delta z = -0.34\text{ m}$ below the ball) and reaches its peak height of $z_{\text{hand}} = 1.03\text{ m}$ at step $22$ ($120\text{ ms}$ after crossing), resulting in a conceded goal.

---

### VIII. LIMITATIONS
Based on measured simulation evidence, the current system has four main limitations:
1. **Kinematic & Reaction-Time Envelope on Top-Corner Shots ($\ge 12\text{ m/s}$, $z \ge 0.98\text{ m}$):** From a standing torso height of $0.54\text{ m}$, K1 requires $\approx 240\text{--}260\text{ ms}$ ($12\text{--}13\text{ steps}$) from the end of its countermovement dip to reach peak hand height ($\approx 1.03\text{--}1.07\text{ m}$). Shots arriving in $320\text{--}360\text{ ms}$ at $z = 1.12\text{ m}$ exceed both the vertical reach and takeoff speed of the current standing-jump strategy.
2. **Asymmetric Outer-Kolong Gap ($y \in [-0.22, -0.15]\text{ m}$):** While `ctrl_5250` blocks $7/12$ kolong shots (including $+0.12\text{ m}$ and $+0.18\text{ m}$ on the left), shots at $y \in [-0.22, -0.15]\text{ m}$ (`ID 901, 902, 910`) still pass between the shins ($y_{\text{shin,L}} \approx +0.03\text{ m}$, $y_{\text{shin,R}} \approx -0.42\text{ m}$) because the active right-hand reach pulls the torso rightward while maintaining a $0.45\text{ m}$ stance width.
3. **Stability vs. Aggressiveness Trade-off:** Fine-tuning for wider leg blocks or shorter arrival times (`Run A`, `Run D`, `Run E`) increases random-shot falls from $40/2,880$ ($1.39\%$) on `v9_5000` to $74\text{--}141/2,880$ ($2.57\%\text{--}4.90\%$).
4. **Unmodeled Aerodynamics & Sim-to-Real Gap:** The simulator models projectile gravity and contact restitution on a flat plane without air drag, Magnus spin forces, artificial turf compliance, or camera perception latency/occlusion.

---

### IX. CONCLUSION
The final demonstration uses `landfix_5350`, which achieved the highest overall save rate among the evaluated landing checkpoints. Final two-foot support remains a limitation. The earlier experiments below establish the baseline and training history.

We presented the design, RoboCup KidSize 2025 physical calibration ($2.60 \times 1.20\text{ m}$ goal, $200\text{ g}$, $r = 0.075\text{ m}$ Size 1 ball), and paired evaluation of a 22-DOF Booster K1 humanoid goalkeeper policy in NVIDIA Isaac Gym. Using a 10-step history encoder with auxiliary ball-landing and region estimators (`v9_5000`), the goalkeeper achieves a **76.94% save rate (554/720)** on standard shots and **45.72% (1,315/2,876)** across all speed modes up to $14\text{ m/s}$, with only **40 falls (1.39%)** across 2,880 trials. Controlled OFAT experiments further demonstrated how ball-size adaptation (`ctrl_5250`, $7/12$ kolong saves, $9/9$ posture-audit saves) and arrival-time curriculum expansion (`fast5050_5200`, $47.71\%$ random saves, $+2.13\%$ over paired control across 4,312 shots) trade off against post-impact standing stability.

---

### REFERENCES
- [1] RoboCup Technical Committee, *RoboCup Humanoid League Laws of the Game 2025*, RoboCup Federation, 2025. [Online]. Available: https://humanoid.robocup.org/wp-content/uploads/RC-HL-2025-Rules.pdf
- [2] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to walk in minutes using massively parallel deep reinforcement learning," in *Proc. Conf. Robot Learning (CoRL)*, PMLR, vol. 164, pp. 91–100, 2022.
- [3] J. Long, Z. Wang, Q. Li, L. Cao, J. Gao, and J. Pang, "Hybrid internal model: Learning agile legged locomotion with simulated robot response," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.
- [4] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, "AMP: Adversarial motion priors for stylized physics-based character control," *ACM Trans. Graph. (SIGGRAPH)*, vol. 40, no. 4, pp. 1–20, 2021.
- [5] J. Ren, J. Long, *et al.*, “Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints,” arXiv:2510.18002, 2025. [Paper](https://arxiv.org/abs/2510.18002)
- [6] X. Huang, Z. Li, Y. Xiang, Y. Ni, Y. Chi, Y. Li, L. Yang, X. B. Peng, and K. Sreenath, "Creating a dynamic quadrupedal robotic goalkeeper with reinforcement learning," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2023, pp. 2715–2722, arXiv:2210.04435. [Paper](https://arxiv.org/abs/2210.04435)
