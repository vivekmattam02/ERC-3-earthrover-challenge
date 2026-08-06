# MBRA Algorithm: Complete Technical Specification

A core-level specification of the Model-Based ReAnnotation (MBRA) algorithm from [Learning to Drive Anywhere with MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) (Hirose et al., 2025).

---

## 1. Overview

MBRA is a two-stage pipeline:

1. **Stage 1 (MBRA):** Train a short-horizon, model-based expert that predicts velocity commands from (observation, goal image). This expert is used to **reannotate** noisy teleop data.
2. **Stage 2 (LogoNav):** Train a long-horizon policy conditioned on **GPS/pose goals** (not goal images) using MBRA-reannotated Frodobots data + original GNM data.

At inference, only LogoNav runs; it uses images + GPS, no goal images.

---

## 2. Data Preparation

### 2.1 Frodobots-2K

- **Source:** [HuggingFace frodobots/FrodoBots-2K](https://huggingface.co/datasets/frodobots/FrodoBots-2K)
- **Conversion:** Use [catglossop/frodo_dataset](https://github.com/catglossop/frodo_dataset) → `convert_to_hf.py` to produce LeRobot format.
- **Output:** Zarr-based dataset with:
  - `observation.images.front` (video)
  - `observation.filtered_position`, `observation.filtered_heading`
  - `observation.relative_position`
  - `action` (linear/angular velocities)
  - `episode_index`, `timestamp`, `frame_index`
- **Sampler files:** `train_yaw_small.pkl`, `train_ped_fix.pkl`, `test_yaw_small.pkl`, `test_ped_fix.pkl` (from HuggingFace) for episode sampling.
- **Split:** Episodes 0 to `split_train_test` (e.g. 10794) = train; rest = test. `ratio_f=0.9` → 90% train.

### 2.2 GNM Dataset Mixture

- **Datasets:** recon, go_stanford, cory_hall, tartan_drive, sacson, seattle, scand
- **Format:** Per-dataset folders with `traj_names.txt` in train/test splits.
- **recon:** HDF5 → process with `process_recon.py` to get `traj_data.pkl` + images per trajectory.
- **Structure:** Each trajectory has `position` (N×2), `yaw`, and images.

### 2.3 Image Preprocessing

- **Size:** 96×96 (config `image_size: [96, 96]`)
- **Normalization:** ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Context:** 5 frames, spacing 3 (MBRA) or 3 (LogoNav)
- **Random horizontal flip:** 50% during training

---

## 3. MBRA Model: ExAug_dist_delay

### 3.1 Architecture

| Component | Details |
|-----------|---------|
| **Obs encoder** | EfficientNet-B0, 3 channels, extracts features → compress to `obs_encoding_size` (1024) |
| **Goal encoder** | EfficientNet-B0: `obsgoal_img = concat(obs_last_frame, goal_img)` (6 ch) or late_fusion (3 ch) |
| **Decoder** | MultiLayerDecoder: embed_dim=1024, seq_len=context_size+16, output_layers=[256,128,64,32], 4 heads, 4 layers |
| **Tokens** | `obs_encoding` (5+1 frames) + `goal_encoding` + `robot_size` + `delay` + `vel_past` |
| **Action head** | Linear(32 → 8×2) + Sigmoid → linear_vel [0, 0.5], angular_vel [-1, 1] |
| **Dist head** | Linear(32 → 1) |

### 3.2 Inputs

- `obs_img`: [B, 18, 96, 96] — 6 frames × 3 channels, flattened
- `goal_img`: [B, 3, 96, 96]
- `robot_size`: [B, 1, 1] — random 0–1 (robot radius in m)
- `delay`: [B, 1, 1] — random 0–4 (temporal delay steps)
- `vel_past`: [B, 12, 1] — 6 steps of (linear_vel, angular_vel) concatenated

### 3.3 Outputs

- `linear_vel`: [B, 8] — m/s, range [0, 0.5]
- `angular_vel`: [B, 8] — rad/s, range [-1, 1]
- `dist_pred`: [B, 1] — scalar distance estimate

### 3.4 Velocity → Pose Integration

**`twist_to_pose_diff_torch(v, w, dt=0.333):`**

```
theta = -w * dt
z = v * dt * sinc(θ/π)
x = -v * dt * sinc(θ/(2π)) * sin(-θ/2)
```

**`robot_pos_model_fix(linear_vel, angular_vel):`**

- For each of 8 steps: compute (x, z, θ) from twist, build 4×4 transform, accumulate.
- Returns `px`, `pz`, `pyaw` — lists of length 8 (camera frame: x right, z forward, yaw from arctan).

---

## 4. MBRA Training

### 4.1 Data Mix

- **Frodobots:** 90% of batch (`batch_frodobot`), via `EpisodeSampler_MBRA`
- **GNM:** 10% (`batch_gnm`), shuffled

### 4.2 Frodobots Sample (FrodbotDataset_MBRA)

For index `idx`:

1. Sample `goal_dist` ∈ [0, min(goal_horizon, episode_remaining)] (frames ahead).
2. Load context images at `idx` and goal image at `idx + goal_dist`.
3. Compute:
   - `goal_pos_relative` = goal position in current robot frame
   - `local_goal_mat` = 3×3 transform from current to goal pose
   - `action` = (forward_vel, steer) from position/heading differences, normalized by dt
4. **Negative mining:** If `goal_dist == 0`, set `distance = 20` and replace goal with random other goal image.

### 4.3 GNM Sample (ViNT_ExAug_Dataset)

- Standard ViNT-style (obs, goal_image, action_label, dist_label) from GNM splits.

### 4.4 MBRA Loss (per batch)

**Combined batch:** Frodobots (Bf) + GNM (Bg)

**Augmentations:**

- `rsize` = rand(Bf+Bg, 1, 1) ∈ [0, 1]
- `delay` = randint(Bf+Bg, 1, 1) ∈ [0, 4]
- `vel_past` = previous 6 steps of (linear_vel, angular_vel); zeros after `delay`
- For `goal_dist==0`: replace goal with random goal from batch

**Forward:**

```
linear_vel, angular_vel, dist_temp = model(obs, goal, rsize, delay, vel_past)
```

**Pose integration:**

- `linear_vel_d` = concat(vel_past_trimmed, linear_vel)
- `angular_vel_d` = concat(angular_vel_past_trimmed, angular_vel)
- `px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)`

**Goal representation:**

- `last_pose_mat` = 3×3 transform from (pz, -px, ry) in robot frame (weight_position=2.0)
- `local_goal_mat` = ground-truth relative transform to goal
- `local_goal_vec` = repeated for 8 steps

**Loss terms:**

1. **dist_loss:** MSE(last_pose_mat, local_goal_mat) — goal reaching
2. **distall_loss:** MSE(robot_traj_vec[6:14], local_goal_vec) — trajectory alignment
3. **diff_loss:** MSE(linear_vel[:,:-1], linear_vel[:,1:]) + same for angular — smoothness
4. **loss_geo:** Geometry loss from depth-estimated point cloud — collision avoidance
5. **dist_temp_loss:** MSE(dist_temp, combined_distance) — distance prediction
6. **social_loss, personal_loss:** Placeholder (0)

**Total loss:**

```
L = 4.0*dist_loss + 0.4*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 0.001*dist_temp_loss
```

### 4.5 Geometry Loss (Collision Avoidance)

- **Depth model:** Depth_est (ResNet18 encoder + DepthDecoder_camera_ada4), pretrained.
- **Input:** Current image resized to 128×416, masked by `mask_360view.csv`.
- **Output:** Point cloud `proj_3d` in camera frame.
- **Process:** For each predicted pose at steps 6–13, transform point cloud to that frame. `geometry_criterion_range` penalizes points that would be inside robot radius `rsize` along the path.
- **Purpose:** Encourage trajectories that avoid obstacles.

### 4.6 Training Config (MBRA.yaml)

- epochs: 200, batch_size: 100, lr: 1e-5, optimizer: adamw
- scheduler: cosine, warmup: 4 epochs
- horizon_short: 30, len_traj_pred: 8, context_size: 5
- ratio_f: 0.9 (90% Frodobots)

---

## 5. LogoNav Model: IL_gps

### 5.1 Architecture

| Component | Details |
|-----------|---------|
| **Obs encoder** | EfficientNet-B0 (same as MBRA) |
| **Goal encoding** | No image. `goal_pose` [B, 4] → Linear(4 → 1024) |
| **Decoder** | MultiLayerDecoder, seq_len=context_size+2 |
| **Action head** | Linear(32 → 8×4); cumsum on [:2]; normalize [2:4] to unit vector |

### 5.2 Inputs

- `obs_img`: [B, 18, 96, 96] — 6 context frames
- `goal_pose`: [B, 4] — `[rel_x_norm, rel_y_norm, cos(Δyaw), sin(Δyaw)]`

### 5.3 Outputs

- `action_pred`: [B, 8, 4] — waypoints as `[x, z, cos(yaw), sin(yaw)]` in normalized space
- Deltas are cumsummed for position; orientation is normalized.

---

## 6. LogoNav Training (Reannotation)

### 6.1 Data Mix

- **Frodobots:** 90%, via `FrodbotDataset_LogoNav` + `EpisodeSampler_MBRA`
- **GNM:** 10%, via `ViNT_Dataset_gps` (GPS-based actions)

### 6.2 Frodobots Sample (FrodbotDataset_LogoNav)

- `goal_dist` = frames to primary goal
- `goal_dist2` = min(8, remaining) — for MBRA’s inverse model goal
- `goal_dist_gps` = frames to GPS goal
- Returns: obs, goal_image, goal_image2, actions (from teleop), action_IL (from poses), goal_pos, local_goal_mat, etc.
- `action_mask`: True when goal_dist ∈ (3, 20) and goal ≠ current frame

### 6.3 Reannotation Step (Core of MBRA)

For each Frodobots batch:

1. **MBRA forward (eval, no grad):**
   ```
   linear_vel, angular_vel, _ = model_mbra(obs_image, goal_image2, rsize=0.3, delay=0, vel_past=straight)
   ```

2. **Integrate to poses:**
   ```
   px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
   ```

3. **Convert to LogoNav action format:**
   ```
   metric_waypoint_spacing = 0.25 * 0.5
   action_estfrod = [z/metric, -x/metric, cos(-yaw), sin(-yaw)] for each of 8 steps
   ```

4. **Replace Frodobots actions with MBRA actions:**
   ```
   combined_actions = [action_estfrod, action_label_sub]  # Frodobots | GNM
   ```

5. **LogoNav forward:**
   ```
   combined_goal_pos_gps = [goal_pose_gps_frodobots, goal_pose_gps_gnm]
   combined_action_pred = model(obs, combined_goal_pos_gps)
   ```

6. **Loss:** `_compute_losses_gps(combined_actions, combined_action_pred, action_mask)`

### 6.4 Goal Pose Format (GPS)

- **Frodobots:** `goal_pose_gps = [goal_pos, cos(goal_yaw), sin(goal_yaw)]` from `local_goal_mat`
- **GNM:** Same from dataset’s goal position and yaw
- **Normalization:** `goal_pos / metric_waypoint_spacing` (0.25 m)

### 6.5 Training Config (LogoNav.yaml)

- load_exaug: ../deployment/model_weights (MBRA checkpoint)
- horizon_short: 20, horizon_long: 100
- normalize: True
- Same datasets as MBRA

---

## 7. Inference (LogoNav_frodobot.py)

### 7.1 Loop (3 Hz)

1. **Fetch:** Front image from `http://127.0.0.1:8000/v2/front`, GPS from `http://127.0.0.1:8000/data`
2. **Context:** Maintain queue of 6 images (context_size+1)
3. **Goal pose:**
   - `cur_utm`, `cur_compass` from robot
   - `goal_utm`, `goal_compass` from user
   - `delta_x, delta_y` = goal - current in UTM
   - `relative_x, relative_y` = rotate to robot frame
   - Cap distance at 30 m for normalization
   - `goal_pose = [relative_y/0.25, -relative_x/0.25, cos(Δyaw), sin(Δyaw)]`
4. **Forward:** `waypoints = model(obs_images, goal_pose)`
5. **Waypoint → velocity:**
   - Use waypoint index 2 (third waypoint)
   - `chosen_waypoint[:2] *= (0.3 / 3.0)` if normalized
   - `dx, dy, hx, hy = chosen_waypoint`
   - If |dx|,|dy| ≈ 0: linear=0, angular=arctan2(hy,hx)/DT
   - Else: linear=dx/DT, angular=arctan(dy/dx)/DT, DT=0.25
   - Clip linear ∈ [0, 0.5], angular ∈ [-1, 1]
6. **Velocity limiting:** Enforce maxv=0.3, maxw=0.3 with turn radius logic
7. **Send:** POST to `http://127.0.0.1:8000/control` with `{linear, angular}`

### 7.2 Coordinate Conventions

- **Robot frame:** x forward, y left (or per SDK)
- **Compass:** Frodobots orientation negated and converted to rad
- **Waypoint:** [x, z, cos(yaw), sin(yaw)] in normalized coords; z≈forward, x≈lateral in camera frame

---

## 8. File and Config Summary

| File | Purpose |
|------|---------|
| `train.py` | Entry point; loads config, datasets, models; dispatches to train loops |
| `config/MBRA.yaml` | MBRA training config |
| `config/LogoNav.yaml` | LogoNav training config |
| `vint_train/models/exaug/exaug.py` | ExAug_dist_delay (MBRA) |
| `vint_train/models/il/il.py` | IL_gps (LogoNav) |
| `vint_train/data/vint_hf_dataset.py` | FrodbotDataset_MBRA, FrodbotDataset_LogoNav, EpisodeSampler_MBRA |
| `vint_train/training/train_utils.py` | train_MBRA, train_LogoNav, robot_pos_model_fix, geometry_criterion_range |
| `deployment/LogoNav_frodobot.py` | FrodoBots deployment |
| `deployment/utils_logonav.py` | load_model, transform_images_mbra |

---

## 9. Checklist: What You Need to Do

### Data

- [ ] Download Frodobots-2K, convert with frodo_dataset
- [ ] Download GNM mixture (recon, go_stanford, etc.), create train/test splits
- [ ] Download sampler PKLs for Frodobots
- [ ] Build `dataset_cache.zarr` for Frodobots

### MBRA Training

- [ ] Set paths in MBRA.yaml (root, data_folder, data_splits)
- [ ] Download Depth_est weights (depthest_ploss/)
- [ ] Run: `python train.py -c ./config/MBRA.yaml`
- [ ] Save best checkpoint to `deployment/model_weights/mbra.pth`

### LogoNav Training

- [ ] Set `load_exaug` in LogoNav.yaml to MBRA checkpoint path
- [ ] Run: `python train.py -c ./config/LogoNav.yaml`
- [ ] Save to `deployment/model_weights/logonav.pth`

### Deployment

- [ ] Clone earth-rovers-sdk, run hypercorn server
- [ ] Copy LogoNav_frodobot.py, utils_logonav.py into SDK utils/
- [ ] Set goal lat/lon and yaw (lines 219–220)
- [ ] Run: `python LogoNav_frodobot.py`

---

## 10. Key Design Choices

1. **Why reannotate Frodobots?** Teleop is noisy; MBRA produces cleaner, goal-directed actions.
2. **Why keep GNM raw?** GNM has accurate GPS and good actions; no need to reannotate.
3. **Why GPS at inference?** Enables long-horizon nav without storing goal images; same as ERC.
4. **Why depth for MBRA?** Geometry loss encourages collision-free trajectories.
5. **Why delay/vel_past?** Simulates latency and history; improves robustness.
