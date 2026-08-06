# MBRA Codebase: File-by-File Explanation

What each file in the [Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) repo does.

---

## Root

| File | Purpose |
|------|---------|
| `README.md` | Setup, datasets (GNM, Frodobots-2K), training commands (MBRA.yaml, LogoNav.yaml), inference (ROS, FrodoBots) |
| `LICENSE` | MIT license |

---

## `train/` — Training Pipeline

### Entry Point & Config

| File | Purpose |
|------|---------|
| **`train.py`** | Main entry point. Loads config (defaults + yaml), builds datasets (Frodobots + GNM), creates model (GNM, ViNT, NoMaD, MBRA/ExAug, LogoNav/IL_gps, LeLaN), sets optimizer/scheduler, and calls the appropriate `train_eval_loop_*`. Dispatches by `model_type` in config. |
| **`config/defaults.yaml`** | Shared defaults (lr, batch size, image size, etc.) |
| **`config/MBRA.yaml`** | MBRA training: `model_type: MBRA`, Frodobots + GNM datasets, `horizon_short: 30`, GNM paths |
| **`config/LogoNav.yaml`** | LogoNav training: `model_type: LogoNav`, `load_exaug` for pretrained MBRA, `horizon_long: 100` |
| **`config/vint.yaml`** | ViNT model config |
| **`config/gnm.yaml`** | GNM model config |
| **`config/nomad.yaml`** | NoMaD model config |
| **`config/late_fusion.yaml`** | Late-fusion variant config |

### Data Processing Scripts

| File | Purpose |
|------|---------|
| **`process_recon.py`** | Converts RECON HDF5 to per-trajectory folders. Reads `recon_release/*.h5`, extracts `jackal/position`, `jackal/yaw`, `images/rgb_left`, writes `traj_data.pkl` and `{i}.jpg` per trajectory. |
| **`process_bags.py`** | Converts ROS bags to trajectory folders. Uses `process_bags_config.yaml` for topic names, extracts images + odom, filters backwards motion, saves `traj_data.pkl` and images. |
| **`process_bag_diff.py`** | Same as `process_bags.py` but for bags with "diff" in the name. Uses `/usb_cam_front/image_raw`, `/chosen_subgoal`, `/odom`. |
| **`data_split.py`** | Splits a dataset folder into train/test. Finds folders with `traj_data.pkl`, shuffles, writes `traj_names.txt` into `data_splits/{dataset}/train/` and `test/`. |
| **`push_dataset_to_hub.py`** | LeRobot script to convert raw data to LeRobot format and push to HuggingFace. Supports `frodobots` via `convert_to_hf`. |

### Environment

| File | Purpose |
|------|---------|
| **`environment_mbra.yml`** | Conda env for MBRA (Python, PyTorch, etc.) |
| **`setup.py`** | Installs `lelan` package via `pip install -e train/` |

---

## `train/vint_train/` — Core Training Logic

### Models

| File | Purpose |
|------|---------|
| **`models/base_model.py`** | Base class for navigation models. Defines `context_size`, `len_traj_pred`, `learn_angle`, `num_action_params` (2 or 4). |
| **`models/exaug/exaug.py`** | **ExAug_dist_delay** — MBRA model. EfficientNet encoders for obs+goal, transformer decoder, outputs `linear_vel`, `angular_vel`, `dist_pred`. Takes `robot_size`, `delay`, `vel_past` for system delay. |
| **`models/exaug/self_attention.py`** | `MultiLayerDecoder`, `MultiLayerDecoder2` — transformer decoder used by ExAug. |
| **`models/exaug/vit.py`** | ViT encoder (used by other models, not ExAug). |
| **`models/il/il.py`** | **IL_dist** (image goal + distance), **IL_gps** (pose goal) — LogoNav policy. IL_gps takes `goal_pose` [rel_x, rel_y, cos(Δyaw), sin(Δyaw)], outputs waypoints. |
| **`models/gnm/gnm.py`** | GNM (Goal-conditioned Neural Motor) model. |
| **`models/gnm/modified_mobilenetv2.py`** | MobileNetV2 backbone for GNM. |
| **`models/vint/vint.py`** | ViNT (Visual Navigation Transformer) model. |
| **`models/vint/vit.py`** | ViT for ViNT. |
| **`models/vint/self_attention.py`** | Attention layers for ViNT. |
| **`models/nomad/nomad.py`** | NoMaD diffusion policy. |
| **`models/nomad/nomad_vint.py`** | NoMaD with ViNT-style encoder. |
| **`models/nomad/vib_placeholder.py`** | Placeholder for ViB encoder. |
| **`models/lelan/lelan.py`** | LeLaN (CLIP + FiLM) for text/image-conditioned nav. |
| **`models/lelan/lelan_comp.py`** | LeLaN_clip_FiLM, LeLaN_clip_FiLM_temp encoders. |
| **`models/lelan/sample_film.py`** | FiLM conditioning utilities. |
| **`models/depth_360.py`** | **Depth_est** — monocular depth for 360° fisheye. ResNet18 encoder + DepthDecoder, outputs 3D point cloud. Used in MBRA for collision loss. |

### Data

| File | Purpose |
|------|---------|
| **`data/vint_dataset.py`** | **ViNT_Dataset** — loads GNM-style data from folders. Reads `traj_names.txt`, loads images + `traj_data.pkl`, samples (obs, goal, action, distance). **ViNT_Dataset_gps**, **ViNT_ExAug_Dataset** variants. |
| **`data/vint_hf_dataset.py`** | **FrodbotDataset_MBRA**, **FrodbotDataset_LogoNav** — load Frodobots from LeRobot/zarr. Use `dataset_cache.zarr`, sample goal within `goal_horizon`, return obs images, goal image, actions, goal_pos, etc. **EpisodeSampler_MBRA** — samples by episode range, uses `train_yaw_small.pkl`, `train_ped_fix.pkl` for filtering. |
| **`data/lelan_dataset.py`** | **LeLaN_Dataset** — for YouTube/LeLaN data. Supports `dataset_name == "youtube"`. |
| **`data/data_utils.py`** | `get_data_path`, `yaw_rotmat`, `to_local_coords`, `calculate_deltas`, `img_path_to_data`, etc. |
| **`data/data_config.yaml`** | Action stats, dataset paths (recon, go_stanford, youtube, etc.). |

### Training

| File | Purpose |
|------|---------|
| **`training/train_eval_loop.py`** | Orchestrates train/eval loops. **train_eval_loop_MBRA** — calls `train_MBRA`, sets up depth model, EpisodeSampler. **train_eval_loop_LogoNav** — loads pretrained MBRA, calls `train_LogoNav`. Also has loops for vint, gnm, nomad, lelan, lelan_col. |
| **`training/train_utils.py`** | **train_MBRA** — one-epoch MBRA training. Batches Frodobots + GNM, runs ExAug, integrates velocities → poses, computes dist_loss, loss_geo, diff_loss, dist_temp_loss, backprops. **train_LogoNav** — runs MBRA in eval to get actions for Frodobots, concatenates with GNM actions, trains IL_gps with `_compute_losses_gps`. **robot_pos_model_fix** — integrates (linear_vel, angular_vel) → (px, pz, yaw) via unicycle. **twist_to_pose_diff_torch** — single-step integration. **geometry_criterion_range** — collision loss from depth point cloud. **_compute_losses**, **_compute_losses_gps** — loss functions. **evaluate_MBRA**, **evaluate_LogoNav** — eval loops. |
| **`training/logger.py`** | **Logger** — moving-average logger for metrics (loss, etc.). |

### Visualization

| File | Purpose |
|------|---------|
| **`visualizing/action_utils.py`** | **visualize_traj_pred** — plots predicted vs label waypoints on images. |
| **`visualizing/distance_utils.py`** | **visualize_dist_pred** — plots distance predictions. |
| **`visualizing/visualize_utils.py`** | `to_numpy`, `numpy_to_img`, color constants. |

### Depth Model Networks

| File | Purpose |
|------|---------|
| **`training/networks/resnet_encoder.py`** | ResNet encoder for depth. |
| **`training/networks/depth_decoder.py`** | Depth decoder. |
| **`training/networks/depth_decoder_camera_ada3.py`** | Camera-adaptive depth decoder. |
| **`training/networks/depth_decoder_camera_ada4.py`** | Used by Depth_est. |
| **`training/networks/depth_decoder_share.py`** | Shared decoder variant. |
| **`training/networks/pose_decoder.py`** | Pose decoder. |
| **`training/networks/pose_cnn.py`** | Pose CNN. |
| **`training/networks/layers.py`** | `BackprojectDepth_fisheye_inter_offset`, etc. |
| **`training/networks/__init__.py`** | Package init. |

---

## `deployment/` — Inference

| File | Purpose |
|------|---------|
| **`LogoNav_frodobot.py`** | FrodoBots deployment. Loads LogoNav (IL_gps) from `logonav.pth`. 3 Hz loop: fetches front image from `http://127.0.0.1:8000/v2/front`, GPS from `http://127.0.0.1:8000/data`, computes relative goal pose from UTM, runs `model(obs_images, goal_pose_torch)`, converts waypoint index 2 to (linear_vel, angular_vel) via `v=dx/DT`, `ω=arctan(dy/dx)/DT`, clips, sends to `http://127.0.0.1:8000/control`. Goal set at lines 219–220. |
| **`LogoNav_ros.py`** | ROS deployment. Subscribes to `/usb_cam/image_raw`, uses fixed goal/pose at lines 104–106, 243–244. Same policy logic as FrodoBots. |
| **`utils_logonav.py`** | **load_model** — loads checkpoint by `model_type` (gnm, vint, nomad, exaug_dist_gnm_delay, il2_gps, lelan, etc.). **transform_images_mbra** — ImageNet normalize + resize. **to_numpy**, **clip_angle** — utilities. |

---

## Config Summary

| Config | Model | Use |
|--------|-------|-----|
| `MBRA.yaml` | ExAug_dist_delay | Train short-horizon reannotator |
| `LogoNav.yaml` | IL_gps | Train long-horizon policy (needs pretrained MBRA) |
| `vint.yaml` | ViNT | Baseline |
| `gnm.yaml` | GNM | Baseline |
| `nomad.yaml` | NoMaD | Diffusion baseline |

---

## Data Flow Summary

1. **Frodobots-2K**: LeRobot/zarr → `FrodbotDataset_MBRA` / `FrodbotDataset_LogoNav` + `EpisodeSampler_MBRA`
2. **GNM**: Folder + `traj_names.txt` → `ViNT_Dataset`, `ViNT_ExAug_Dataset`, `ViNT_Dataset_gps`
3. **RECON**: HDF5 → `process_recon.py` → folder + `traj_data.pkl` + images
4. **Train MBRA**: `train.py -c MBRA.yaml` → `train_eval_loop_MBRA` → `train_MBRA`
5. **Train LogoNav**: `train.py -c LogoNav.yaml` → `train_eval_loop_LogoNav` → `train_LogoNav` (uses MBRA for reannotation)
6. **Deploy**: `LogoNav_frodobot.py` or `LogoNav_ros.py` → `utils_logonav.load_model` → run policy loop
