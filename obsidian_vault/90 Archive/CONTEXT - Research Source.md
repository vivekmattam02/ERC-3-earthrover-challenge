# Rover Project — Context File

**Purpose:** Persistent knowledge base for this project. Updated with each Q&A. Read at the start of new conversations.

**Last updated:** 2025-03-07

---

## 1. Project Overview

- **Rover repo:** CityWalker–EarthRover integration for autonomous urban navigation
- **Earth Rover Challenge (ERC):** Real-world robotics competition (FrodoBots) where AI and human gamers do the same missions on small sidewalk robots over 4G
- **NYU role (ERC 3):** Indoor track (6MTC, 5MTC, image goals) + campus outdoor track (MetroTech, GPS)
- **CityWalker:** CVPR 2025 navigation model deployed on FrodoBots EarthRover Zero

---

## 2. Model-Based Reannotation (MBRA) — Core Concept

**Source:** [Learning to Drive Anywhere with MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) (Hirose et al., 2025, arXiv:2505.05592)

### What It Is

- **MBRA** = short-horizon, model-based expert that **relabels** or **generates** actions for:
  - Crowd-sourced teleop (e.g. FrodoBots-2K)
  - Unlabeled data (e.g. YouTube videos)
- **LogoNav** = long-horizon policy conditioned on visual goals or GPS waypoints, trained on MBRA-reannotated data

### Key Insight: Model-Based Learning (MBL)

MBRA does **not** imitate noisy human actions. It optimizes through a **differentiable forward model**:

- **Forward model** \(f\): given current observation \(O_c\) and actions \(\{a^s_i\}\), produces states \(\{s^s_i\}\)
- **State** \(s_i = [\hat{p}_i, \hat{c}_i, \Delta a^s_i]\): pose, collision score, action smoothness
- **Objective:** \(\min \sum_i (s^{ref} - s^s_i)^2\) with \(s^{ref} = [p_g, 0, 0]\)
- Gradients flow: loss → poses/collision/smoothness → velocities → MBRA policy \(\pi^s\)

The forward model (kinematics + depth) is **fixed**; only the MBRA policy is trained.

---

## 3. MBRA Pipeline (Two Stages)

### Stage 1: Train MBRA (ExAug_dist_delay)

- **Inputs:** obs (5 context frames), goal image, robot_size, delay, vel_past
- **Outputs:** 8-step linear/angular velocities
- **Training:** MBL with dist_loss, loss_geo (collision), diff_loss (smoothness)
- **Data:** FrodoBots-2K (90%) + GNM mixture (recon, go_stanford, cory_hall, etc.)

### Stage 2: Train LogoNav (IL_gps)

- **Reannotation:** For Frodobots batches, MBRA (frozen) generates actions from (obs, goal_image2); these replace human labels
- **GNM batches:** Keep original GPS-based actions
- **LogoNav input:** obs + goal_pose = [rel_x, rel_y, cos(Δyaw), sin(Δyaw)]
- **LogoNav output:** 8 waypoints [dx, dy, cos(yaw), sin(yaw)]

### Inference

- **LogoNav only:** Images + GPS → relative goal pose → waypoints → (v, ω) via simple kinematic conversion
- **No goal images** at deployment for the GPS-conditioned variant

---

## 4. Implementation Details (from code)

### MBRA Model (ExAug_dist_delay)

- EfficientNet-B0 encoders, transformer decoder
- dt = 0.333 s for velocity→pose integration
- `robot_pos_model_fix` = unicycle integration
- Depth model (Depth_est) for collision; `geometry_criterion_range` penalizes points within robot radius

### LogoNav (IL_gps)

- Goal encoded via MLP from 4D pose, not image
- Waypoints: cumsum on position, L2-normalize angles
- Deployment: 3 Hz, waypoint index 2 → v = dx/DT, ω = arctan(dy/dx)/DT, DT=0.25

### System Delay

- `delay` (0–5 steps) and `vel_past` (6 previous velocities) model 4G latency
- Prevents overshooting around target trajectories

---

## 5. Data

| Dataset | Role | Format |
|---------|------|--------|
| FrodoBots-2K | Main passive data; reannotated by MBRA | LeRobot/zarr via frodo_dataset |
| GNM mixture | Clean expert data (recon, go_stanford, etc.) | Per-dataset folders + traj_names.txt |
| YouTube (LeLaN) | Action-free video; MBRA generates labels | LeLaN_Dataset, dataset_name="youtube" |

---

## 6. Deployment (LogoNav_frodobot.py)

- Fetches frames: `http://127.0.0.1:8000/v2/front`
- GPS: `http://127.0.0.1:8000/data`
- Control: `http://127.0.0.1:8000/control`
- Goal pose set at lines 219–220 (lat/lon, yaw)

---

## 7. Earth Rover Challenge (from .tex docs)

- **ERC 1:** IROS 2024, Abu Dhabi — outdoor GPS, best AI ~36%
- **ERC 2:** ICRA 2025, Atlanta — NUS GeNIE 79%, zero interventions
- **ERC 3:** Expected IROS 2026 — indoor (NYU), off-road (GMU), global urban
- **NYU tracks:** 6MTC/5MTC indoor (image goals), MetroTech campus outdoor (GPS, 15 m radius)

---

## 8. MBRA Code: File-by-File Reference

**Full doc:** `docs/MBRA_CODE_FILE_BY_FILE.md`

### Summary

| Area | Key Files |
|------|-----------|
| **Entry** | `train/train.py` — main; loads config, datasets, model; dispatches to train_eval_loop_* |
| **Config** | `config/MBRA.yaml`, `config/LogoNav.yaml` — image_size, datasets, model params |
| **Data scripts** | `process_recon.py` (HDF5→folders), `process_bags.py` (ROS bags), `data_split.py` (train/test) |
| **MBRA model** | `models/exaug/exaug.py` — ExAug_dist_delay |
| **LogoNav** | `models/il/il.py` — IL_gps |
| **Depth** | `models/depth_360.py` — Depth_est for collision loss |
| **Datasets** | `vint_hf_dataset.py` (Frodobots), `vint_dataset.py` (GNM) |
| **Training** | `train_utils.py` — train_MBRA, train_LogoNav, robot_pos_model_fix, geometry_criterion_range |
| **Deploy** | `LogoNav_frodobot.py`, `LogoNav_ros.py`, `utils_logonav.py` |

---

## 9. Image Size (96×96) — Why and Control

### Why 96×96?

- **Compute/latency:** Small enough for 3 Hz inference over 4G (~500 ms round-trip)
- **Consistency:** MBRA, LogoNav, NoMaD use 96×96; GNM/ViNT use 85×64
- **EfficientNet:** Accepts variable input sizes; 96×96 is a speed/resolution tradeoff
- Paper does not explicitly justify the choice

### Where It's Controlled

| Component | Configurable? | How |
|-----------|---------------|-----|
| MBRA training | Yes | `config/MBRA.yaml` → `image_size: [96, 96]` |
| LogoNav training | Yes | `config/LogoNav.yaml` → `image_size: [96, 96]` |
| ROS deployment | Yes | `LogoNav_ros.py` reads `model_params["image_size"]` |
| **FrodoBots deployment** | **No** | `LogoNav_frodobot.py` line 85: hardcoded `newsize = (96, 96)` |

### Data Flow

- **Config** → `train.py` → datasets (`FrodbotDataset_MBRA`, `FrodbotDataset_LogoNav`, `ViNT_Dataset`) → `TF.resize(img, self.image_size)`
- **LogoNav_frodobot.py** does NOT load config for image size; it hardcodes (96, 96)
- **Fix:** Load LogoNav config and use `model_params["image_size"]` like `LogoNav_ros.py`

### Hardcoded 96×96 Elsewhere

- `train_utils.py` lines 908, 910, 2146, 2148: LeLaN+NoMaD evaluation only (not MBRA/LogoNav)

---

## 10. Indoor Use & Training Data

### Did they train on indoor data?

| Dataset | Indoor? |
|---------|---------|
| **FrodoBots-2K** | Mostly outdoor urban (sidewalks, streets) |
| **GNM mixture** | Mixed: cory_hall (indoor), recon, go_stanford, tartan_drive, etc. |
| **YouTube (LeLaN)** | "Inside and outside walking tours from 32 countries" |

### Indoor evaluation

- Paper: "indoor and outdoor environments"; VizBot evaluated **indoors**; "four indoor + four outdoor trajectories" for YouTube
- Indoor data is in the mix but **outdoor is primary**

### Will it work indoors?

- May work to some degree; indoor performance likely weaker than outdoor
- For strong indoor (e.g. NYU 6MTC/5MTC): add indoor data, fine-tune, or use image-goal + topological graph

---

## 11. EarthRover Mini+ Compatibility

**Yes.** earth-rovers-sdk supports both Zero and Mini (`BOT_TYPE`). Same HTTP API (`/v2/front`, `/data`, `/control`). LogoNav_frodobot does not depend on bot type.

**Check:** (1) API compatibility, (2) action scaling if wheelbase/speed differs, (3) camera resolution/FOV if different.

---

## 12. Image-Conditioned vs GPS Navigation

### What the released code has

- **LogoNav (IL_gps)** — GPS/pose-conditioned only (LogoNav_frodobot)
- **MBRA (ExAug)** — Image-conditioned: `(obs, goal_image) → velocities`

### How image-conditioned deployment works (from paper)

1. **Build topological graph:** Teleoperate, record images at ~1 Hz; each node = one image
2. **At runtime:** Get current image → find closest node → use **next** node's image as goal → run MBRA → repeat until goal reached
3. MBRA is short-horizon (~3 m); chain nodes for long distances

### What's NOT in the repo

- No image-conditioned deployment script
- No topological graph construction
- No "closest node" / "next node" logic
- Only GPS-based LogoNav deployment is released

### To add image-conditioned nav

1. Load **MBRA model** (`mbra.pth` from HuggingFace)
2. **Build topological graph:** Drive route, save images at ~1 Hz, store as nodes with connectivity
3. **Deployment script:** Load MBRA → get current image → find closest node (e.g. visual similarity) → next node as goal → `model(obs, goal_image)` → send velocities → loop

---

## 13. Q&A Log (append below)

*New exchanges will be appended here for continuity.*

| Date | Topic | Key points |
|------|-------|------------|
| 2025-03-07 | MBRA overview | Two-stage pipeline, MBL vs imitation, reannotation flow |
| 2025-03-07 | MBRA code file-by-file | Created `MBRA_CODE_FILE_BY_FILE.md`; summary in §8 |
| 2025-03-07 | 96×96 image size | Config control in YAML; LogoNav_frodobot hardcodes it |
| 2025-03-07 | Indoor, Mini+, image-goal | Indoor: mixed training data; Mini+: compatible; image-goal: use MBRA + topological graph, not in repo |

---
