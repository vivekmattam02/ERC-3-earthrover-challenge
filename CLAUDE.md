# ERC Indoor Corridor Navigation — System Guide

## Quick Start

### Terminal 1 — SDK server
```bash
sudo fuser -k 8000/tcp 2>/dev/null
cd ~/Desktop/rover_desktop/ERC-3-earthrover-challenge/earth-rovers-sdk
conda activate erv
hypercorn main:app --reload
```

### Terminal 2 — Start mission
```bash
curl -X POST http://127.0.0.1:8000/start-mission
```

### Terminal 3 — Run indoor controller (MBRA)
```bash
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

**Dry run** (no commands sent, for testing):
```bash
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --controller mbra --depth-safety
```

## Architecture

```
live_indoor_runtime.py          Main control loop (3 Hz with MBRA)
  |
  +-- NavigationRuntime          Coordinates localization + planning
  |     +-- CorridorLocalizer    CosPlace VPR (ResNet18, 512-dim, 1865 DB images)
  |     |     +-- TemporalLocalizer   Stabilizes frame-wise retrieval
  |     +-- GraphPlanner         Topological graph, subgoal selection
  |
  +-- MBRALocalController        Learned visual navigation policy
  |     (or SimpleLocalController)
  +-- SensorStateFilter          Heading/gyro/RPM smoothing
  +-- DepthEstimator             Monocular depth for obstacle avoidance (optional)
  +-- EarthRoverInterface        SDK HTTP client
```

## How It Works

1. **Localize**: Each camera frame is encoded via CosPlace into a 512-dim descriptor.
   Nearest-neighbor search against the corridor database (1865 images) gives candidate
   positions. Temporal filtering penalizes large jumps and backward motion to stabilize.

2. **Plan**: The graph planner finds the shortest path from current position to the
   active checkpoint. A subgoal is selected N hops ahead (8 for MBRA, 15 for simple).

3. **Control**: MBRA receives the current frame (6-frame history) and the subgoal
   *image* to produce (linear, angular) commands. It's a learned policy that steers
   toward visual targets. The simple controller uses heading error only.

4. **Safety**: Monocular depth estimates forward clearance. Hard stop at 0.25m (MBRA)
   or 0.4m (simple). Speed scaling between stop and slow thresholds.

## Key Design Decisions

### MBRA is sole motion authority
When using `--controller mbra`, the runtime does NOT override MBRA's steering or trigger
reverse maneuvers. Recovery backup, angular saturation override, and RPM stall backup
are all disabled for MBRA. Reversing destroys MBRA's 6-frame visual context and causes
oscillation. The only runtime interventions are:
- **Hard stop** on depth < 0.25m or low confidence (no reverse)
- **Speed reduction** near obstacles (depth_slow) and near target (proximity slowdown)
- **Context reset** if stuck at same step for 10+ ticks

### Compass disabled for localization
Indoor compass is unreliable (motor interference). `observation_heading_deg=None` is
passed to the localizer. Heading from the compass is still used by the simple controller
for gyro-based drift correction during forward drive.

### Forward-only graph
The corridor graph is directed — no backward paths. If the robot localizes past a
checkpoint, it skips to the next one instead of trying to reverse.

## Tuning Parameters

### MBRA Controller (`src/mbra_controller.py`)
| Parameter | Default | Notes |
|---|---|---|
| max_linear | 0.40 m/s | Upper speed limit |
| min_linear | 0.18 m/s | Not enforced — MBRA can output 0 |
| max_angular | 0.34 rad/s | Turning limit |
| min_confidence | 0.45 | Below this → stop |
| low_confidence_linear_scale | 0.7 | Speed reduction when conf < 0.60 |
| vel_past_linear | 0.5 | Fixed input to model (not feedback) |
| vel_past_angular | 0.0 | Fixed input to model |

### Runtime (`live_indoor_runtime.py`)
| Parameter | Default | Notes |
|---|---|---|
| tick_hz | 3.0 (mbra) / 2.0 (simple) | Control loop rate |
| max_subgoal_hops | 8 (mbra) / 15 (simple) | How far ahead to pick subgoal |
| depth_stop_m | 0.25 (mbra) / 0.4 (simple) | Hard stop distance |
| depth_slow_m | 0.6 (mbra) / 0.8 (simple) | Start slowing distance |
| NO_PROGRESS_RESET_TICKS | 10 | Reset MBRA context if stuck |
| JUMP_REJECT_THRESHOLD | 30 | Max step jump before requiring higher confidence |
| PROXIMITY_SLOWDOWN_STEPS | 15 | Slow down within this many steps of target |

### Temporal Localizer (`src/temporal_localization.py`)
| Parameter | Default | Notes |
|---|---|---|
| top_k | 10 | Candidates considered |
| max_step_jump | 20 | Continuity penalty kicks in above this |
| jump_penalty | 0.05 | Per-step penalty for large jumps |
| backward_penalty | 0.15 | Per-step penalty for going backward |
| ambiguity_margin | 0.05 | Hold position if top-2 scores are this close |

## File Map

| File | Purpose |
|---|---|
| `live_indoor_runtime.py` | Main runtime loop with safety layers |
| `src/corridor_localizer.py` | CosPlace VPR + temporal filtering |
| `src/temporal_localization.py` | Continuity-aware position filtering |
| `src/graph_planner.py` | Topological graph, checkpoint management |
| `src/navigation_runtime.py` | Localization + planning coordinator |
| `src/mbra_controller.py` | MBRA learned visual navigation policy |
| `src/local_controller.py` | Simple heading-based local controller |
| `src/sensor_state.py` | Heading/gyro/RPM EMA filtering |
| `src/earthrover_interface.py` | SDK HTTP interface |
| `src/depth_estimator.py` | Depth Anything V2 monocular depth |
| `baseline.py` | CosPlace descriptor utilities + DB building |
| `data/corrider_db/` | Corridor database (descriptors.npz, navigation_graph.json) |

## Corridor Database

- **1865 images** from a recorded corridor traversal
- **1024x576 RGBA** source images, CosPlace encodes at 512x512
- **512-dimensional** L2-normalized descriptors (CosPlace ResNet18)
- **11 checkpoints**: steps 45, 480, 761, 821, 1094, 1208, 1345, 1430, 1544, 1638, 1764

## MBRA Model Details

- **Type**: ExAug_dist_delay (loaded as `exaug_dist_gnm_delay`)
- **Input**: 6-frame observation context (96x96) + 1 goal image (96x96)
- **Additional inputs**: robot_size=0.3, delay=0.0, vel_past (fixed: linear=0.5, angular=0.0)
- **Output**: 8-step trajectory of (linear_vel, angular_vel, distance). Index [0] = immediate command
- **Linear range**: [0, 0.5] m/s. **Angular range**: [-1.0, 1.0] rad/s
- **Weights**: `mbra_repo/deployment/model_weights/mbra.pth`
- **Config**: `mbra_repo/train/config/MBRA.yaml`

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| Robot oscillates back and forth | Recovery backup fighting MBRA | Use `--controller mbra` (backups disabled) |
| `runtime_no_path_stop` repeating | Robot past checkpoint in directed graph | `--auto-advance-checkpoints` skips past |
| Localization hallucinating | Compass heading adding noise | Already fixed: heading=None for localizer |
| MBRA stuck, not moving | Low confidence or depth stop | Check logs for `depth_stop` or `low_confidence` |
| `FileNotFoundError` on subgoal image | Path mismatch from different machine | Fixed: path resolution via `/data/` marker |

## Outdoor System

The outdoor GPS navigation system is separate. See `live_outdoor_runtime.py` and
`src/outdoor_gps_controller.py`. Run with:
```bash
python live_outdoor_runtime.py --mission --send-control --traversability
```
