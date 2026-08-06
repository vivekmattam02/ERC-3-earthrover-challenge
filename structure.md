# Repository Structure

This is the current repository map. It is not a raw terminal dump. The point is to
show what exists now and what each major directory is for.

## Top Level

```text
ERC-3-earthrover-challenge/
├── baseline.py                         # Corridor DB build/query utilities
├── live_indoor_runtime.py             # Main indoor runtime
├── live_outdoor_runtime.py            # Main outdoor runtime
├── mbra_gps.py                        # Older outdoor MBRA/GPS experiment
├── new_mbra_gps.py                    # Refactored outdoor MBRA/GPS experiment
├── T_gps_navigator.py                 # Pure-Python GPS math helpers
├── README.md
├── CLAUDE.md
├── guide.md
├── outdoor_controller.md
├── CONTEXT.md
├── chat_handoff.md
├── chat_session_reconstructed.md
├── codex_handoff_prompt.md
├── live_indoor_runtime_story.tex
├── live_outdoor_runtime_explained.tex
├── live_outdoor_ultra_marathon_story.tex
├── outdoor_perception_review.tex
├── semantic_segmentation_research_review.tex
├── competition_technical_report_2026_03_31.tex
├── competition_results.tex
├── final_system_overview.tex
├── docs/
├── obsidian_vault/                    # Curated current-truth and project-story notes
├── legacy/                            # Superseded indoor runtime/controller snapshots
├── src/
├── scripts/
├── tools/
├── data/
├── earth-rovers-sdk/
├── mbra_repo/
├── models/
├── test_outdoor/
└── third_party/
```

## `src/` — Shared Runtime Modules

```text
src/
├── corridor_localizer.py
├── temporal_localization.py
├── graph_planner.py
├── navigation_runtime.py
├── local_controller.py
├── adaptive_pursuit_controller.py
├── mbra_controller.py
├── sensor_state.py
├── earthrover_interface.py
├── depth_estimator.py
├── depth_safety.py
├── outdoor_gps_controller.py
├── outdoor_logonav_controller.py
├── osm_router.py
├── outdoor_traversability.py
├── semantic_risk_estimator.py
├── imu_safety.py
└── vision_safety_monitor.py
```

Purpose:

- indoor localization and planning,
- indoor and outdoor controllers,
- shared safety and SDK wrappers.

## `scripts/` — Diagnostics, Probes, and Calibration

```text
scripts/
├── preflight_marathon.py
├── diagnose_localization.py
├── prelocalize_checkpoints.py
├── calibrate_traversability.py
├── sweep_trav.py
├── probe_semantics.py
├── semantic_second_pass.py
├── record_sdk_session.py              # Front-first manual H5 recorder
├── prepare_manual_route.py            # H5 -> visual route package
├── visualize_manual_route.py          # Route contact-sheet utility
├── run_prepared_route.py              # No-GPS route-repeat launcher
├── semantic_corridor_probe.py
├── semantic_corridor_debug/
├── semantic_debug/
├── semantic_debug_v2/
└── trav_debug/
```

The four debug directories are generated outputs, not core source.

## `legacy/` — Superseded Indoor Experiments

`legacy/` keeps the earlier MBRA-first runtime, heavy-recovery runtime, and
old `mbra_repo_1` controller adapter for provenance. They are not current run
entrypoints. Use `live_indoor_runtime.py` for every new indoor or no-GPS run.

## Reference Workspaces

- `mbra_repo/` supplies the optional current MBRA model path.
- `mbra_repo_1/` is a smaller historical image-memory research snapshot. It
  has no model weights and is not in the active runtime path.
- `nyu-earthrover-main/` is an ignored local baseline clone for reference;
  `earth-rovers-sdk/` is the local SDK server used by this project.

## `tools/` — Offline Data Utilities

```text
tools/
├── extract_h5_dataset.py
├── evaluate_temporal_localization.py
└── verify_workspace.py
```

## `data/` — Indoor Memory and Raw Recording

```text
data/
├── corrider.h5
├── corrider_extracted/
│   ├── front_images/
│   └── metadata/
├── corrider_db/
└── corrider_db_step5/
```

`front_images/` contains many extracted frames and is intentionally not expanded
line-by-line here.

## `earth-rovers-sdk/` — SDK Bridge

```text
earth-rovers-sdk/
├── main.py
├── browser_service.py
├── rtm_client.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── index.html
├── static/
├── examples/
└── assets/
```

## `mbra_repo/` — Imported Research Workspace

```text
mbra_repo/
├── deployment/
│   ├── LogoNav_frodobot.py
│   ├── LogoNav_ros.py
│   ├── utils.py
│   ├── utils_logonav.py
│   └── model_weights/
├── train/
│   ├── config/
│   ├── train.py
│   ├── setup.py
│   └── vint_train/
└── README.md
```

## `docs/` — Long-Form Reports and References

```text
docs/
├── INDEX.md
├── DOCUMENTATION_ARCHITECTURE.md
├── MBRA_ALGORITHM_SPEC.md
├── MBRA_CODE_FILE_BY_FILE.md
├── CONTEXT.md
├── current_codebase_deep_read.tex
├── discoveries.tex
├── indoor_navigation_strategy.tex
├── known_corridor_runtime_plan.tex
├── laserfocus.tex
├── march19.tex
├── nyu_indoor_track.tex
├── our_approach.tex
├── our_mbra_discoveries.tex
├── stack_breakdown.tex
└── team_start_here.tex
```

Several of these `.tex` files are historical planning notes. Current runtime
truth lives in active code plus the source-of-truth notes in `obsidian_vault/`.
For no-GPS rough-terrain work, start with
`01 Source of Truth/No-GPS Field Trial - Findings.md`; `CLAUDE.md` is the
historical indoor-system reference.
