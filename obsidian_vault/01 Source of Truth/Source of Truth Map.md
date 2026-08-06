# Source of Truth Map

This note defines what I should trust first when different notes disagree.

## Decision Order

Use this order when sources disagree:

1. **Code** decides what the runtime is capable of doing now.
2. **Field evidence** decides whether that capability actually worked on the rover.
3. **Current-status notes** summarize the resulting decision.
4. **Narratives and reports** explain why the system took this shape.
5. **Archive notes** preserve history and never override the first four.

Implemented does not mean field-proven.

## Code First

- `live_indoor_runtime.py`
- `scripts/run_prepared_route.py`
- `src/adaptive_pursuit_controller.py`
- `src/corridor_localizer.py`
- `live_outdoor_runtime.py`
- `src/temporal_localization.py`
- `src/graph_planner.py`
- `src/mbra_controller.py`
- `src/outdoor_logonav_controller.py`
- `src/outdoor_traversability.py`
- `src/semantic_risk_estimator.py`
- `src/imu_safety.py`

## Documentation First

- [[00 Home/Current No-GPS - Read This First]]
- [[01 Source of Truth/No-GPS Field Trial - Findings]]
- [[erc3_full_documentation]]
- [[competition_results]]
- [[final_system_overview]]

## Companion Notes

- [[live_indoor_runtime_story]]
- [[live_outdoor_ultra_marathon_story]]
- [[live_outdoor_runtime_explained]]
- [[outdoor_perception_review]]
- [[semantic_segmentation_research_review]]
- [[docs/current_codebase_deep_read]]
- [[docs/our_mbra_discoveries]]

## Raw Context / Reference

- [[90 Archive/CONTEXT - Indoor Source]]
- [[90 Archive/CONTEXT - Research Source]]
- [[90 Archive/MBRA Algorithm Spec]]
- [[90 Archive/MBRA Code File By File]]
- [[90 Archive/Documentation Architecture]]
- [[90 Archive/ERC-3 Mathematics and Intuition Guide]]

Archive notes preserve decisions and source material. They may describe an
earlier system state and must not override code or the field-trial evidence.

## Workspace Boundaries

| Path | Role | Operational status |
| --- | --- | --- |
| `live_indoor_runtime.py` + `scripts/run_prepared_route.py` | Maintained route-repeat runtime and launcher | Active code path |
| `src/adaptive_pursuit_controller.py` | Rough-terrain local controller | Active no-GPS controller family |
| `legacy/` | Older MBRA-first and recovery runtime forks | Historical only; do not launch them |
| `mbra_repo/` | Optional MBRA model workspace and weights | Only relevant for explicit `--controller mbra` work; not the active no-GPS controller |
| `mbra_repo_1/` | Image-memory/MBRA research snapshot | Reference only; no model weights or active integration |
| `earth-rovers-sdk/` | Local SDK bridge and control endpoint | Required runtime dependency |
| `nyu-earthrover-main/` | Ignored local baseline clone | Reference only; not the SDK server used here |
| `obsidian_vault/` | Curated project narrative and current evidence | Separate Git repository with its own remote |
| `.claude/` | Local assistant permission metadata | Not project or field evidence |
| `.github/workflows/docs-ci.yml` | Documentation checks | Does not validate rover behavior |

## Personal Fast Notes

- [[03 Personal Notes/Current Truth]]
- [[03 Personal Notes/Architecture in My Words]]
- [[03 Personal Notes/Things I Keep Forgetting]]
