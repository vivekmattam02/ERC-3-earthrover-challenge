# ERC-3 Documentation Index

This file is the map of the documentation set. The repository now contains three
different kinds of writing:

- current operating documents and field evidence that describe what actually runs,
- narrative technical reports that explain how the system evolved,
- historical planning notes that are still useful, but should not be treated as
  the current source of truth.

The curated current-truth layer lives in `obsidian_vault/`. In particular,
`obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md` is the
authority for the maturity of the current no-GPS rough-terrain branch. It is
implemented but has not demonstrated a reliable physical autonomous repeat.

If you are trying to understand the project quickly, do not open files at
random. Use one of the reading paths below.

## Start Here

| If you are... | Read in this order |
|---|---|
| A new teammate | `README.md` → `CLAUDE.md` → `guide.md` → `obsidian_vault/00 Home/Vault Home.md` |
| Evaluating no-GPS work | `Current No-GPS - Read This First.md` → `No-GPS Field Trial - Findings.md` → active runtime code |
| A professor / judge / reviewer | `README.md` → `competition_results.tex` → `final_system_overview.tex` → `live_outdoor_ultra_marathon_story.tex` |
| A returning developer | `CLAUDE.md` → the subsystem-specific report below |

## ADR Log

Architecture decisions are now tracked in:
- `docs/adrs/README.md`
- `docs/adrs/0001-indoor-outdoor-split.md`
- `docs/adrs/0002-mbra-local-controller-only.md`
- `docs/adrs/0003-outdoor-logonav-layered-runtime.md`
- `docs/adrs/0004-no-reverse-for-learned-controllers.md`
- `docs/adrs/0005-osm-waypoint-semantics.md`

## Complete Inventory

| File | Lines | Description | Status | Last meaningful update |
|---|---:|---|---|---|
| `README.md` | 279 | Repository landing page, setup notes, high-level stack split | Current | 2026-04-01 |
| `CLAUDE.md` | 168 | Historical indoor guide with a current-status warning | Current reference | 2026-07-16 |
| `guide.md` | 227 | Practical run guide for indoor and outdoor workflows | Current | untracked |
| `no_gps_route_repeat_story.tex` | 266 | Full no-GPS route-repeat design, field evidence, and validation gate | Current narrative | 2026-07-16 |
| `obsidian_vault/00 Home/Current No-GPS - Read This First.md` | new | Short operating brief: problem contract, reference decision, and maturity | Current | 2026-07-16 |
| `obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md` | new | Field evidence, current maturity, and validation gate for no-GPS repeat | Current evidence | 2026-07-16 |
| `structure.md` | 433 | Annotated repository tree and directory purpose map | Current | 2026-04-01 |
| `outdoor_controller.md` | 743 | Outdoor planning document plus post-competition status review | Current | 2026-04-01 |
| `competition_results.tex` | new | Race-day narrative and honest competition outcomes | Current | 2026-04-04 |
| `final_system_overview.tex` | new | One-document architecture overview for the full project | Current | 2026-04-04 |
| `live_indoor_runtime_story.tex` | 559 | Indoor system story: checkpoint-step mode, MBRA, failures, fixes | Current narrative | untracked |
| `live_outdoor_runtime_explained.tex` | 525 | Plain-language walkthrough of the outdoor runtime | Current narrative | untracked |
| `live_outdoor_ultra_marathon_story.tex` | 600 | Outdoor marathon story and later runtime hardening | Current narrative | untracked |
| `outdoor_perception_review.tex` | 586 | Depth / traversability / semantics investigation | Current narrative | untracked |
| `semantic_segmentation_research_review.tex` | 680 | Semantic segmentation research trail from probe to runtime | Current narrative | untracked |
| `competition_technical_report_2026_03_31.tex` | 276 | Competition-facing technical report snapshot | Reference snapshot | untracked |
| `docs/march19.tex` | 1035 | Mid-March indoor retrospective and architecture reasoning | Current historical narrative | untracked |
| `docs/our_mbra_discoveries.tex` | 705 | MBRA post-mortem and indoor controller lessons | Current historical narrative | untracked |
| `docs/current_codebase_deep_read.tex` | 536 | Codebase survey from before the system stabilized | Historical but useful | untracked |
| `docs/discoveries.tex` | 489 | Literature and method review (ViNT, PlaceNav, MBRA, related ideas) | Reference / historical | untracked |
| `docs/indoor_navigation_strategy.tex` | 524 | Early indoor strategy note | Historical | 2026-03-19 |
| `docs/known_corridor_runtime_plan.tex` | 593 | Early indoor runtime plan | Historical | 2026-03-19 |
| `docs/laserfocus.tex` | 248 | Scope-tightening document from early indoor phase | Historical | untracked |
| `docs/our_approach.tex` | 265 | Early high-level approach overview | Historical | 2026-03-19 |
| `docs/team_start_here.tex` | 194 | Early onboarding note for teammates | Historical | 2026-03-19 |
| `docs/stack_breakdown.tex` | 608 | Module-by-module breakdown of the older stack | Historical reference | 2026-03-19 |
| `docs/nyu_indoor_track.tex` | 308 | Indoor track rules and competition framing | Reference | 2026-03-19 |
| `docs/MBRA_ALGORITHM_SPEC.md` | 348 | MBRA algorithm interpretation and parameter notes | Reference | 2026-03-19 |
| `docs/MBRA_CODE_FILE_BY_FILE.md` | 145 | File-by-file MBRA code reference | Reference | 2026-03-19 |
| `docs/DOCUMENTATION_ARCHITECTURE.md` | 334 | Meta-document explaining how the docs are organized | Current reference | untracked |
| `CONTEXT.md` | 965 | Living engineering context; current field-status override at top, historical detail below | Current support | 2026-07-16 |
| `docs/CONTEXT.md` | 234 | Early indoor context note | Historical archive | 2026-03-19 |
| `chat_handoff.md` | 321 | Session handoff artifact | Archive | untracked |
| `chat_session_reconstructed.md` | 732 | Reconstructed conversation log | Archive | untracked |
| `codex_handoff_prompt.md` | 324 | Prior handoff instructions for code/doc audit | Archive | untracked |

## Cross-Reference Matrix

| Document | Indoor Perception | Indoor Control | Outdoor Control | Outdoor Safety | MBRA / LogoNav | Perception Research | Competition Results | Architecture |
|---|---|---|---|---|---|---|---|---|
| `README.md` | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `CLAUDE.md` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| `guide.md` | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |
| `competition_results.tex` | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `final_system_overview.tex` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `live_indoor_runtime_story.tex` | ✓ | ✓ |  |  | ✓ |  | ✓ | ✓ |
| `live_outdoor_runtime_explained.tex` |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| `live_outdoor_ultra_marathon_story.tex` |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `outdoor_perception_review.tex` |  |  |  | ✓ |  | ✓ |  | ✓ |
| `semantic_segmentation_research_review.tex` |  |  |  | ✓ |  | ✓ |  |  |
| `docs/march19.tex` | ✓ | ✓ |  |  | ✓ |  |  | ✓ |
| `docs/our_mbra_discoveries.tex` |  | ✓ |  |  | ✓ |  |  | ✓ |
| `docs/current_codebase_deep_read.tex` | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| `docs/nyu_indoor_track.tex` |  |  |  |  |  |  | ✓ |  |

## Dependency Order

- `README.md` is the outermost orientation layer.
- Active code and current field evidence are the system-truth layer. For no-GPS work, start with `No-GPS Field Trial - Findings.md`.
- `CLAUDE.md` is a useful indoor reference, not the sole current truth layer.
- `guide.md` depends on `README.md` and `CLAUDE.md`.
- `competition_results.tex` depends on `README.md`, `CLAUDE.md`, `docs/nyu_indoor_track.tex`, and the indoor/outdoor story files.
- `final_system_overview.tex` depends on `CLAUDE.md`, `live_indoor_runtime.py`, `live_outdoor_runtime.py`, and the major `src/` modules.
- `live_indoor_runtime_story.tex` assumes familiarity with the indoor runtime files and `docs/our_mbra_discoveries.tex`.
- `live_outdoor_runtime_explained.tex` and `live_outdoor_ultra_marathon_story.tex` assume familiarity with `live_outdoor_runtime.py`, `outdoor_controller.md`, and the outdoor perception notes.
- `outdoor_perception_review.tex` and `semantic_segmentation_research_review.tex` depend on `src/semantic_risk_estimator.py`, `src/outdoor_traversability.py`, and the probe scripts.
- Historical planning notes are useful for intent and evolution, but should be read only after the current docs above.

## Which Document Owns What

- If you need current no-GPS behavior: `obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md` and active code.
- If you need the historical indoor reference: `CLAUDE.md`.
- If you need exact run commands and operator flow: `guide.md`.
- If you need the race-day story: `competition_results.tex`.
- If you need the big-picture architecture: `final_system_overview.tex`.
- If you need the indoor story: `live_indoor_runtime_story.tex`.
- If you need the outdoor story: `live_outdoor_ultra_marathon_story.tex`.
- If you need the perception story: `outdoor_perception_review.tex` and `semantic_segmentation_research_review.tex`.
