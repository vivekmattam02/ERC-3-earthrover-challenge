# Documentation Architecture

Type: Documentation System Plan  
Status: Roadmap / partly implemented  
Scope: Whole repo

## Purpose

This file defines the documentation structure that should be used for the ERC-3 EarthRover Challenge repo. It describes the intended docs-as-code layout; it is not itself a claim that every proposed path exists.

The implemented curated current-truth layer is the `obsidian_vault/` tree.
For current behavior, use active code plus the vault's source-of-truth notes;
for example, the no-GPS field maturity is recorded in
`obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md`.

The repo already contains a large amount of written material:
- current guides
- runtime explanations
- experiment reports
- historical narratives
- handoff notes
- code deep reads

The main problem is no longer lack of documentation. The main problem is that these materials are mixed together without a clear authority model.

This architecture reorganizes the documentation into a system that separates:
- current truth
- operator procedures
- technical reference
- architecture / explanation
- decision records
- historical reports
- experiment evidence

This structure is informed by:
- Diataxis / Divio: tutorials, how-to, reference, explanation
- Docs-as-Code practices from Write the Docs
- architecture structuring ideas similar to arc42

## Core Principle

Not every document should try to do everything.

For this project, each doc should be one of the following:
- `Overview`: what the project is, what it does, current maturity
- `Getting Started`: setup and first run
- `How-To`: concrete operational steps
- `Reference`: exact facts, flags, parameters, log fields, paths
- `Architecture / Explanation`: why the system is structured the way it is
- `Decision Record`: a short record of a design decision and its consequences
- `Experiment / Evidence`: offline probes, evaluations, failures, analyses
- `Historical Report`: story-style writeups preserving the engineering journey

## Canonical Documentation Structure

The recommended structure for this repo is:

```text
docs/
  index.md
  overview/
    current-system.md
    project-scope.md
  getting-started/
    setup.md
    first-indoor-run.md
    first-outdoor-run.md
  howto/
    run-indoor.md
    run-outdoor-mission.md
    run-outdoor-marathon.md
    preflight-marathon.md
    resume-mission.md
    debug-bad-run.md
  architecture/
    indoor-stack.md
    outdoor-stack.md
    perception-stack.md
    safety-architecture.md
  reference/
    runtime-flags.md
    models-and-weights.md
    logging-and-telemetry.md
    sdk-and-mission-endpoints.md
    file-map.md
  decisions/
    ADR-001-indoor-controller-mbra.md
    ADR-002-no-reverse-for-mbra.md
    ADR-003-outdoor-controller-logonav.md
    ADR-004-osm-route-corridor-guard.md
    ADR-005-dynamic-intermediate-waypoint-radius.md
    ADR-006-semantic-runtime-gating.md
  experiments/
    depth-traversability.md
    semantic-segmentation.md
    offline-calibration.md
  reports/
    *.tex
```

## What Goes Where

### 1. Overview

Purpose:
- tell a new reader what the project is
- explain the indoor vs outdoor split
- state the current maturity honestly
- state what the system is not

This section should answer:
- what does the rover do?
- what are the indoor and outdoor tracks?
- what are the main stacks currently used?
- what is stable and what is still experimental?

### 2. Getting Started

Purpose:
- help a new teammate get the environment running
- avoid relying on scattered setup notes

This section should answer:
- which conda env is used?
- which model weights must exist?
- which services must be running?
- what is the first dry run command?

### 3. How-To

Purpose:
- operator-facing procedures
- concrete step-by-step workflows

This section should include:
- how to run indoor
- how to run outdoor mission mode
- how to run ultra-marathon / night-safe
- how to do preflight
- how to resume a mission
- how to stop and recover from bad behavior
- how to interpret the most important log patterns

### 4. Architecture

Purpose:
- explain why the system is structured the way it is
- preserve engineering understanding without turning every file into a story

This section should include:
- indoor stack: localizer, graph planner, MBRA, simple controller fallback
- outdoor stack: mission mode, OSM routing, LogoNav, rerouting, corridor guard
- perception stack: depth, traversability, semantic risk, vision safety
- safety architecture: IMU, GPS/telemetry, battery, vision, route corridor, semantics, operator intervention

### 5. Reference

Purpose:
- exact facts only
- code-grounded information
- easy lookup

This section should include:
- CLI flags and defaults
- thresholds and parameter values
- model IDs, weight locations, config paths
- log field meanings
- telemetry field meanings
- SDK endpoints and mission semantics
- file map for the important source files

### 6. Decisions

Purpose:
- keep major design decisions out of long narrative docs
- make future changes easier to evaluate

Each ADR should contain:
- context
- decision
- alternatives considered
- consequences

### 7. Experiments

Purpose:
- preserve offline evidence and exploratory results in a concise form
- bridge between short ADRs and long reports

These are not meant to replace the `.tex` reports.
They are meant to summarize the evidence in a maintainable docs-as-code format.

### 8. Reports

Purpose:
- preserve the long-form engineering story
- keep the historical narrative and detailed analyses
- provide professor/interviewer/judge-readable technical writeups

These should mostly be the existing `.tex` files.

## Existing Documentation: Recommended Mapping

### Current repo-root files

| Existing file | Recommended role | Notes |
|---|---|---|
| `README.md` | Overview / entrypoint | Keep short; should point into `docs/index.md` |
| `CLAUDE.md` | Developer guide source material | Useful content, but should not remain the only current guide |
| `CONTEXT.md` | Living engineering context plus historical support | Its field-status override is current support material; it is still not the sole canonical user-facing doc |
| `guide.md` | Candidate how-to source material | Fold into structured `howto/` docs |
| `structure.md` | Candidate file-map source material | Fold into `reference/file-map.md` |
| `outdoor_controller.md` | Architecture / design note / history hybrid | Mine for `architecture/outdoor-stack.md` and reports |
| `chat_handoff.md` | Historical handoff | Keep as support material, not canonical |
| `chat_session_reconstructed.md` | Historical reconstruction | Support material only |
| `codex_handoff_prompt.md` | Historical workflow artifact | Not canonical docs |

### Existing top-level `.tex` files

| Existing file | Recommended role | Notes |
|---|---|---|
| `live_indoor_runtime_story.tex` | Report / historical narrative | Indoor story and lessons learned |
| `live_outdoor_runtime_explained.tex` | Architecture + reference source material | Good source for `architecture/outdoor-stack.md` and `reference/runtime-flags.md` |
| `live_outdoor_ultra_marathon_story.tex` | Report / historical narrative | Marathon-specific engineering story |
| `outdoor_perception_review.tex` | Experiment / evidence report | Perception history and conclusions |
| `semantic_segmentation_research_review.tex` | Experiment / evidence report | Semantic work and runtime implications |

### Existing `docs/` files

| Existing file | Recommended role | Notes |
|---|---|---|
| `docs/team_start_here.tex` | Historical onboarding report | Keep under reports |
| `docs/march19.tex` | Historical report | Indoor progress history |
| `docs/our_mbra_discoveries.tex` | Decision history + technical report | Also source material for ADRs |
| `docs/current_codebase_deep_read.tex` | Architecture / codebase report | Good source for architecture docs |
| `docs/MBRA_ALGORITHM_SPEC.md` | Reference / architecture source material | Useful for indoor architecture + reference |
| `docs/MBRA_CODE_FILE_BY_FILE.md` | Reference source material | Good basis for `reference/file-map.md` |
| `docs/indoor_navigation_strategy.tex` | Historical strategy report | Keep under reports |
| `docs/known_corridor_runtime_plan.tex` | Historical planning report | Keep under reports |
| `docs/discoveries.tex` | Historical findings report | Keep under reports |
| `docs/laserfocus.tex` | Historical / experimental | Keep under reports |
| `docs/nyu_indoor_track.tex` | Historical / track-specific | Keep under reports |
| `docs/our_approach.tex` | Historical / high-level narrative | Keep under reports |
| `docs/stack_breakdown.tex` | Architecture source material | Good input for architecture docs |
| `docs/CONTEXT.md` | Internal context dump | Not canonical docs |

## Minimum New Canonical Docs To Create

These are the highest-value missing docs.

### Tier 1: Must Have
- `docs/index.md`
- `docs/overview/current-system.md`
- `docs/howto/run-indoor.md`
- `docs/howto/run-outdoor-mission.md`
- `docs/howto/run-outdoor-marathon.md`
- `docs/reference/runtime-flags.md`
- `docs/reference/models-and-weights.md`
- `docs/reference/logging-and-telemetry.md`
- `docs/architecture/indoor-stack.md`
- `docs/architecture/outdoor-stack.md`

### Tier 2: Strongly Recommended
- `docs/architecture/perception-stack.md`
- `docs/architecture/safety-architecture.md`
- `docs/howto/preflight-marathon.md`
- `docs/howto/resume-mission.md`
- `docs/howto/debug-bad-run.md`
- `docs/reference/sdk-and-mission-endpoints.md`
- `docs/reference/file-map.md`

### Tier 3: Decision Layer
- `docs/decisions/ADR-001-indoor-controller-mbra.md`
- `docs/decisions/ADR-002-no-reverse-for-mbra.md`
- `docs/decisions/ADR-003-outdoor-controller-logonav.md`
- `docs/decisions/ADR-004-osm-route-corridor-guard.md`
- `docs/decisions/ADR-005-dynamic-intermediate-waypoint-radius.md`
- `docs/decisions/ADR-006-semantic-runtime-gating.md`

## Documentation Writing Rules For This Repo

These rules should govern future docs work.

### Rule 1: Mark doc type and status
Each canonical doc should begin with metadata like:
- `Type: How-To`
- `Status: Current`
- `Audience: Operator` or `Audience: Developer`

### Rule 2: Separate current truth from history
- Current docs should describe the current repo state.
- Reports should preserve the story and detours.
- A report should not be treated as the only source of current truth.

### Rule 3: Make reference docs code-grounded
Reference docs should be verified against actual code:
- flags
- thresholds
- defaults
- paths
- model IDs
- runtime behavior gates

### Rule 4: Keep long reports, but link them
Do not throw away the `.tex` files. They are valuable.
But they should sit under `docs/reports/` conceptually and be linked from summary pages instead of standing in for the whole documentation layer.

### Rule 5: Update docs with behavior changes
When runtime behavior changes, update:
- one current doc
- one reference doc
- and an ADR if the change is architectural

## Best Next Actions

The best next implementation order is:

1. create `docs/index.md`
2. create `docs/overview/current-system.md`
3. create `docs/howto/run-indoor.md`
4. create `docs/howto/run-outdoor-marathon.md`
5. create `docs/reference/runtime-flags.md`
6. create `docs/architecture/indoor-stack.md`
7. create `docs/architecture/outdoor-stack.md`
8. create the ADR folder and first 3--6 ADRs

This is better than continuing to deepen random files because the repo already has enough raw material. The main need now is a clean, authoritative structure.

## Practical Interpretation

If someone asks, “where do I learn the project?”, the answer should become:
- start at `docs/index.md`
- read `docs/overview/current-system.md`
- use `docs/howto/` to run things
- use `docs/reference/` to check exact values
- use `docs/architecture/` to understand why the stack is designed this way
- use `docs/reports/` for the full engineering story and evidence

That is what will make the documentation actually work.
