# Docs Index

Type: Overview  
Status: Current documentation map / partly historical inventory  
Audience: Developer / Operator / Reviewer

## Purpose

This is the entrypoint for the project documentation.

The repo contains three different kinds of written material:
- canonical current docs
- deep technical reports and historical writeups
- internal handoff/context artifacts

The goal of this index is to make it clear where to start and where to go next.

The curated current-truth layer is in `obsidian_vault/`. The field maturity of
the no-GPS rough-terrain branch is owned by
[No-GPS Field Trial - Findings](../obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md):
the pipeline works, but reliable autonomous repeat has not been demonstrated.

## Start Here

If you are new to the project, read these in order:

1. [README](../README.md)
2. [Current No-GPS - Read This First](../obsidian_vault/00%20Home/Current%20No-GPS%20-%20Read%20This%20First.md)
3. [Vault Home](../obsidian_vault/00%20Home/Vault%20Home.md)
4. [CLAUDE.md](../CLAUDE.md) for the historical indoor reference
5. [Documentation Architecture](./DOCUMENTATION_ARCHITECTURE.md) for the documentation roadmap

## Most Important Existing Technical Reports

Indoor:
- [Indoor Runtime Story](../live_indoor_runtime_story.tex)
- [MBRA Discoveries](./our_mbra_discoveries.tex)
- [Current Codebase Deep Read](./current_codebase_deep_read.tex)
- [March 19 Retrospective](./march19.tex)

Outdoor:
- [No-GPS Rough-Terrain Route Repeat](../no_gps_route_repeat_story.tex)
- [Outdoor Runtime Explained](../live_outdoor_runtime_explained.tex)
- [Outdoor Ultra Marathon Story](../live_outdoor_ultra_marathon_story.tex)
- [Outdoor Perception Review](../outdoor_perception_review.tex)
- [Semantic Segmentation Research Review](../semantic_segmentation_research_review.tex)
- [Outdoor Controller Notes](../outdoor_controller.md)

## Current Problem

The repo has a curated current-truth layer, but older reports and planning notes
still live alongside it.

The remaining documentation task is maintenance: update current evidence when
runtime behavior changes, while retaining older reports as historical context.

## Documentation Roadmap

The structure to build is documented in:
- [Documentation Architecture](./DOCUMENTATION_ARCHITECTURE.md)

The highest-value new docs to create next are:
- `docs/overview/current-system.md`
- `docs/howto/run-indoor.md`
- `docs/howto/run-outdoor-mission.md`
- `docs/howto/run-outdoor-marathon.md`
- `docs/reference/runtime-flags.md`
- `docs/reference/models-and-weights.md`
- `docs/reference/logging-and-telemetry.md`
- `docs/architecture/indoor-stack.md`
- `docs/architecture/outdoor-stack.md`
- `docs/adrs/` (decision log)

## Existing Docs By Role

### Current guides / source material
- [No-GPS Field Trial - Findings](../obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md)
- [Vault Home](../obsidian_vault/00%20Home/Vault%20Home.md)
- [CLAUDE.md](../CLAUDE.md) (historical indoor reference)
- [guide.md](../guide.md)
- [structure.md](../structure.md)
- [ADR Log](./adrs/README.md)
- [MBRA Algorithm Spec](./MBRA_ALGORITHM_SPEC.md)
- [MBRA Code File By File](./MBRA_CODE_FILE_BY_FILE.md)

### Reports / historical narratives
- [No-GPS Rough-Terrain Route Repeat](../no_gps_route_repeat_story.tex)
- [Indoor Runtime Story](../live_indoor_runtime_story.tex)
- [Outdoor Ultra Marathon Story](../live_outdoor_ultra_marathon_story.tex)
- [Outdoor Runtime Explained](../live_outdoor_runtime_explained.tex)
- [Outdoor Perception Review](../outdoor_perception_review.tex)
- [Semantic Segmentation Research Review](../semantic_segmentation_research_review.tex)
- [Our MBRA Discoveries](./our_mbra_discoveries.tex)
- [Current Codebase Deep Read](./current_codebase_deep_read.tex)
- [March 19](./march19.tex)

### Internal context / handoff artifacts
- [CONTEXT.md](../CONTEXT.md)
- [docs/CONTEXT.md](./CONTEXT.md)
- [chat_handoff.md](../chat_handoff.md)
- [chat_session_reconstructed.md](../chat_session_reconstructed.md)
- [codex_handoff_prompt.md](../codex_handoff_prompt.md)

## Documentation Rule Going Forward

When runtime behavior changes, update:
- the relevant source-of-truth note
- the operator/reference document that exposes the behavior
- an ADR only if the change is architectural

That is the minimum needed to keep the documentation system healthy.
