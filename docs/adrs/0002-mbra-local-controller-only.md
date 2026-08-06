# ADR 0002: Use MBRA as Indoor Local Controller Only

- Status: Accepted
- Date: 2026-04-04

## Context

Indoor performance improved only after localization and graph semantics were treated as the backbone. Attempts to treat MBRA as a full navigation stack produced confusion about responsibility and recovery behavior.

## Decision

Use MBRA only as short-horizon local control between graph-selected subgoals.
Do not use MBRA as:
- a place localizer,
- a graph planner,
- a global recovery planner.

## Consequences

### Positive
- Clear separation of concerns.
- Better runtime explainability and debugging.
- Cleaner handling of checkpoint-step progression.

### Negative
- Requires more explicit state wiring (localization/planner/controller interfaces).
- MBRA quality depends on subgoal semantics being correct.
