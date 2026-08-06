# ADR 0004: Disable Reverse-Centric Recovery for Learned Controllers

- Status: Accepted
- Date: 2026-04-04

## Context

Reverse maneuvers repeatedly damaged learned-controller context (especially MBRA indoor), causing oscillation loops and lower effective progress.

## Decision

Prefer reset-and-reacquire strategies over reverse-based recovery when learned visual controllers are active:
- no-reverse mode in strict outdoor profiles,
- stale-context/no-progress reset indoors,
- skip-past-checkpoint handling for forward-only graph edge cases.

## Consequences

### Positive
- Fewer self-induced oscillation loops.
- Better alignment with forward-conditioned model behavior.
- More stable local command sequences.

### Negative
- Reduced maneuver options in constrained spaces.
- Requires stronger target semantics and transition logic.
