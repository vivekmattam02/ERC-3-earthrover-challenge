# ADR 0001: Split Indoor and Outdoor Architectures

- Status: Accepted
- Date: 2026-04-04

## Context

The project originally described navigation as one generic problem. In practice, indoor and outdoor runs failed for different reasons:
- Indoor: corridor aliasing, checkpoint-step semantics, graph progression issues.
- Outdoor: mission/waypoint transitions, reroute semantics, safety gating interactions.

Treating both tracks as one architecture made debugging slower and design tradeoffs unclear.

## Decision

Maintain two explicit runtime architectures:
- Indoor runtime centered on corridor localization + temporal stabilization + graph progression + MBRA local control.
- Outdoor runtime centered on mission checkpoints + OSM routing + LogoNav/GPS local control + layered runtime safety.

## Consequences

### Positive
- Failures map to the right subsystem faster.
- Configuration defaults and safety envelopes are easier to reason about.
- Interview and handoff explanation is defensible.

### Negative
- Documentation and maintenance are heavier than a single-stack narrative.
- Some shared abstractions are intentionally duplicated between tracks.
