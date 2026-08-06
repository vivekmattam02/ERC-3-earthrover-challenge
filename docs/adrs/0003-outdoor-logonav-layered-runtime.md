# ADR 0003: Keep LogoNav and Harden the Outdoor Runtime Around It

- Status: Accepted
- Date: 2026-04-04

## Context

The repository already had a runnable outdoor path. Main failures were not just controller outputs; they were target transition semantics, route drift, and safety/recovery interaction under field conditions.

## Decision

Keep LogoNav as the primary outdoor local motion policy and invest in runtime hardening:
- mission checkpoint handling,
- OSM route expansion,
- route-corridor guard,
- traversability + semantic gating,
- IMU/vision/GPS safety checks,
- reroute and target handoff logic.

## Consequences

### Positive
- Preserved existing runnable path.
- Focused work on the highest failure leverage points.
- Delivered one full outdoor completion and clearer failure diagnosis.

### Negative
- Runtime complexity increased.
- Transition behavior remains the dominant residual risk.
