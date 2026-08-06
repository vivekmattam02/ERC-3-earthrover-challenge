# ADR 0005: Treat OSM Intermediate Waypoints Differently from Mission Checkpoints

- Status: Accepted
- Date: 2026-04-04

## Context

Outdoor instability often occurred when intermediate routed waypoints were treated like mission checkpoints:
- premature handoff,
- behind-the-rover targets after reroute,
- excessive align behavior.

## Decision

Define distinct semantics for target classes:
- Mission checkpoints: task-level targets with checkpoint reporting semantics.
- OSM intermediates: transient control aids with tighter dynamic radius and prune rules.

Apply transition controls:
- dynamic intermediate radius,
- behind-waypoint pruning,
- align gating by distance and target type.

## Consequences

### Positive
- Better target-handoff behavior.
- Fewer pathological loops after reroute.
- More interpretable outdoor logs.

### Negative
- More conditional logic in runtime.
- More parameters to calibrate for route geometry and transition behavior.
