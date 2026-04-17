# MBRA vs LogoNav

This is the cleanest concept note in the vault for the controller split.

## One-Line Distinction

- **MBRA** was used as the indoor short-horizon visual controller.
- **LogoNav** was used as the outdoor short-horizon mission-following controller.

Neither one was the whole autonomy stack.

## MBRA

- local visual controller
- goal-conditioned over nearby visual subgoals
- useful once localization and graph progression are already clean
- strongest mental model: "given a nearby visual target, how should I move right now?"

In this project, MBRA was **not**:

- the indoor localizer
- the global planner
- the full indoor system

Useful references:

- [[erc3_full_documentation]]
- [[docs/our_mbra_discoveries]]
- [[90 Archive/MBRA Algorithm Spec]]
- [[90 Archive/MBRA Code File By File]]

## LogoNav

- outdoor local motion policy
- used inside a larger mission runtime
- works with checkpoint semantics, OSM-expanded waypoints, and safety layers
- strongest mental model: "given the active outdoor target geometry, what local command should I propose?"

In this project, LogoNav was **not**:

- the global mission manager
- the route planner
- the full outdoor safety system

Useful references:

- [[erc3_full_documentation]]
- [[live_outdoor_runtime_explained]]
- [[live_outdoor_ultra_marathon_story]]

## Why The Split Matters

The project only becomes understandable when the responsibilities are separated:

- indoor = visual localization + temporal stabilization + graph progression + MBRA local control
- outdoor = mission checkpoints + route semantics + LogoNav local control + safety/runtime gates

## Common Misunderstanding

If I say "we used MBRA indoors and LogoNav outdoors," that is incomplete.

The honest version is:

- MBRA and LogoNav were the local controller choices
- the real system quality came from the runtime around them
