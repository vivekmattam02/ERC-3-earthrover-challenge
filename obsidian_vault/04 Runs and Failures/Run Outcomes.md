# Run Outcomes

This note is the shortest clean summary of how the project actually performed.

## Indoor

- target: 11 checkpoints
- achieved: 8 checkpoints
- main limitations: ambiguity windows, transition edge cases, recovery/state issues

## Outdoor

- target: full checkpoint mission
- achieved: one full success, others partial
- main limitations: waypoint handoff instability, reroute semantics, transition alignment behavior

## Marathon

- target: long supervised outdoor run with stricter safety
- achieved: first checkpoint reached
- failure: transition-spin instability and platform tipping risk

## Current Off-Road Branch

- target: no-GPS repeated-route autonomy on rough terrain
- achieved: manual bag collection, prepared routes, front-camera live localization, and sent motion commands
- field result: no reliable autonomous repeat yet; runs repeatedly stalled near startup steps and cycled through relocalization behavior
- current limitation: reference/start mismatch, localization stability, and terrain recovery matter more than adding modules
- strongest coverage/reference candidate: `smoke_run01_c`
- current best post-processed comparison route: `run_01_1521_pp_v3`

See [[01 Source of Truth/No-GPS Field Trial - Findings]] for the evidence and
the required proof before calling this branch field-ready.

## Best Supporting Notes

- [[competition_results]]
- [[live_indoor_runtime_story]]
- [[live_outdoor_ultra_marathon_story]]
- [[04 Runs and Failures/Marathon Failure - What Actually Happened]]
