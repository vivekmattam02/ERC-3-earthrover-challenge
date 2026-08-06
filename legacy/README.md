# Legacy Runtime Snapshots

This directory contains superseded indoor-runtime experiments retained for
provenance. They are not supported run entrypoints.

Use `../live_indoor_runtime.py` for every new indoor or no-GPS route-repeat
run. It is the only maintained live runtime and supports the `simple`,
`adaptive`, and `mbra` controller choices.

| File | Historical role | Replacement |
| --- | --- | --- |
| `live_indoor_runtime_mbra.py` | MBRA-first runtime fork | `../live_indoor_runtime.py --controller mbra` |
| `live_indoor_runtime_recovery.py` | Heavy recovery runtime fork | `../live_indoor_runtime.py` |
| `mbra_local_controller.py` | Old adapter targeting `mbra_repo_1` | `../src/mbra_controller.py` |

The reference workspace `mbra_repo_1/` is also historical. It has no model
weights and is not on the active control path.
