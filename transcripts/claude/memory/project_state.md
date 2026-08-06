---
name: erc-project-state-no-gps-teach-and-repeat-is-the-active-branch
description: Current state of the rover project — no-GPS rough-terrain teach-and-repeat is active but unproven; indoor MBRA and outdoor GPS are both retired
metadata: 
  node_type: memory
  type: project
  originSessionId: bd8b6e76-aa26-45d8-a6af-82e9da5b97b5
  modified: 2026-08-06T20:40:14.381Z
---

As of 2026-04-27 (last field session; re-verified 2026-08-06) the active problem is
**no-GPS, rough-terrain, differential-drive visual teach-and-repeat**, front camera
only. Both earlier systems are historical:

- **Outdoor GPS/LogoNav** — dead end for the current bot. On 2026-04-22 the SDK
  returned `latitude=1000, longitude=1000` (placeholder), so `live_outdoor_runtime.py`
  refuses to run. Not a code bug — the session simply has no GPS.
- **Indoor MBRA corridor navigation** — superseded. `live_indoor_runtime_mbra.py`,
  `live_indoor_runtime_recovery.py`, `src/mbra_local_controller.py` moved to `legacy/`.

## Active path

`scripts/run_prepared_route.py` → `live_indoor_runtime.py` →
`src/adaptive_pursuit_controller.py`. Rough-terrain runs default to the `adaptive`
controller and to `--no-route-heading` (compass/route heading made heading error
worse in the field).

Reference route: `data/manual_routes/smoke_run01_c` (126 frames, target_step=125),
built from `recordings/manual_flag_collection/2026-04-22/run_01_1521.h5`.

## Status: implemented, NOT field-proven

Pipeline works end to end (record → postprocess → route package → localize → send
commands) and the rover moves, but no reliable autonomous route repeat was ever
demonstrated. Field failure signature: `cur` stuck at step 0/1/8, conf ~0.49–0.59,
repeated `relocalize_scan` / `relocalize_probe_forward` loops.

## Repo layout (settled 2026-08-06)

All no-GPS work is committed and pushed as `c166e98`. `obsidian_vault/` is now a
**plain folder** inside the main repo — its nested `.git` was deleted, so there is
one repo and one commit per change. Do not re-init a repo inside it.

`data/` and `recordings/` stay gitignored, so route packages and teach bags exist
only on this machine + the SSD backup zip. That is the remaining single-copy risk.

Two local-only backups live outside the repo, at `~/Desktop/rover/`:
`obsidian_vault_git_backup.git` (the vault's old history) and `active-perception/`
(a *different* project that had been force-pushed over `vivekmattam02/obsidian`
on 2026-04-22 — that GitHub repo is NOT the rover vault, leave it alone).

## Known open items (2026-08-06)

- `recordings/run_05_clean_teach.h5` (160 frames) was never turned into a route package.
- The recorder writes a `controls` dataset but it is always empty — teleop goes
  through the browser / `keyboard_control.py`, not the recorder, so taught commands
  can never be replayed.

See [[codex_collaboration]] and [[startup_sequence]]. Authoritative in-repo docs:
`obsidian_vault/00 Home/Current No-GPS - Read This First.md` and
`obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md`.
