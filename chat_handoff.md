# ERC Indoor Chat Handoff

This is a detailed handoff note reconstructed from the full working session.
It is not a raw platform export, but it is intended to preserve the important
technical details, decisions, corrections, warnings, and outcomes from the
entire discussion.

## 1. Repo cleanup and baseline direction

- `baseline.py` was rewritten away from hardcoded one-off notebook style code
  into a shared baseline with CLI entry points.
- The baseline direction was set to:
  - known-corridor visual localization
  - graph planning
  - separate local controller
  - separate safety / recovery
- The project explicitly stopped treating MBRA as the whole stack.

## 2. Direct SLAM discussion

- Direct SLAM was discussed and rejected as the main backbone for this exact
  competition setting.
- Reason:
  - low FPS
  - latency
  - compressed stream
  - reflective / repetitive corridor structure
  - poor match to checkpoint / graph mission logic
- Final position:
  - direct SLAM is not impossible
  - but it is not the clean main backbone for this known-corridor task

## 3. Git / repo-sharing decisions

- The project should be shared as its own repo:
  - `/home/vivek/Desktop/rover/ERC-3-earthrover-challenge`
- Not from the parent `/home/vivek/Desktop/rover` workspace.
- `.gitignore` was updated to ignore:
  - local `.env`
  - generated artifacts
  - weights
  - `data/`
  - `docs/`
  - `*.tex`
  - `guide.md`
  - `CONTEXT.md`

## 4. DBR / Depth-Anything cleanup

- The misleading `DBR/` naming inside this repo was removed.
- `Depth-Anything-V2` was moved under a clearer third-party path.
- The conclusion was:
  - DBR was not an active runtime component here
  - it was mostly just a container for vendored depth dependency code

## 5. MBRA role correction

- A key conceptual correction was made:
  - MBRA should not automatically be treated as the deployed online controller
  - MBRA is best understood as a short-horizon expert / control candidate
  - LogoNav is the long-horizon deployed-policy side in the original project
- Final stack interpretation:
  - localization/planning backbone stays graph-based
  - MBRA, if used, belongs only in the local-controller slot

## 6. Sensor stance correction

- The project originally sounded too camera-only.
- A correction was made:
  - vision stays primary
  - IMU/orientation should support temporal filtering and control
  - RPMs can be used as weak motion hints if reliable
- Final stance:
  - not full EKF backbone
  - yes to lightweight motion-prior support

## 7. H5 recording inspection

- `data/corrider.h5` was inspected.
- It was found to contain:
  - front frames
  - controls
  - telemetry
  - accelerometer
  - gyroscope
  - magnetometer
  - RPMs
- It became the main useful dataset seed for the corridor baseline.

## 8. H5 extraction

- A new extractor was created:
  - `tools/extract_h5_dataset.py`
- It exports:
  - `front_images/`
  - `metadata/data_info.json`
  - metadata CSVs
  - summary JSON
- Important correction:
  - stream alignment was done by relative time, not naive raw timestamp matching

## 9. Corridor DB and graph build

- The corridor database was built from the extracted recording.
- Built outputs include:
  - `descriptors.npz`
  - `config.json`
  - `place_graph.json`
  - `navigation_graph.json`
- `baseline.py` was fixed to handle JSON serialization of NumPy scalar types.

## 10. Localization evaluation

- Single-frame query sanity checks looked correct.
- Held-out localization against a step-5 subsampled DB also behaved sensibly.
- A temporal localizer was then introduced and evaluated.
- Best reported held-out numbers:
  - exact: `95.7%`
  - near: `97.9%`
  - moving-frame exact: `99.1%`
  - moving-frame near: `100%`
- Important interpretation:
  - the main scary failure region was mostly stationary duplicate frames
  - this was not a normal moving-localization failure

## 11. Runtime localization/planning modules

- Runtime modules were created:
  - `src/temporal_localization.py`
  - `src/corridor_localizer.py`
  - `src/graph_planner.py`
  - `src/navigation_runtime.py`
- Their intended split:
  - localizer: current node estimate + confidence
  - planner: target path + next subgoal
  - runtime: handoff bundle for control

## 12. Simple local controller baseline

- A first baseline controller was created:
  - `src/local_controller.py`
- It began as a simple heading-aware heuristic.
- Purpose:
  - make the full stack executable end-to-end
  - not to claim a final competition controller

## 13. Live runtime loop

- A live runner was created:
  - `live_indoor_runtime.py`
- It connects:
  - EarthRover SDK
  - localization
  - planning
  - controller
- It defaults to dry-run and only sends commands if explicitly told to.

## 14. Reality check from live tests

- Live localization was tested on the real corridor feed.
- Important observation:
  - localization looked genuinely good when the robot was placed in known
    corridor locations
- This was a major confidence boost for the chosen localization backbone.

## 15. Controller problems observed live

- The simple controller showed weak behavior:
  - spinning in place
  - hesitation
  - oscillation
  - no-progress situations
- Main conclusion:
  - localization is not the main problem now
  - control is the main weak point

## 16. Controller improvements

- `src/local_controller.py` was improved multiple times:
  - align-heading mode
  - hysteresis
  - no-progress detection
  - less aggressive turn behavior
  - stale motion-state stop
  - better debug output
- Even after improvements, the controller is still treated as:
  - better baseline
  - not final validated controller

## 17. Motion-prior / IMU integration

- A lightweight motion state filter was added:
  - `src/sensor_state.py`
- It provides:
  - filtered heading
  - gyro-z turn-rate estimate
  - RPM mean hint
- `live_indoor_runtime.py` was updated to pass that into the controller.
- Final interpretation:
  - this is useful and sensible
  - but it is intentionally not a full EKF-centric system

## 18. Recovery and safety

- The runtime now has some basic protections:
  - low-confidence stop
  - no-path stop
  - stale-motion-state stop
- But a full recover / relocalize / resume state machine is still missing.
- Depth safety exists in repo form, but is not fully integrated into the live
  runtime path yet.

## 19. MBRA integration status

- `src/mbra_controller.py` was added as an optional controller wrapper.
- `live_indoor_runtime.py` supports:
  - `--controller simple`
  - `--controller mbra`
- Important caution:
  - this is only an architectural integration
  - MBRA is not yet validated as a working controller in this repo
- Real blockers confirmed:
  - missing Python dependencies in the active env
  - missing `mbra.pth` weights
- Important conceptual final position:
  - MBRA is only a short-horizon controller candidate
  - not the localization backbone
  - not the graph planner
  - not the whole system

## 20. Literature fact-check

- The approach was checked against external sources.
- Final literature-based conclusion:
  - the high-level architecture is broadly correct
  - repeated-route visual localization + topological planning is well supported
  - the weak part is local execution/control, not the graph-localization idea
- Main sources referenced:
  - PlaceNav
  - MBRA project page / paper / repo
  - teach-and-repeat papers
  - visual-inertial teach-and-repeat
  - RoboHop
  - GNM / ViNT as supporting background

## 21. Docs created / updated

- `docs/march19.tex`
  - broad project explanation
- `docs/discoveries.tex`
  - literature fact-check note
  - includes reference inventory
- `CONTEXT.md`
  - local working status note
- `guide.md`
  - local step-by-step run guide

## 22. Notes about what is solid vs partial

### Solid

- corridor localization
- temporal stabilization
- graph planning
- live runtime wiring
- debug visibility

### Partial / unfinished

- local controller quality
- safety integration
- recovery state machine
- MBRA runtime setup and validation

## 23. Current best mental model

The system should be understood as:

1. camera image tells us where we are
2. temporal filtering keeps that estimate stable
3. graph planner tells us the next nearby place to aim for
4. controller tries to move the robot toward that nearby place
5. safety and recovery should sit around the controller

## 24. What should happen next

If working off-corridor:

- improve diagnostics
- improve controller logic
- improve recovery/safety logic
- prepare MBRA env properly if desired

If back in the corridor:

- re-test short node-to-node motion
- verify improved controller behavior
- only then consider MBRA as a replacement candidate

## 25. File transfer / second laptop notes

- `scp` was used successfully to transfer `data/`
- docs can be transferred with:
  - `scp -r .../docs ...`
- `.env` failed once because the source path used was wrong, not because `scp`
  was broken
- SSH between laptops did work once the correct reachable IP was used

## 26. Most important overall conclusion

The project is not failing because the architecture is wrong.

The strongest evidence so far is that localization works well on the real
corridor.  That means the biggest remaining engineering problem is:

\begin{center}
\textbf{make short-horizon execution reliable}
\end{center}

That is the main remaining gap between the current repo and a truly reliable
indoor runtime.
