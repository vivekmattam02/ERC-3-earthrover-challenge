# ERC Run Guide

Start with the current no-GPS field evidence when working on rough-terrain
repeat; use [`CLAUDE.md`](CLAUDE.md) as the historical indoor reference. This
guide is the practical companion: what to run, in what order, and what to verify
before you trust the rover.

## Where to run commands

Unless a section says otherwise, run commands from the repository root:

```bash
cd ERC-3-earthrover-challenge
```

## Basic rule before autonomy

Do not jump straight into a live autonomous run.

Always verify:

1. the SDK is reachable,
2. manual control works,
3. the camera is live,
4. the runtime starts cleanly in dry-run mode before you send commands.

## No-GPS Rough-Terrain Route Repeat

This is an experimental front-camera teach-and-repeat branch, not a validated
competition-run workflow. The current evidence shows that commands and motion
work but sustained field progression is not yet reliable.

Before a controlled test:

1. confirm the physical start visually matches the teach-route start;
2. use the prepared-route launcher in dry-run mode;
3. verify localization advances through a short segment;
4. only then enable control under supervision.

The route package `data/manual_routes/smoke_run01_c` is the best current
coverage candidate, not a field-proven deployment route. The runtime uses the
front camera only; existing manual bags do not contain teleop controls, so they
cannot support exact command replay.

Read [`Current No-GPS - Read This First`](obsidian_vault/00%20Home/Current%20No-GPS%20-%20Read%20This%20First.md)
first, then the full [`No-GPS Field Trial - Findings`](obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md)
before a live test.

## Indoor Quick Start

### 1. Start the SDK bridge

In terminal 1:

```bash
cd earth-rovers-sdk
hypercorn main:app --reload
```

Keep that terminal open.

### 2. Verify manual control

In terminal 2:

```bash
python earth-rovers-sdk/examples/simple_control.py
```

Useful commands inside the script:

```text
w
s
a
d
wa
wd
x
q
```

If manual control does not work, stop there and fix the SDK / bot connection first.

### 3. Build the indoor dataset only if needed

If the extracted dataset does not already exist:

```bash
python tools/extract_h5_dataset.py \
  --input-h5 data/corrider.h5 \
  --output-dir data/corrider_extracted
```

### 4. Build the corridor database only if needed

```bash
python baseline.py build-db \
  --image-dir data/corrider_extracted/front_images \
  --output-dir data/corrider_db \
  --data-info-json data/corrider_extracted/metadata/data_info.json
```

### 5. Dry-run the indoor runtime first

```bash
python live_indoor_runtime.py --target-step 400 --max-steps 30
```

Check the log for:

- current step,
- target step,
- subgoal step,
- confidence,
- command output.

If localization is obviously wrong, do not send control.

### 6. Indoor competition-style run

```bash
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

## Outdoor Quick Start

### 1. Standard outdoor mission run

```bash
python live_outdoor_runtime.py --mission --send-control --controller logonav --osm-route
```

### 2. Standard outdoor run with traversability enabled

```bash
python live_outdoor_runtime.py --mission --send-control --traversability
```

The default controller is `logonav`, so this shorter command path still uses the
learned outdoor controller.

### 3. Marathon mode

```bash
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

### 4. Night-safe mode

```bash
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --night-safe
```

### 5. Preflight before a long outdoor run

```bash
python scripts/preflight_marathon.py
```

## What a healthy run looks like

### Indoor

- localization is stable over several ticks,
- target and subgoal make sense,
- checkpoint progression advances without repeated `no_path` stalls,
- MBRA is moving forward rather than oscillating in place.

### Outdoor

- mission checkpoints load correctly,
- the active waypoint is reasonable,
- route corridor distance is not growing uncontrollably,
- repeated align-turn loops do not persist,
- traversability and safety messages are rare and interpretable.

## Common mistakes

- Running autonomy before testing manual control.
- Trusting a dry-run log without checking the camera feed.
- Mixing indoor and outdoor assumptions.
- Treating a routed waypoint as if it were the same thing as a real mission
  checkpoint.

## Related Documents

- [`README.md`](README.md) — repository overview
- [`No-GPS Field Trial - Findings`](obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md) — current no-GPS field truth
- [`CLAUDE.md`](CLAUDE.md) — historical indoor reference
- [`docs/INDEX.md`](docs/INDEX.md) — documentation map
- [`live_indoor_runtime_story.tex`](live_indoor_runtime_story.tex) — indoor story
- [`live_outdoor_runtime_explained.tex`](live_outdoor_runtime_explained.tex) — outdoor runtime walkthrough
- [`live_outdoor_ultra_marathon_story.tex`](live_outdoor_ultra_marathon_story.tex) — outdoor marathon story
