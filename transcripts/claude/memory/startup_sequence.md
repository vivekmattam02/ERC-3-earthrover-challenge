---
name: competition-startup-sequence
description: exact startup sequences for both ERC outdoor mission and indoor corridor navigation
metadata: 
  node_type: memory
  type: project
  originSessionId: bd8b6e76-aa26-45d8-a6af-82e9da5b97b5
  modified: 2026-08-06T20:23:51.101Z
---

# NO-GPS TEACH-AND-REPEAT (current active path, see [[project_state]])

Terminal 1 — SDK server (same as below). Terminal 2 — `curl -X POST http://127.0.0.1:8000/start-mission`.

Terminal 3 — run the prepared route:
```bash
python scripts/run_prepared_route.py --route-dir data/manual_routes/smoke_run01_c \
  --rough-terrain --no-route-heading --startup-step-hint 0 --send-control
```
Add `--print-only` to see the expanded `live_indoor_runtime.py` command, drop
`--send-control` for a dry run. Emergency stop:
```bash
curl -s -X POST http://127.0.0.1:8000/control-legacy -H 'Content-Type: application/json' \
  --data '{"command":{"linear":0.0,"angular":0.0,"lamp":0}}'
```

Record a teach bag:
```bash
python scripts/record_sdk_session.py --sdk-url http://127.0.0.1:8000 \
  --output-h5 recordings/<name>.h5 --poll-hz 2.0 --session-name clean_teach_route
```
Never build the output path with `$(date ...)` in an interactive paste — it has
repeatedly wrapped and produced malformed filenames.

# OUTDOOR (historical — bot returns placeholder GPS 1000,1000)

Terminal 1 — SDK server:
```bash
sudo fuser -k 8000/tcp 2>/dev/null; cd ~/Desktop/rover/ERC-3-earthrover-challenge/earth-rovers-sdk && conda activate erv && hypercorn main:app --reload
```

Terminal 2 — start mission:
```bash
curl -X POST http://127.0.0.1:8000/start-mission
```

Terminal 3 — run controller (base, proven):
```bash
python live_outdoor_runtime.py --mission --send-control
```

Terminal 3 — with soft traversability bias:
```bash
python live_outdoor_runtime.py --mission --send-control --traversability
```

Terminal 3 — with semantics (new, untested on robot):
```bash
python live_outdoor_runtime.py --mission --send-control --traversability --semantics
```

# INDOOR

Terminal 1 — SDK server (same as outdoor):
```bash
sudo fuser -k 8000/tcp 2>/dev/null; cd ~/Desktop/rover/ERC-3-earthrover-challenge/earth-rovers-sdk && conda activate erv && hypercorn main:app --reload
```

Terminal 2 — start mission:
```bash
curl -X POST http://127.0.0.1:8000/start-mission
```

Terminal 3 — single target step:
```bash
python live_indoor_runtime.py --target-step <STEP> --send-control --controller simple
```

Terminal 3 — with checkpoint steps (from prelocalization):
```bash
python live_indoor_runtime.py --checkpoint-steps 120 180 240 ... --auto-advance-checkpoints --send-control
```

Terminal 3 — with checkpoint images (if images match DB names):
```bash
python live_indoor_runtime.py --checkpoint-images img1.png img2.png ... --auto-advance-checkpoints --send-control
```

**Why:** hypercorn (not npm start) is the correct ASGI server for the SDK. Mission must be started via API before the runtime connects. Indoor uses `--controller simple` (default). `--send-control` is REQUIRED or robot won't move (DRY RUN mode).
