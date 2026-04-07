# Professor Presentation Cheat Sheet

> Source: `nyutemplate/professor_presentation_nyu_cheatsheet.tex`
> Master Note: [[erc3_full_documentation]]

\thispagestyle{empty}

# Opening

Short update on the EarthRover project. Keep it simple: indoor, outdoor, marathon. For each one: what we chose, why we chose it, what worked, what failed.

# Indoor

**What we chose**
- MBRA as the main indoor controller
- exact checkpoint-step navigation
- a cleaner runtime around it

**Why**
- indoor is a known corridor problem
- exact steps are cleaner than vague targets
- MBRA worked better with simpler recovery

**Result**
- reached **8 / 11** checkpoints
- system became more stable and more understandable

**Main failures**
- forward-only graph caused no-path issues
- stale context could trap the robot
- recovery still needed care

# Outdoor

**What we chose**
- LogoNav as the main outdoor controller
- OSM route expansion
- safety layers around the runtime

**Why**
- outdoor already had a working base runtime
- the bigger need was safety and stability, not replacing the controller

**Result**
- one run reached **all checkpoints**
- the rest were around **50%** success overall
- **3 rounds failed**

**Main failures**
- spinning
- bad waypoint transitions
- curbs and steps still weak

# Marathon

**What we chose**
- same outdoor base runtime
- stricter safety layers
- focus on stability, not full autonomy

**What we added**
- IMU safety
- route corridor guard
- rerouting from live GPS
- better waypoint handling
- clearer logs

**Result**
- reached **1 checkpoint**
- then the robot spun and toppled

**Reason**
- likely unstable target handling after checkpoint transition
- repeated alignment / turning in place
- repeated aggressive turning made the robot unstable

# Final Summary

- Indoor: best direction is MBRA-first, result **8 / 11**
- Outdoor: best direction is LogoNav + route + safety layers, result **1 full success**, others around **50%**
- Marathon: main issue is stability after checkpoint and waypoint transitions

# Numbers To Remember

- indoor: **8 / 11**
- outdoor best run: **all checkpoints**
- outdoor overall: **around 50%**
- outdoor failed rounds: **3**
- marathon: **1 checkpoint, then toppled**

# One Good Closing Line

The project is much better understood now, and the remaining weaknesses are much clearer.
