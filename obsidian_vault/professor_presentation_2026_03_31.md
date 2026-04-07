# ERC-3 EarthRover Project Update

> Source: `professor_presentation_2026_03_31.tex`
> Master Note: [[erc3_full_documentation]]

## Indoor

**What we chose**
- MBRA as the main indoor controller
- exact checkpoint-step navigation
- a simpler runtime around it

**Why we chose it**
- indoor is a known corridor problem
- MBRA worked better with simpler recovery
- exact steps were cleaner than vague targets

## Indoor Results

**What worked**
- **8 / 11 checkpoints reached**
- the system became more stable
- the MBRA-first direction became clearer

**What failed / what was difficult**
- forward-only graph caused no-path issues
- stale context could trap the robot
- recovery still needed care

## Outdoor

**What we chose**
- LogoNav as the main outdoor controller
- OSM route expansion
- safety layers around the runtime

**Why we chose it**
- outdoor was not a from-scratch controller problem
- we already had a working base runtime
- the bigger need was safety and stability

## Outdoor Results

**What worked**
- one run reached **all checkpoints**
- the others were around **50% success**
- rerouting and waypoint fixes helped a lot

**What failed / what was difficult**
- **3 rounds failed**
- spinning and bad waypoint transitions hurt us
- curbs and steps are still a weak point

## Marathon

**What we chose**
- keep the same outdoor base runtime
- add stricter safety layers
- focus on stability, not full autonomy

**What we added**
- IMU safety
- route corridor guard
- rerouting from live GPS
- better waypoint handling
- clearer logs

## Marathon Result And Failure Reason

**Reached 1 checkpoint**

**Then the robot spun and toppled**

**What happened**
- after checkpoint progress, the next target handling became unstable
- that led to repeated turning in place

**Why this likely happened**
- the runtime likely got confused after checkpoint transition
- that created bad alignment behavior
- repeated aggressive turning made the robot unstable

**Main lesson**

The marathon problem was not just navigation. It was stability after checkpoint and waypoint transitions.

## Final Summary

**Indoor**
- best current direction: MBRA-first
- result: **8 / 11**

**Outdoor**
- best current direction: LogoNav + route + safety layers
- result: **1 full success**, others around **50%**

**Marathon**
- result: **1 checkpoint, then toppled**
- next focus: stability after transitions
