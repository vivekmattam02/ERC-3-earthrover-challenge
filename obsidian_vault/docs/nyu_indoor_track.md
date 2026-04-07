# NYU Tracks for the EarthRover Challenge 3 - [0.3em] Indoor Navigation \& Campus Outdoor Navigation - [0.5em] \normalsize Suggestions for the Team (Rules Not Yet Final)

> Source: `docs/nyu_indoor_track.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.85\textwidth}{
**Note:** This document is a set of *suggestions* for how the NYU indoor and campus outdoor tracks could be run. The rules are **not yet confirmed** by the ERC organizers. A final rulebook will be shared with the team once we have confirmation from the ERC team.
}}

# How Previous Competitions Worked

The EarthRover Challenge has run twice so far. Both editions were outdoor-only, urban GPS-goal competitions. Understanding how they worked is useful context when we propose our own track design.

## ERC 1 --- IROS 2024, Abu Dhabi

Seven experienced human gamers and three AI teams (Seoul National University, National University of Singapore, UT Austin) competed across 8 cities in 6 countries over 2 days. Each mission was a sequence of GPS checkpoints on city sidewalks. A checkpoint counted as "reached" when the robot came within 15 meters of its GPS coordinates. Missions were scored 1--10 by difficulty. Full completion earned the mission's difficulty score; partial completion earned a proportional fraction.

AI teams were allowed up to 3 brief human interventions per mission, but using any intervention halved the mission's points. Three human gamers achieved the maximum possible score (42/42). The best AI team (SNU) scored 15.2/42. The worst human still more than doubled the best AI.

## ERC 2 --- ICRA 2025, Atlanta

Same format, but across 6 countries on 3 continents. The NUS team (GeNIE) won with 79% of the maximum possible score and zero interventions, a dramatic improvement over ERC~1. The scoring, checkpoint radius, intervention penalty, and round-robin format were all carried over from ERC~1.

## What the ERC 3 Proposal Adds

The ERC~3 proposal (submitted by FrodoBots, NYU, GMU, Google DeepMind, and Princeton) expands the competition from one domain to three:

1. **Indoor Navigation (NYU):** Missions in structured campus buildings, using image goals.
1. **Off-Road Terrain Navigation (GMU):** Missions on natural, uneven terrain, using image goals.
1. **In-the-Wild Urban Navigation (Multiple Cities):** The existing outdoor format with GPS goals.

This document suggests how NYU could run its two contributions: the **indoor track** and a **campus outdoor track**. None of this is final until the ERC team confirms.

# Robot Platform and Connectivity

The robot is the same EarthRover unit used in all previous editions. This part is fixed by the platform, not something we define:

- Weight $<$2 kg, max speed $\sim$3 km/h, 4-wheel skid-steer
- Front and rear RGB cameras
- GPS, IMU, wheel encoders
- **4G cellular SIM card** for all communication (indoor and outdoor)

**Suggestion:** Use 4G for indoor missions at NYU as well, not campus WiFi. That keeps conditions consistent with outdoor urban missions and with how the robot was designed. Teams would then face the same $\sim$500~ms round-trip latency and 3--5 FPS effective frame rate indoors as outdoors. From our testing, 4G signal inside 6MTC and 5MTC is adequate.

Teams would interact with the robot through the standard FrodoBots Remote Access SDK (HTTP endpoints for camera frames, sensor data, and velocity commands), exactly as in ERC~1 and~2.

# Suggested NYU Indoor Track

## Location

We suggest using the ground floors of two buildings on the NYU Tandon campus in Brooklyn:

- **6 MetroTech Center (6MTC), Ground Floor** --- wide main corridors, glass-walled labs and offices, T-junctions, elevator lobbies, and a large open atrium area.
- **5 MetroTech Center (5MTC), Ground Floor** --- narrower corridors, more turns, side hallways, and connections to adjacent buildings.

Between them, these two floors would support a good range of indoor difficulty: from simple straight runs down a wide hallway up to multi-turn routes through narrow passages with foot traffic. Exact routes and allowed areas would be fixed once we have ERC confirmation and campus approvals.

## Mission Design (Suggestion)

We suggest that each indoor mission be a sequence of **image-defined checkpoints** that the robot must visit in order, then return to the finish point:

- **Start:** Robot placed at a marked start pose by the track marshal.
- **Checkpoints:** 3--6 image goals per mission, provided to the team as an ordered list before the run.
- **Finish:** Usually the same location as the start.

Each image goal would be a single photograph taken from the robot's own camera at the intended checkpoint pose. The team's system would need to navigate to each goal; arrival could be judged either by an automated visual criterion (see Section~\ref{sec:goal-criterion}) or by a marshal, depending on what the ERC team decides.

## Suggested Indoor Difficulty Levels

We suggest assigning indoor missions difficulty levels 1--10, in the same spirit as the outdoor rubric but adapted for buildings:

\begin{table}[h]
\renewcommand{\arraystretch}{1.15}
```text
\toprule
\textbf{Level} & \textbf{Suggested description} \\
\midrule
1--2 & Short route ($<$50 m), wide corridor, straight or one turn, no people, good lighting. \\
3--4 & Medium route (50--100 m), 2--3 turns, T-junctions, some foot traffic, mix of corridor widths. \\
5--6 & Longer route (100--150 m), multiple junctions, narrow passages, moderate foot traffic, some visually repetitive sections. \\
7--8 & Route spans both 6MTC and 5MTC ground floors, building connections, heavier foot traffic, doors, potentially confusing junctions. \\
9--10 & Long multi-building route, many turns, poor lighting or reflective surfaces, high foot traffic, visually ambiguous corridors. \\
\bottomrule
```
\caption{Suggested indoor difficulty levels. Final rubric to be confirmed with ERC.}
\end{table}

# Suggested NYU Campus Outdoor Track

In addition to indoor missions, we suggest offering a **campus outdoor track** that can serve as one of the "seen" environments for the broader ERC outdoor domain.

## Candidate Routes (Suggestions)

The MetroTech campus area has sidewalks, plazas, and pedestrian paths connecting several buildings. Example outdoor missions we could offer:

- **370 Jay Street to 6MTC:** Sidewalk route of roughly 200--300 meters through the MetroTech plaza area. Pedestrian crossings, open plaza, curb transitions. Suggested difficulty: 3--5 depending on foot traffic.
- **6MTC to 5MTC (exterior):** Shorter route ($\sim$100 m) around the outside of the buildings. Suggested difficulty: 2--3.
- **370 Jay Street loop:** Start at 370 Jay, through MetroTech plaza to 6MTC, then to 5MTC, and return to 370 Jay. Suggested difficulty: 5--7 depending on time of day and pedestrian density.

These would use **GPS checkpoints** with the standard 15-meter radius, identical to the global urban missions. The benefit is that we could offer this as a "seen" testing environment before the competition, with global city missions remaining "unseen" during the actual event. Whether and how this is used will depend on ERC confirmation.

# Suggested Image-Goal Success Criterion

Indoor checkpoints cannot use GPS. We need a visual way to decide when the robot has "reached" a goal image. Below is a **suggested** method that we think is objective and robust enough to use in a rulebook. The final criterion will be set only after ERC confirmation.

## What We Need From a Criterion

1. **Objective:** Different judges or systems should reach the same conclusion.
1. **Robust:** Small differences in viewpoint, lighting, or people in the frame should not cause false negatives.
1. **Discriminative:** Visually similar but physically different locations (e.g., two identical-looking hallway sections) should not cause false positives.

## Suggested Method: Two-Stage Visual Verification

We suggest a two-stage approach used in visual place recognition work: a fast global descriptor check, then a local feature verification.

### Stage 1: Global Descriptor Matching

A pretrained VPR model would encode the goal image and each incoming camera frame into a compact global descriptor. Compute the **cosine similarity** between the current frame and the goal. If similarity exceeds a threshold $\tau_1$, proceed to Stage~2.

Possible choices for the global descriptor (to be finalized with ERC and after calibration):

- **DINOv2 ViT-B/14** --- strong general visual features, widely used in VPR benchmarks.
- **NetVLAD / SuperVLAD** --- purpose-built for place recognition, compact.
- **CLIP ViT-L/14** --- good semantic generalization, less spatially precise.

Stage~1 is fast and filters out frames that are clearly not at the goal.

### Stage 2: Local Feature Verification

Once Stage~1 triggers, run a local feature matcher to confirm geometric consistency:

1. Extract keypoints and descriptors from both images (e.g.\ **SuperPoint**).
1. Match features (e.g.\ **LightGlue** or SuperGlue).
1. Estimate the fundamental matrix via RANSAC; count **geometrically verified inlier matches** $N_{\text{inliers}}$.

We could define the checkpoint as **reached** when $N_{\text{inliers}} \geq \tau_2$ (e.g., $\tau_2 = 50$ inliers), and require this to hold for **3 consecutive frames** (about 1 second at 3 FPS) to avoid single-frame flukes.

The two-stage design gives a fast filter (Stage~1) and a strict geometric check (Stage~2). Two images will only have many verified inliers if they show the same physical scene from a similar viewpoint.

## Suggested Edge-Case Handling

- **Consecutive-frame requirement:** As above --- 3 consecutive frames to reduce spurious positives.
- **Visually repetitive corridors:** Geometric verification in Stage~2 helps. We could also choose goal images that include distinctive landmarks (door numbers, posters, furniture) where possible.
- **Occlusion by pedestrians:** Match may drop temporarily; the 3-frame requirement means the robot can wait for the person to pass.
- **Marshal fallback:** If the automated system and visual evidence clearly disagree, the on-site marshal could have final say. We expect this to be rare if thresholds are calibrated well.

## Suggested Pre-Competition Calibration

Before the competition we could run a calibration on the actual track: collect reference images at each goal, drive past at various offsets, record similarity and inlier count vs.\ distance, and set $\tau_1$, $\tau_2$ so the system triggers within $\sim$2 m of the goal and does not trigger from $>$5 m or from a different corridor. The chosen model and thresholds would then be published to all teams. All of this is a suggestion; the final process will follow ERC confirmation.

# Suggested Scoring and Autonomy Rules

We suggest keeping the same scoring and autonomy rules as ERC~1 and~2, applied to indoor, outdoor campus, and global urban missions alike. This is something the ERC team will confirm.

- **Mission points:** Each mission has difficulty 1--10. Full completion earns that many points; partial completion earns a proportional fraction.
- **Interventions:** Up to **3 brief human interventions per mission**; **any intervention halves the mission's earned points**.
- **Time limit:** e.g.\ 15--20 minutes per indoor mission (by route length), 30--60 minutes per outdoor campus mission (aligned with global urban limits).
- **Tiebreaker:** Faster total completion time.

We also suggest a per-track sub-leaderboard (indoor, outdoor campus, global urban) plus an overall combined ranking. Final structure is for the ERC team to decide.

# Suggested Competition Flow at NYU

## Before the Competition (Suggestions)

- Teams receive floor-plan sketches of 6MTC and 5MTC ground floors, and a map of the MetroTech campus for outdoor routes.
- We provide a small sample dataset: a few teleoperated runs on the indoor track with camera images, sensor logs, and example image goals (indoor analogue of FrodoBots-7K).
- Teams can schedule limited remote testing (e.g.\ at least 5 hours per team) on the indoor and outdoor campus tracks, under supervision, with a bot walker on site.
- The image-goal matching model, thresholds, and calibration data (if we use the suggested method) would be published to all teams in advance.

## During the Competition (Suggestions)

Based on the original ERC~3 proposal, the indoor round was slotted for Day~1 afternoon. We suggest a possible NYU-day layout:

\begin{table}[h]
```text
\toprule
\textbf{Time} & \textbf{Suggested activity} \\
\midrule
09:00 -- 09:30 & Connectivity check for NYU tracks \\
09:30 -- 12:00 & Indoor missions (6MTC and 5MTC), round-robin \\
12:00 -- 13:30 & Lunch \\
13:30 -- 15:30 & Outdoor campus missions (370 Jay $\to$ 6MTC $\to$ 5MTC routes) \\
15:30 -- 16:00 & Buffer / re-runs if needed \\
\bottomrule
```
\caption{Suggested NYU track schedule for one competition day. Global urban and GMU off-road would run on the other day or in parallel. Final schedule subject to ERC confirmation.}
\end{table}

All runs would be live-streamed to the conference venue, with the track marshal monitoring the robot's feed for safety.

# Suggested Safety and Operations

- A **track marshal** with physical emergency stop at all times.
- **Indoor speed cap:** We suggest reducing max linear velocity indoors (e.g.\ 50--70% of outdoor max) for closer walls and pedestrians.
- **Off-limits areas** (stairwells, elevators, restricted labs) clearly marked; marshal stops the run if the robot enters them.
- **Pedestrian safety:** Schedule indoor rounds during lower-traffic hours where possible; post signs in corridors.
- **Safety-stopped runs:** Organizers could either count the run as failed or restart from the last confirmed checkpoint, at the marshal's discretion.

# Summary for Teams (If These Suggestions Are Adopted)

If the ERC team confirms rules along these lines, participating teams could expect:

1. Robot uses **4G** even indoors; same latency and frame rate as outdoor missions.
1. Indoor checkpoints are **image goals**; systems must do visual place recognition or image-goal navigation.
1. Outdoor campus checkpoints use **GPS** with 15~m radius, same as global urban.
1. Image-goal matching (model, thresholds, calibration) would be **published in advance** for testing.
1. Intervention penalty same as ERC~1/2: any intervention halves mission score.
1. Indoor environments: corridors, T-junctions, glass walls, doors, signs, varying lighting, pedestrians; no stairs or elevators.
1. Floor-plan sketches and sample data provided; no detailed 3D map of the buildings.

# Final Rulebook After ERC Confirmation

\addcontentsline{toc}{section}{Final Rulebook After ERC Confirmation}

Everything in this document is a **proposal for the team and for discussion with the ERC organizers**. The locations (6MTC, 5MTC, 370 Jay, MetroTech routes), mission format, image-goal criterion, scoring, schedule, and safety measures are all **suggestions** until the EarthRover Challenge organizing team confirms the official rules.

Once we have confirmation from the ERC team, we will prepare and share a **final rulebook** that reflects the agreed rules. That document will be the single source of truth for the NYU tracks. Until then, treat this document as a working draft to align the team and to iterate with organizers.
