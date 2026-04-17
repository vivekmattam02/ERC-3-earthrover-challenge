ERC-3 Mathematics and Intuition Guide
=====================================

What this file is
-----------------

This is the math-and-intuition companion to the ERC-3 project.

It is not written like a textbook and it is not written for someone who already
thinks in equations all day.

The goal is:

1. explain the math that appears in this project,
2. explain why that math exists,
3. connect each idea to the code and runtime behavior,
4. help you build intuition first and symbols second,
5. make you strong enough to explain the system clearly under pressure.

If you feel “I am not a math person,” this file is for you.


How to read this
----------------

For each topic, read in this order:

1. intuition
2. what problem the math is solving
3. the actual formula
4. what the formula means in code
5. what mistake people make when they hear the term


PART 1: The Big Picture
=======================

The project uses math in five main places:

1. indoor localization
2. temporal stabilization of localization
3. graph progression and checkpoint logic
4. outdoor control geometry
5. safety scoring from depth, semantics, and IMU

The important thing to understand is that the math is not there to look smart.
Each piece of math exists because the robot has to answer a very specific question.

Those questions are:

1. Which place in the corridor does this image look like?
2. If the last frame said “step 400” and this frame says “step 1200,” should I trust that?
3. What checkpoint comes next, and what nearby target should I use?
4. If the goal is over there, how should I steer?
5. Does the image or IMU suggest the current command is unsafe?


PART 2: Indoor Localization Math
================================

2.1 What is a descriptor?
-------------------------

Intuition:

A descriptor is a compact numeric fingerprint of an image.

Instead of comparing raw images pixel-by-pixel, the model converts each image into
a vector. In this project, CosPlace produces a 512-dimensional descriptor.

So instead of saying:

    “Does this frame look like image 431?”

the system really asks:

    “Is this 512-dimensional vector close to the vector for image 431?”

That is much more stable than raw pixel comparison.


2.2 Nearest-neighbor matching
-----------------------------

Problem:

We need to decide which stored corridor image is most similar to the current live frame.

Core idea:

Take the live descriptor and compare it to every stored descriptor.
Whichever stored descriptor is closest is the best match.

Mathematically:

If q is the query descriptor and d_i is the i-th database descriptor, then we compare:

    distance_i = || q - d_i ||

That “|| q - d_i ||” means Euclidean distance.

Plain English:

- subtract the two vectors
- square and sum the differences
- take the square root

Smaller distance = more similar.

What that means in the project:

- each corridor frame in the database has a descriptor
- the live image gets a descriptor
- the nearest stored descriptor suggests the current corridor step

Common misunderstanding:

People sometimes think the localizer is “recognizing objects.”
It is not. It is recognizing place similarity in descriptor space.


2.3 Why one-frame localization is not enough
--------------------------------------------

If you only trust one frame, the robot can jump to the wrong place because some
corridor regions look similar.

So one-frame matching gives you a guess.
Temporal stabilization decides whether that guess is believable.


PART 3: Temporal Stabilization
==============================

3.1 What problem is it solving?
-------------------------------

Suppose the robot was just localized at step 420.

Now the next frame says:

    best match = step 1190

This could be true.
But it could also be nonsense caused by repeated corridor appearance.

The temporal localizer asks:

    “How expensive would it be to believe this jump?”


3.2 The actual score
--------------------

In the code, each candidate gets a score:

    score = distance term + continuity cost + heading cost

That means:

1. base image-descriptor distance
2. extra cost if the candidate is too far from the last step
3. extra cost if heading is inconsistent


3.3 Continuity penalty
----------------------

From the code:

- max_step_jump = 20
- jump_penalty = 0.05
- backward_penalty = 0.15

Idea:

Small forward motion is normal.
Huge jumps are suspicious.
Backward jumps are even more suspicious in this corridor task.

If delta = candidate_step - previous_step:

- if abs(delta) <= 20, no jump penalty
- if abs(delta) > 20, add:

    0.05 * (abs(delta) - 20)

- if delta < 0, also add:

    0.15 * abs(delta)

Why backward jumps are punished harder:

Because the corridor graph behaves like forward progression most of the time.
If the robot suddenly “goes backward” many steps, that is more likely to be a bad
match than real motion.


3.4 Heading penalty
-------------------

If heading is available, the code adds:

    heading_penalty * absolute_heading_difference

with:

    heading_penalty = 0.002

Meaning:

If the candidate image appears to face a direction very different from the observed
heading, it becomes slightly less believable.

Important project lesson:

Indoors, compass heading was noisy enough that this penalty could do more harm than
good. That is why indoor localization eventually stopped trusting heading.


3.5 Confidence
--------------

After scoring, the code converts the best score into a confidence:

    confidence = 1 / (1 + score)

Why that form?

Because:

- confidence should get smaller when score gets worse
- confidence should stay between 0 and 1

If score = 0, confidence = 1.
If score is large, confidence shrinks.

This is not a probability from a full probabilistic model.
It is a practical confidence surrogate.


3.6 Ambiguity hold
------------------

The code also says:

If the top two candidates are very close in score, and the new best candidate is
far enough away from the current state, then keep the old state instead of jumping.

This is “hold on ambiguity.”

Why it matters:

Sometimes the system is not really sure.
The smartest move is to do nothing rather than hallucinate a location jump.


PART 4: Graph Planning and Checkpoint Progression
=================================================

4.1 Why use a graph?
--------------------

The corridor is not just a pile of images.
It has structure.

A graph captures:

- what comes next
- which steps are connected
- which route to follow to reach a checkpoint

So instead of saying:

    “Drive toward whatever image looks close to the checkpoint”

the runtime says:

    “I am at graph node A, target is graph node B, now follow the path from A to B.”


4.2 Shortest path
-----------------

The graph planner uses shortest path on the corridor graph.

Plain English:

Find the sequence of nodes that gets from current node to target node with minimum
path length.

This is topological planning, not metric motion planning.

It does not care about continuous x-y geometry.
It cares about corridor sequence structure.


4.3 Subgoal hops
----------------

The planner does not always hand the controller the final checkpoint directly.
It picks a subgoal a few hops ahead.

Why?

Because local controllers work better with nearby goals than with far goals.

In the project:

- graph planner default max_subgoal_hops = 3
- indoor runtime in MBRA mode overrides this to 8

So the effective idea is:

    “Don’t chase the entire route at once. Chase the next sensible local target.”


4.4 Checkpoint reached logic
----------------------------

The graph planner checks whether a checkpoint is reached by:

1. requiring enough confidence
2. requiring the current step to be within a tolerance of the target step

The defaults are:

- min_confidence_to_advance = 0.55
- checkpoint_reach_tolerance = 3 steps

So checkpoint reached is not:

    “the controller thinks we’re done”

It is:

    “the localizer says we are close enough to the target step with enough confidence”


4.5 Why skip-past-checkpoint logic exists
-----------------------------------------

This is one of the most important indoor ideas.

If the graph is effectively forward-only, then once the rover has passed a
checkpoint step, there may be no valid backward path.

Without special handling, the system can get stuck forever.

So the runtime adds logic like:

    if current_step > target_step and confidence is good enough:
        advance to next checkpoint

This is not cheating.
It is the runtime admitting the real geometry of the problem.


PART 5: MBRA Math and Intuition
===============================

5.1 What MBRA is doing
----------------------

MBRA is not solving the whole navigation problem.
It is solving the local motion problem:

    current image + goal image -> motion command

That is the key.


5.2 Context window
------------------

The code uses a 6-frame observation history.

Why use multiple frames instead of one?

Because motion is dynamic.

A single frame can be ambiguous.
A short history helps the model understand motion trend and visual progression.


5.3 Fixed vel_past
------------------

One unusual detail is that MBRA uses fixed vel_past inputs:

- linear = 0.5
- angular = 0.0

Why fixed?

Because the project discovered that feeding noisy live control history into the
model did not help.
The fixed values act more like a stable prior than a fragile feedback signal.


5.4 Output trajectory
---------------------

MBRA outputs an 8-step trajectory.
The runtime uses the first step as the immediate command.

Why?

Because the model is really predicting a short future plan, but the runtime only
needs the very next action right now.


5.5 Why no reverse
------------------

This is more intuition than algebra.

MBRA expects forward-view visual context.
If you reverse, the visual relationship between “where I am” and “where I should go”
can become much less meaningful.

So no reverse is not an arbitrary style choice.
It is a context-preservation choice.


PART 6: Outdoor Control Geometry
================================

6.1 Bearing to goal
-------------------

If the rover is at (x1, y1) and the goal is at (x2, y2), then:

    dx = x2 - x1
    dy = y2 - y1

Distance to goal:

    distance = sqrt(dx^2 + dy^2)

Goal bearing:

    bearing = atan2(dy, dx)

Bearing error:

    error = wrap_angle(bearing - current_heading)

This is classical geometry.


6.2 Why wrap angles?
--------------------

Angles loop around.

If current heading is 179 degrees and goal is -179 degrees, naive subtraction says:

    -179 - 179 = -358 degrees

But the real turn needed is only 2 degrees.

That is why the code wraps angles into a standard range like [-pi, pi].


6.3 Classical GPS controller intuition
--------------------------------------

The fallback outdoor GPS controller is a proportional heading controller.

Meaning:

    angular_command = gain * heading_error

clipped to a maximum.

Then forward speed is reduced if the turn is large or if the goal is close.

That is a very standard control idea:

- steer more when the error is larger
- don’t drive fast when sharply misaligned


PART 7: Traversability Math
===========================

7.1 Why a middle image band?
----------------------------

The old idea of using the bottom of the image is bad on this platform because the
bottom often just sees nearby ground.

So the traversability code uses a middle band:

- crop_top_frac = 0.15
- crop_bot_frac = 0.60

Meaning:

Look at the region that is more likely to contain trunks, walls, bushes, or true
forward obstacles.


7.2 Angular bins
----------------

The image is split into angular bins:

- num_bins = 16

Each bin corresponds to a horizontal direction in front of the rover.

For each bin, the code computes a clearance estimate from depth values.

This means the output is not just:

    “obstacle yes/no”

It is:

    “how open is each direction?”


7.3 Why use the 10th percentile?
--------------------------------

Within a bin, the code takes:

    percentile(valid_depths, 10)

Why not mean?

Because mean can hide obstacles.
If most pixels are far away but a few pixels belong to a close obstacle, mean depth
can still look safe.

The 10th percentile is more conservative.
It asks:

    “what does the closer part of this bin look like?”


7.4 Obstacle memory
-------------------

The code keeps a small history:

- memory_frames = 4

Then it takes a minimum over recent per-bin clearances.

Why?

Because if the rover sees an obstacle and then one noisy frame says it disappeared,
the runtime should not instantly trust that disappearance.

This makes obstacles “sticky.”


7.5 Forward clearance
---------------------

The rover also computes forward clearance from the center bins.

Then the logic is roughly:

- if forward clearance < stop_distance_m: stop
- else if forward clearance < slow_distance_m: slow
- else continue normally

Default values:

- obstacle_distance_m = 1.5
- stop_distance_m = 0.60
- slow_distance_m = 1.20


PART 8: Semantic Risk Math
==========================

8.1 The main idea
-----------------

Semantics is not using the full segmentation map in a fancy end-to-end way.
It extracts a few meaningful fractions from specific regions.

Examples:

- how much of the center region is person?
- how much is animal?
- how much is drivable?
- how much is caution-heavy vegetation?


8.2 ROI geometry
----------------

The code uses a raised forward corridor region:

- top = 40% of image height
- bottom = 80%
- left = 30% of image width
- right = 70%

Then it defines:

- a center corridor
- a left half
- a right half

This is important:

The math is not “what is the dominant label in the whole image?”
The math is “what is happening in the part of the image that matters for forward movement?”


8.3 Label grouping
------------------

The code groups labels into:

- DRIVABLE = road, earth, path, sidewalk, dirt_track
- NEUTRAL = grass, field
- HAZARD = person, animal, pole, wall, fence
- CAUTION = tree, plant
- IGNORE = sky

Why grouping matters:

Raw label names are too detailed and too inconsistent.
The runtime needs a smaller decision vocabulary.


8.4 Risk score terms
--------------------

The center-region risk score uses explicit thresholds:

- if person > 0.002:
      score += 0.55 + 18 * person

- if animal > 0.002:
      score += 0.55 + 18 * animal

- if pole > 0.002:
      score += 0.35 + 10 * pole

- if wall > 0.010:
      score += 0.30 + 8 * wall

Vegetation-blocked is different:

    drivable_center < 0.10 AND caution_center > 0.60

Then:

    score += 0.45 + 0.50 * max(0, caution_center - 0.60)

Interpretation:

The system is not asking “is there any tree pixel?”
It is asking whether the center corridor is dominated by the wrong kind of content.


8.5 Left-right bias
-------------------

The code computes a “free score” per side.

For each side:

    free = drivable + 0.30 * neutral - 0.60 * caution

If hard-mode hazards exist, it subtracts even more:

    free -= 4.0 * (person + animal)
    free -= 3.0 * (pole + wall)

Then:

    diff = left_free - right_free
    scale = max(0.25, |left_free| + |right_free|)
    bias = diff / scale

Finally the bias is clipped:

    bias in [-0.50, +0.50]

Meaning:

- positive bias = left looks freer
- negative bias = right looks freer

This is a soft steering suggestion, not a full planner.


PART 9: IMU Safety Math
=======================

9.1 Tilt from acceleration
--------------------------

At rest, gravity defines a reference direction.

The code normalizes the accelerometer vector and compares it to the learned gravity
reference using the dot product.

If u is current normalized acceleration and g is the reference:

    dot = u · g

Then:

    tilt = arccos(dot)

converted into degrees.

Why this works:

If the robot is upright, the vectors align and dot is near 1.
If the robot tilts, the angle between them grows.


9.2 Angular rate
----------------

The code also measures pitch/roll rate using gx and gy:

    rate = sqrt(gx^2 + gy^2)

then converts to degrees per second.

Yaw is not included because yaw can just mean normal turning.


9.3 Why debounce exists
-----------------------

A single noisy sample should not stop the robot.

So the IMU safety logic requires consecutive bad readings before triggering.

That is standard debounce logic:

    “don’t trust one weird tick”


PART 10: Why This Math Was Chosen
=================================

The math in this project is not fancy for the sake of being fancy.
Most of it is actually conservative engineering math:

- nearest-neighbor similarity
- penalties for implausible jumps
- graph shortest path
- proportional steering
- percentile-based obstacle caution
- thresholded semantic scoring
- tilt-angle computation from IMU vectors

The project’s strength is not exotic mathematics.
The project’s strength is using simple math in the right place and not pretending a
single model solves everything.


PART 11: Common Misunderstandings
=================================

Misunderstanding 1:
“CosPlace is the whole indoor navigation system.”

No. It is the localization backbone.

Misunderstanding 2:
“MBRA is doing global planning.”

No. It is short-horizon local control.

Misunderstanding 3:
“LogoNav is the whole outdoor system.”

No. It is the local controller inside the runtime.

Misunderstanding 4:
“Depth or semantics should have solved obstacle handling by themselves.”

No. In this project they were useful but not trustworthy enough for full authority.

Misunderstanding 5:
“More safety layers automatically means safer behavior.”

No. Marathon showed that too much unstable caution can still lead to bad behavior.


PART 12: Quiz Yourself
======================

Basic
-----

1. Why is nearest-neighbor matching better than raw pixel comparison for corridor localization?
2. What problem does temporal stabilization solve?
3. Why is MBRA treated as a local controller instead of a planner?
4. Why is a routed waypoint not the same thing as a mission checkpoint?
5. Why did the project avoid reverse behavior for MBRA?

Intermediate
------------

6. If the previous corridor step was 400 and the best new candidate is 460, what kind of temporal penalty logic matters?
7. If the best semantic label in the whole image is “sky,” why is that not useful for safety?
8. Why is the 10th percentile of depth in a bin more useful than the average depth?
9. Why does the route corridor guard exist even if OSM routing already exists?
10. What exactly does “sole motion authority” mean in the indoor runtime?

Advanced
--------

11. Explain why confidence = 1 / (1 + score) is a useful runtime quantity even though it is not a true probability.
12. Explain why backward penalties make sense in indoor corridor localization.
13. Explain the vegetation-blocked rule in semantics and why it is a compound condition.
14. Explain how aggressive alignment behavior can produce a physical tipping problem even when the software is “trying to be safe.”
15. Explain why the most honest outdoor problem statement is about transition stability, not controller selection alone.


PART 13: The Best Mental Model To Leave With
============================================

If you forget every equation, remember this:

Indoor:

    localization tells us where we are,
    graph planning tells us what comes next,
    MBRA tells us how to move right now.

Outdoor:

    mission logic tells us what target matters,
    routing tells us what local path structure to follow,
    LogoNav tells us how to move right now,
    safety layers decide whether that motion is still acceptable.

Marathon:

    the biggest remaining weakness is transition stability.

If you can explain that clearly, you already understand the project much more deeply
than most people who only read the file names.
