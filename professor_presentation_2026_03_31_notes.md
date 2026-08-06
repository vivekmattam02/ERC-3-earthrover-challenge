# Professor Presentation Notes

Type: Presentation Notes  
Status: Current  
Audience: Presenter

Related slides:
- `professor_presentation_2026_03_31.tex`

## How To Use These Notes

This talk should be:
- short
- simple
- casual
- clear

Do not try to sound too technical.
Do not try to explain every subsystem.
Just explain:
- what we did,
- why we chose it,
- what worked,
- what failed.

Target pacing:
- around 6 to 8 minutes

---

## Slide 1: Title

### What to say
"This is a short update on the EarthRover project. I’ll keep it very simple and only talk about indoor, outdoor, and marathon."

---

## Slide 2: Indoor

### What to say
"I’ll keep this simple and talk about indoor, outdoor, and marathon. For indoor, we chose MBRA as the main controller, and we used exact checkpoint-step navigation. The reason is that indoor is really a known corridor problem, so exact step targets made more sense than vague goals."

### Main point
Indoor choice was made because the problem became clearer.

---

## Slide 3: Indoor Results

### What to say
"For indoor, we reached 8 checkpoints out of 11. That is the strongest result we got there. It also helped us understand the system much better. The main failures indoors were no-path issues from the forward-only graph, stale context, and bad recovery behavior."

### Main point
State the result clearly and honestly.

### Good short line
"Indoor is not perfect, but it is now much cleaner and more understandable."

---

## Slide 4: Outdoor

### What to say
"For outdoor, we kept LogoNav as the main controller and used OSM route expansion with safety layers around it. We chose that because outdoor already had a working base, so the bigger problem was not inventing a new controller. The bigger problem was making the runtime safer and more stable."

### Main point
Outdoor was about runtime hardening, not starting over.

---

## Slide 5: Outdoor Results

### What to say
"For outdoor, one of the runs reached all checkpoints, which was our best result. The others were around a 50 percent success rate overall, and three rounds failed. The main problems were spinning, bad waypoint transitions, and general instability in live conditions."

### Main point
This is the honest outdoor result slide.

---

## Slide 6: Marathon

### What to say
"For the marathon, we kept the same outdoor base but added stricter safety layers like IMU safety, route corridor guard, rerouting, and better waypoint handling. The idea was to make it safer and more stable instead of pretending the robot was fully autonomous."

### Main point
The marathon work was a safety and stability extension.

---

## Slide 7: Marathon Result And Failure Reason

### What to say
"In the marathon run, the robot reached one checkpoint, but after that it started spinning and eventually toppled. The likely reason is that after checkpoint progress, the runtime got confused about the next target and entered repeated alignment behavior. Once the robot keeps turning aggressively like that, stability becomes a real problem."

### Main point
Give one clean reason, not ten speculative reasons.

### Good phrase
"The marathon failure was less about basic navigation and more about unstable behavior after checkpoint transitions."

---

## Slide 8: Final Summary

### What to say
"So the short summary is: indoors, MBRA-first is our best direction and we reached 8 out of 11 checkpoints. Outdoors, LogoNav plus route and safety layers is still our best direction, with one full success and the others around 50 percent. For the marathon, the main thing we still need to fix is stability after checkpoint and waypoint transitions."

### Main point
End in a clean, memorable way.

---

## Final Advice

- keep it calm
- keep it short
- do not overexplain
- say the numbers clearly:
  - indoor: 8 out of 11
  - outdoor: one full success, others around 50 percent, three failed rounds
  - marathon: reached one checkpoint, then spun and toppled
- if you get stuck, come back to:
  - what we chose
  - why we chose it
  - what worked
  - what failed
