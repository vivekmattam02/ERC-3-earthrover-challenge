---
name: Collaboration feedback
description: Rules for how to approach work on this project
type: feedback
---

Do not restart the architecture from scratch unless you find a concrete technical flaw in the code or evidence.
**Why:** The user has rebuilt context many times across sessions and is tired of it.
**How to apply:** Localization backbone is strong; planning is broadly fine; do not propose swapping them out.

---

MBRA is only a candidate short-horizon local controller, not the backbone.
**Why:** Key conceptual correction from earlier sessions — MBRA/LogoNav were misread as the whole stack.
**How to apply:** Never let MBRA creep back into localization, planning, or the full system role.

---

CONTEXT.md is a living shared note — update it continuously as work progresses.
**Why:** User uses CONTEXT.md as the primary persistent state between sessions.
**How to apply:** After any meaningful code change or discovery, append to the Session Log section in CONTEXT.md.

---

The correct manual teleop script is `keyboard_control.py`, not `simple_control.py`.
**Why:** keyboard_control.py uses the browser-backed `/control` endpoint; simple_control.py uses `/control-legacy` (direct RTM).
**How to apply:** For manual testing, default to keyboard_control.py. simple_control.py is still usable for quick legacy tests.

---

Use `erv` conda env for all baseline/runtime work; only activate `mbra` env for MBRA-specific testing.
**Why:** erv has all needed deps for the simple controller path. mbra env is still not fully validated for runtime deps.
**How to apply:** Don't suggest activating mbra unless explicitly testing --controller mbra.

---

Always reach consensus with Codex before making code changes.
**Why:** User runs Claude + Codex in parallel with a "one edits, one inspects" workflow. Making changes without agreement wastes time and causes confusion.
**How to apply:** Propose exact changes first. Wait for user to relay Codex's approval. Only then edit. Never jump ahead.

---

Prefer small, safe changes over ambitious rewrites.
**Why:** User explicitly said "not so catastrophic changes that will make the robot stop." Past depth-safety work proved that untested changes can freeze the robot (100% stop rate from wrong max_depth).
**How to apply:** Always validate offline first. Never deploy hard stop/slow/veto behavior without calibration data proving it won't over-trigger.

---

Metric depth on this camera is not reliable for hard safety decisions.
**Why:** Offline calibration proved thin obstacles (people, poles) are invisible to Depth Anything V2 on this camera. Broad vegetation/walls show weak signal (4-5m vs 5-6m open). The depth model over-estimates distances for small objects.
**How to apply:** Do not build hard stop/slow/veto around metric depth thresholds. Only use depth as a soft angular bias. Next safety layer should be semantic segmentation.

---

Indoor navigation is NOT GPS-based — it is visual place recognition in a known corridor.
**Why:** User corrected this misframing once. The indoor system uses CosPlace VPR + temporal filtering + topological graph planning. Compass is disabled indoors; gyro is the only heading signal.
**How to apply:** Never reference GPS, compass, or satellite positioning when discussing the indoor system. Heading correction uses gyro Z only.

---

Always include `--send-control` when running the robot.
**Why:** Without it, the runtime runs in DRY RUN mode and sends no commands — robot won't move. User hit this during outdoor testing.
**How to apply:** Every run command suggestion must include `--send-control`. If the user is testing without the robot, omit it deliberately but flag it.
