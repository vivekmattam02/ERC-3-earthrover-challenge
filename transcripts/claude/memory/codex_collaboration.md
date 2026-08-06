---
name: Codex collaboration workflow
description: How Claude and Codex work together on this project, division of labor, consensus process
type: feedback
---

User runs Claude (me) and Codex in parallel on the same repo.

**Workflow:** "one edits, one inspects"
- Claude proposes changes → user relays to Codex → Codex reviews → user relays approval → Claude edits
- Codex proposes changes → user relays to Claude → Claude reviews → consensus reached → implementation

**Division of labor (as of 2026-03-25):**
- Claude: indoor controller fixes (gyro correction, EMA smoothing, min_linear, freeze detection), context/memory management, prelocalize_checkpoints.py tool, runtime integration
- Codex: semantic segmentation research (SegFormer-B0), runtime semantic_risk_estimator.py, indoor CLI additions (--checkpoint-images, --target-image-name, --checkpoint-steps), architectural reviews, offline analysis

**Key consensus decisions reached:**
1. Traversability is soft bias only (15% angular), no hard stop/slow from depth
2. LogoNav remains primary outdoor controller
3. max_depth auto-detection: vkitti→80, hypersim→20
4. Semantic segmentation: soft angular bias only (15% vegetation, 25% hard alerts), never stop/slow
5. Indoor gyro-based heading correction is the primary course-correction signal (compass disabled indoors)
6. Indoor checkpoint pre-localization for competition advantage

**How to apply:** Always propose before editing. Never make changes without Codex agreement relayed through the user.
