# Current Truth

This is the most important personal note in the vault.

If I am about to say something publicly, I should check this note first.

## No-GPS In One Minute

- **Implemented:** front-camera recording, route preparation, localization,
  adaptive-pursuit launch, and sent rover commands.
- **Not proven:** a reliable autonomous repeat of the taught route.
- **Do not claim:** `smoke_run01_c` is ready for final autonomous use.
- **Next proof:** a short HIL repeat with stable localization and sustained step
  progression.

Read [[00 Home/Current No-GPS - Read This First]] before using the longer
sections below.

## What Is Solid

- indoor corridor localization
- temporal stabilization of localization
- graph path planning and nearby subgoal selection
- runtime wiring from SDK input to localize -> plan -> control
- indoor checkpoint-step competition framing
- manual no-GPS bag recording from the SDK
- bag -> extracted route -> descriptor DB -> navigation graph pipeline
- prepared-route launch flow for no-GPS repeat runs
- `run_01_1521.h5` is the strongest current teach bag
- front-camera-only live route repeat

## What Is Partially Working

- indoor local control under broader edge cases
- no-GPS route-repeat on rough terrain: live commands work, reliable progression does not
- rough-terrain steering and relocalization search: implemented but field-unproven
- post-processing for teach bags
- outdoor mission runtime repeatability
- traversability as a useful but still tuning-sensitive outdoor layer
- semantic risk as a support layer rather than a trusted authority
- marathon safety layering without race-level stability confidence

## What Is Still Unproven

- MBRA as a fully validated practical real-corridor replacement under all live conditions
- final controller choice for no-GPS off-road differential-drive repeat
- long repeat runs on rough terrain without human intervention
- full safety-aware outdoor autonomy with robust repeatability
- long-run outdoor stability without human intervention
- calm waypoint/checkpoint transitions in marathon conditions

## What Is Active Right Now

- the active problem is **not** GPS outdoor mission execution
- the active problem is **no-GPS visual teach-and-repeat**
- current best coverage/reference candidate: `smoke_run01_c`
- current best post-processed comparison route: `run_01_1521_pp_v3`
- current evidence: repeated field trials stuck near startup steps and entered relocalization loops
- current next phase: validate the live reference/start match and prove a short, repeatable HIL segment before another autonomous run

## What We Corrected

- the cleanest-looking clip is **not** automatically the best teach route
- a visually stable segment can be bad for repeat navigation if it has poor
  route progression
- for this rover, route quality depends more on
  `motion-rich coverage + visual progression + relocalization evidence`
  than on visual cleanliness alone
- `run_01_1521_pp_v3` is useful because it proves the post-processing
  direction is improving, but it still does **not** replace `smoke_run01_c`
  as the main route

## What Live Results Actually Say

- indoor reached **8/11 checkpoints**
- outdoor achieved **one full completion** with mixed repeatability across runs
- marathon reached **one checkpoint** and then failed due to unstable turning/transition behavior plus platform instability

## Things I Should Not Overclaim

- do not say outdoor is solved
- do not say semantics became the main outdoor authority
- do not say MBRA was the whole indoor system
- do not say marathon failure means all safety work was useless
- do not say the current no-GPS controller is finished
- do not call `smoke_run01_c` an operational or deployment route
- do not claim the manual bags permit exact teleop replay; their recorded controls are empty
- do not say bag 2 is a better teach route than bag 1
- do not say the post-processed route already surpassed `smoke_run01_c`

## Where This Comes From

- [[erc3_full_documentation]]
- [[competition_results]]
- [[01 Source of Truth/Offroad Track - Full Story]]
- [[01 Source of Truth/Offroad Controller - Full Story]]
- [[01 Source of Truth/No-GPS Field Trial - Findings]]
- [[03 Personal Notes/What We Were Thinking Wrong About Teach Bags]]
- [[90 Archive/CONTEXT - Indoor Source]]
- [[90 Archive/CONTEXT - Research Source]]
