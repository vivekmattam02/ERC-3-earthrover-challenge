# Marathon Failure - What Actually Happened

This note is here because the marathon result is easy to oversimplify.

## Wrong Explanation

"The robot failed because the controller was bad."

That is too shallow.

## Better Explanation

- the robot made initial progress
- after checkpoint/waypoint transition, target handling became unstable
- alignment behavior stayed too aggressive for too long
- repeated turning happened on real terrain
- platform stability was not strong enough to absorb that behavior
- the robot toppled

## What This Means

The marathon failure was a **transition + platform dynamics** failure.

It was not proof that:

- routing was useless
- safety layers were useless
- the whole outdoor system had no value

## What To Say Publicly

- we reached the first checkpoint
- failure happened after transition instability
- the lesson was that stability after target handoff matters more than simply adding more caution

## Best Follow-Up Notes

- [[live_outdoor_ultra_marathon_story]]
- [[erc3_full_documentation]]
- [[04 Runs and Failures/Run Outcomes]]
