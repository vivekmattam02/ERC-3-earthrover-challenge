# Checkpoint-Step Logic

This concept is what made indoor much cleaner.

## What It Means

Instead of treating indoor goals as vague target images, the competition path was reframed as exact ordered checkpoint steps in the known corridor dataset.

## Why It Helped

This removed a lot of ambiguity:

- exact target semantics
- cleaner graph planning
- cleaner progression rules
- less operator ambiguity

## What It Changed

The indoor problem became:

- localize in the known corridor
- figure out where the active checkpoint step is
- advance through the checkpoint list in order

That is much better than "drive toward a vaguely similar image."

## Best Related Notes

- [[03 Personal Notes/Indoor Story - Distilled]]
- [[live_indoor_runtime_story]]
- [[erc3_full_documentation]]
