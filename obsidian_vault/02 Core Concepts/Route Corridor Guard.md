# Route Corridor Guard

This is one of the most important outdoor runtime concepts.

## What It Means

OSM routing gives a preferred pedestrian route, but routing alone is not enough.

The runtime still needs a rule that says:

"If the rover drifts too far from the intended route corridor, stop and react."

## Why It Matters

The outdoor problem was not just reaching checkpoints. It was staying near the correct path while doing so.

That is especially important for:

- sidewalk discipline
- reroute logic
- marathon safety

## What It Did In Practice

- watched deviation from the routed path
- triggered stop/reroute behavior when deviation persisted
- made route discipline executable instead of aspirational

## Best Related Notes

- [[live_outdoor_ultra_marathon_story]]
- [[live_outdoor_runtime_explained]]
- [[03 Personal Notes/Outdoor Story - Distilled]]
