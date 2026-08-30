# ADR 0001: Use density-map regression

## Status

Accepted

## Context

Crowd scenes are dense and perspective-distorted. Traditional object detection is unreliable because people overlap heavily and vary significantly in scale.

## Decision

Use density-map regression as the supervised target instead of a single scalar count or detection-only representation.

## Consequences

- preserves spatial distribution information,
- enables count estimation by integrating the output map,
- requires stable preprocessing and target scaling.
