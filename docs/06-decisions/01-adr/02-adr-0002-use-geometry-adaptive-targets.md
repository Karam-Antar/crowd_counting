# ADR 0002: Use geometry-adaptive targets

## Status

Accepted

## Context

Different parts of the same image may contain people at very different scales because of perspective and camera distance.

People are highly overlapping because of high density.

## Decision

Use geometry-aware density targets that better reflect the varying crowd scale across the image.

## Consequences

- better handling of scale variation,
- more realistic target generation for dense scenes,
- stronger dependence on clean dataset preparation.
