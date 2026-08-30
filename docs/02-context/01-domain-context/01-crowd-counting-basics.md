# Crowd Counting Basics

Crowd counting estimates the number of people in a scene from an image. In dense or crowded scenes, direct detection is often unreliable because people overlap, occlude each other, and appear at varied scales.

Density-map regression provides a spatial estimate of where people are concentrated. The total count can then be recovered by summing all density values in the output map.
