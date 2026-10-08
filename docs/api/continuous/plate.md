# Plates

Indexed draws and observations. A
[`PlateDraw`](#quivers.continuous.plate.PlateDraw) draws one value per
element of a finite set, as `sample v : A <- F(...)` does in source; a
[`VectorisedObserve`](#quivers.continuous.plate.VectorisedObserve)
scores a column of responses at once, as an indexed `observe` does. The
marginalization helpers integrate a finite latent out of a plate of
scores.

::: quivers.continuous.plate
