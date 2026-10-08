# Program Steps

The records a [`MonadicProgram`](programs.md#quivers.continuous.programs.MonadicProgram)
is built from: [`Draw`](#quivers.continuous.program_steps.Draw),
[`Observe`](#quivers.continuous.program_steps.Observe),
[`Let`](#quivers.continuous.program_steps.Let), and
[`Score`](#quivers.continuous.program_steps.Score), together with the
[`Indexed`](#quivers.continuous.program_steps.Indexed) argument form and
[`reading`](#quivers.continuous.program_steps.reading), through which a
step function declares the names it reads. The DSL compiler emits these
records, so a program built from them in Python is the program the
compiler would build from the equivalent source; the
[programs guide](../../guides/programs.md#building-programs-in-python)
works through one.

::: quivers.continuous.program_steps
