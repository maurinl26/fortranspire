# wave_kernels — tolerance & rationale

## Kernel

Two 2D finite-difference stencils (`update_vx`, `update_sigma`) iterated
for `nsteps = 20` over a 64×64 grid. Initial condition: a smooth
Gaussian bump on `sigma_xx`, uniform `vx = vy = 1e-3`.

## Tolerance

| Metric | Value | Why |
| ------ | ----- | --- |
| `atol` | `1e-12` | Both stencils are pure additions / multiplications over IEEE-754 doubles. No FMA, no transcendentals, no reduction. Bit-equivalence is achievable up to compiler-driven instruction reordering. |
| `rtol` | `1e-10` | Headroom for compiler instruction-scheduling differences between the serial path (`-O2`) and the OpenACC path (`-O2 -fopenacc`, which enables additional loop parallelism). |

## Rationale

The arithmetic in `openacc.f90` is **byte-identical** to `original.f90`
— the only deltas are `!$acc parallel loop` directives plus
`!$acc end parallel loop`. `gfortran -fopenacc` does not change the
arithmetic produced; it changes the **schedule** under which it runs
(CPU threading via libgomp).

Under that constraint, deviations between the two outputs come from:

1. **Loop-iteration order** — `parallel loop collapse(2)` can execute
   `(i, j)` pairs in any order. Each pair writes a distinct array
   cell, so the writes are race-free, but the order in which
   floating-point updates land in cache lines can differ.
2. **Compiler reassociation** — at `-O2`, gfortran applies
   `-fassociative-math`-class optimizations independently in the
   serial and parallel paths. This can swap the addition order on
   the right-hand side of `sigma_xx(i, j) + (... - ...)`.

Both effects are at the last-ULP level for our grid size and step
count. `atol=1e-12, rtol=1e-10` covers them with three orders of
magnitude headroom.

## When to revisit

- If we add a kernel with a **reduction** (sum, max), tolerance needs
  to widen — reductions are genuinely order-sensitive under
  parallelism. Document the new tolerance and the kernel-specific
  reason in this file.
- If we change the grid size to > 1024² or the step count to > 1000,
  error accumulation in the time-stepping may push us past `rtol`.
  Document the empirical bound observed under stable input.
- If we ever switch to `real(4)` (single precision), drop both
  tolerances by ~4 orders of magnitude (single precision has ~7
  decimal digits, double has ~16).

## How to validate locally

```bash
cd tests/fixtures/equivalence/wave_kernels/
gfortran -O2 -fsyntax-only original.f90 driver.f90      # syntax-only sanity
pytest tests/test_equivalence_real_kernels.py -m slow -v
```
