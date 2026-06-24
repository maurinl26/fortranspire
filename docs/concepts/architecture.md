# Architecture

The pipeline is a six-stage LangGraph state machine. Two stages call the
LLM; four are deterministic.

```text
📂 kernel.f90  (Fortran monolithique)
     │
     ▼ 🔍 parser          Loki AST — detects INTENT, SAVE, COMMON, loops, I/O
     │                    Deterministic — no LLM
     │
     ▼ 🔧 extractor       LLM (1 call) — extracts 2-D loops as MODULE
     │                    subroutines, removes COMMON, exposes SAVE
     │                    as INTENT(INOUT)
     │                    → module_kernels.f90  +  driver.f90
     │
     ▼ ✨ pure_elemental   AST rules — annotate PURE/ELEMENTAL
     │                    Validates: no I/O, no SAVE, INTENT explicit
     │
     ▼ 🚀 openacc         LLM (1 driver call) — !$acc parallel loop
     │                    collapse(2), !$acc data copyin/copy around
     │                    the time loop
     │
     ▼ 🐍 cython_wrapper  LLM (2 calls) — .pyx + kernel_c.h (iso_c_binding),
     │                    NumPy typed memoryviews, np.asfortranarray()
     │
     ▼ ✅ validation       gfortran × 2 flavors → nvfortran -acc (GPU)
     │                    Deterministic — compilation
     │
     📦 output/fortran_gpu/module_kernels_gpu.f90
        output/cython/module.pyx
```

## LLM budget per run

Four LLM calls maximum:

| Stage             | Calls | Role                                                                   |
| ----------------- | ----- | ---------------------------------------------------------------------- |
| `extractor`       |   1   | Refactor a monolithic `PROGRAM` into a `MODULE` of subroutines         |
| `openacc`         |   1   | Insert OpenACC parallel-loop and data-region pragmas around the driver |
| `cython_wrapper`  |   2   | Generate the `.pyx` body and the `iso_c_binding` C header              |

At Mistral-Large tariffs this is roughly 0.06 USD per kernel and about two
minutes wall-clock. Loki carries the deterministic AST work; the LLM only
intervenes where semantic understanding is required.

## State shape

The pipeline state is a typed dict carried through the LangGraph nodes:

- `fortran_filepath`, `fortran_code` — inputs.
- `ast_info` — Loki AST summary.
- `module_fortran`, `driver_fortran`, `kernel_names` — after extraction.
- `pure_elemental_fortran`, `openacc_fortran` — after purity and pragma
  passes.
- `cython_pyx`, `cython_header`, `cython_setup` — wrapper artifacts.
- `validation_passed`, `validation_log` — final compiler outcome.

Each node reads what it needs and writes only its own keys, so individual
stages can be replayed without rerunning the full pipeline.

## Human-in-the-loop

Every intermediate artifact is written to disk before the next stage runs,
so a reviewer can inspect (or hand-edit) the extracted module, the OpenACC
driver, or the Cython wrapper between stages. Re-running the pipeline from
an existing intermediate file skips the LLM call for that stage.

## Why this sequence?

The pipeline follows an **activation order** — each stage makes the next
possible. It's not arbitrary.

**Stage 1 — Extraction: monolithic → modular**

Codes like `seismic_CPML_2D` are monolithic `PROGRAM`s with inline FD
loops and no explicit `INTENT`.

- Without explicit `INTENT` → impossible to decide `copyin` (read-only)
  vs `copy` (modified in-place) for OpenACC.
- Without separate subroutines → OpenACC can't target the right loops.
- Without a `MODULE` → Cython can't emit a clean `cdef extern`.

**Stage 2 — `PURE`/`ELEMENTAL`: side-effects → pure functions**

| Property | GPU relevance | JAX relevance |
| -------- | ------------- | ------------- |
| No I/O | I/O doesn't execute on device | Same |
| No `SAVE` | No hidden state → independent threads | Required for `jit` |
| Explicit `INTENT` | Determines `copyin` vs `copy` | Determines JAX arguments |
| Determinism | Result identical regardless of thread order | Required for `vmap` |

**Stage 3 — OpenACC: complete pattern for a 2D FD stencil**

```fortran
! Kernel — !$acc parallel loop collapse(2), PURE removed
subroutine update_velocity_x(vx, sigma_xx, sigma_xy, rho, DELTAX, DELTAY, DELTAT, NX, NY)
  real(dp), intent(in)    :: sigma_xx(NX,NY), sigma_xy(NX,NY), rho(NX,NY)
  real(dp), intent(inout) :: vx(NX,NY)
  real(dp), intent(in)    :: DELTAX, DELTAY, DELTAT
  integer,  intent(in)    :: NX, NY
  real(dp) :: value_dsigma_xx_dx, value_dsigma_xy_dy   ! scalars → private()

  !$acc parallel loop collapse(2) private(value_dsigma_xx_dx, value_dsigma_xy_dy)
  do j = 2, NY
    do i = 2, NX
      value_dsigma_xx_dx = (sigma_xx(i,j) - sigma_xx(i-1,j)) / DELTAX
      value_dsigma_xy_dy = (sigma_xy(i,j) - sigma_xy(i,j-1)) / DELTAY
      vx(i,j) = vx(i,j) + (value_dsigma_xx_dx + value_dsigma_xy_dy) * DELTAT / rho(i,j)
    enddo
  enddo
  !$acc end parallel
end subroutine

! Driver — !$acc data ONCE around the 2000 time steps
!$acc data copyin(lambda,mu,rho,b_x,a_x,K_x,...) &
!$acc      copy(vx,vy,sigma_xx,sigma_yy,sigma_xy,memory_dvx_dx,...)
do it = 1, NSTEP
  call update_stress_xx_yy(...)
  call update_velocity_x(...)
  if (mod(it, IT_DISPLAY) == 0) then
    !$acc update host(vx, vy)    ! pull back to CPU for display only
    print *, 'velocnorm =', maxval(sqrt(vx**2 + vy**2))
  endif
enddo
!$acc end data
```

Expected gain: NX=101, NY=641, NSTEP=2000 → ~10s CPU → ~0.1s A100 (×100).

**Stage 4 — Cython → Python without copy**

```python
import numpy as np
import seismic_cpml_2d_gpu as gpu_module

vx = np.asfortranarray(np.zeros((NX, NY)))   # column-major = Fortran layout
gpu_module.update_velocity_x(vx, sigma_xx, sigma_xy, rho, ...)
# Typed memoryviews = direct NumPy buffer access, zero copy
```

**Phase 2 — JAX: `PURE` subroutines become JAX functions directly**

| Fortran `PURE` | JAX equivalent |
| -------------- | -------------- |
| `PURE subroutine f(a, b, c_inout)` | `@jax.jit def f(a, b) -> c` |
| `INTENT(IN)` | JAX argument (immutable) |
| `INTENT(INOUT)` | Returned value |
| Independent `do i,j` | `jax.vmap` or implicit vectorisation |
| `do it` (time loop with state) | `jax.lax.scan` with carry |
| `ELEMENTAL function f(x)` | `jax.vmap(f, in_axes=0)` |

The translation Fortran `PURE` → JAX is **mechanical**: `PURE`
subroutines are pure mathematical functions — exactly what JAX
compiles to XLA.
