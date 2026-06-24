# Fortran patterns & transformation rules

`fortranspire` combines deterministic Loki AST analysis with targeted LLM
calls to handle the recurring patterns in legacy scientific Fortran 90.
This page documents each pattern with before/after code, and the
transformation rule applied to OpenACC (Phase 1) and JAX (Phase 2)
targets.

## Summary table

| # | Pattern | What the pipeline does |
| -- | ------- | ---------------------- |
| 1 | Missing `INTENT` | Inferred from data-flow; `INTENT(IN|OUT|INOUT)` added |
| 2 | `SAVE` (implicit or explicit) | Promoted to explicit `INTENT(INOUT)` argument |
| 3 | `COMMON` blocks | Replaced by explicit argument list; driver passes through |
| 4 | `POINTER` / `TARGET` | Detected; flagged when they prevent `PURE`/`ELEMENTAL` |
| 5 | Implicit typing | `IMPLICIT NONE` enforced; explicit `KIND` for all reals |
| 6 | AoS → SoA + `collapse` | Derived-type arrays split; loop nests fused |
| 7 | Stencil vs recurrence | Stencil → `parallel loop collapse(2)`; recurrence → `lax.scan` |
| 8 | `ELEMENTAL` + `!$acc routine seq` | Pointwise functions made callable from GPU loops |
| 9 | Module-private state | Surfaced as `INTENT(INOUT)` at the kernel boundary |
| 10 | `LOGICAL PARAMETER` flags | Converted to `#ifdef` blocks for compile-time elimination |
| 11 | MPI halo exchange | Phase 3 target — GHEX GPU-to-GPU (planned) |
| 12 | Fortran I/O | Phase 4 target — xarray/zarr + DLPack (planned) |

---

## 1. `INTENT` — the key to everything else

`INTENT` defines the contract of each argument. Without explicit
`INTENT`, neither OpenACC nor JAX can work correctly.

```fortran
! Before — implicit INTENT, total ambiguity
subroutine update_stress(vx, sigma_xx, NX)
  double precision vx(NX), sigma_xx(NX)   ! IN or INOUT?

! After — explicit INTENT, clear contract
subroutine update_stress(vx, sigma_xx, NX)
  integer,          intent(in)    :: NX
  double precision, intent(in)    :: vx(NX)
  double precision, intent(inout) :: sigma_xx(NX)
```

**Transformation rules:**

| INTENT | OpenACC | JAX |
| ------ | ------- | --- |
| `IN` | `copyin(arr)` — copied once to GPU before the loop | Immutable argument |
| `INOUT` | `copy(arr)` — bidirectional sync | Returned by the function |
| `OUT` | `copyout(arr)` — pulled back after compute | Return value |
| Not declared | ⚠️ Loki infers from reads/writes | ⚠️ Blocking — must be resolved |

---

## 2. `COMMON` blocks — the global state to eliminate

`COMMON` is a memory block shared between routines — the enemy of both
GPU and JAX.

```fortran
! Before — COMMON block, implicit global state
COMMON /grid/ dx, dy, NX, NY
COMMON /fields/ vx(1000,1000), sigma_xx(1000,1000)

subroutine update_stress()
  ! vx and sigma_xx are accessible implicitly
  sigma_xx(i,j) = sigma_xx(i,j) + vx(i,j) * dx
end subroutine

! After — explicit arguments inside a MODULE
MODULE seismic_kernels
contains
  subroutine update_stress(vx, sigma_xx, dx, NX, NY)
    integer,          intent(in)    :: NX, NY
    double precision, intent(in)    :: dx, vx(NX,NY)
    double precision, intent(inout) :: sigma_xx(NX,NY)
    sigma_xx(i,j) = sigma_xx(i,j) + vx(i,j) * dx
  end subroutine
END MODULE
```

**Transformation rules:**

- **OpenACC**: impossible to annotate `copyin`/`copy` on `COMMON` →
  extract as explicit arguments.
- **JAX**: no notion of mutable global state → everything must be an
  argument or return.
- **Action**: the `extractor` agent replaces every `COMMON` with
  `INTENT(IN|INOUT)` arguments in the generated `MODULE`.

---

## 3. `SAVE` — hidden state between calls

`SAVE` preserves a local variable's value across calls — hidden state.

```fortran
! Before — SAVE variable, hidden state between calls
subroutine update_memory(dvx_dx)
  real, save :: psi_vx = 0.0    ! initialised once, persists
  psi_vx = b_x * psi_vx + a_x * dvx_dx
  dvx_dx = dvx_dx / K_x + psi_vx
end subroutine

! After — state passed explicitly
subroutine update_memory(dvx_dx, psi_vx, b_x, a_x, K_x)
  real, intent(inout) :: psi_vx   ! state exposed, caller-managed
  real, intent(inout) :: dvx_dx
  real, intent(in)    :: b_x, a_x, K_x
  psi_vx = b_x * psi_vx + a_x * dvx_dx
  dvx_dx = dvx_dx / K_x + psi_vx
end subroutine
```

**Transformation rules:**

- **OpenACC**: a `SAVE` variable per GPU thread → race condition. Must
  become `INTENT(INOUT)` or a thread-indexed array.
- **JAX**: `SAVE` breaks functional purity → `jax.lax.scan` handles
  state across iterations.
- **Action**: the `extractor` lifts `SAVE` variables to `INTENT(INOUT)`
  arguments; subroutines containing `SAVE` cannot be annotated `PURE`.

---

## 4. `POINTER` — dangerous aliasing for GPU

Fortran pointers can reference arbitrary memory — incompatible with
`!$acc data` clauses that require concrete arrays of known size.

```fortran
! Before — pointer with potential aliasing
real, pointer :: field(:,:)
field => vx    ! or sigma_xx depending on context

! After option A — replace with allocatable (if ownership is clear)
real, allocatable :: field(:,:)
allocate(field(NX, NY))

! After option B — pass the target directly as argument INTENT(IN)
subroutine process(field, NX, NY)
  real, intent(inout) :: field(NX, NY)
```

**Transformation rules:**

- **OpenACC**: pointers work if the target is known and unique, but
  `!$acc data` requires a concrete sized array → prefer `allocatable`.
- **JAX**: no pointers — replace with array slices (`jnp.array[i:j]`).
- **Action**: Loki detects pointer-target associations; if the target
  is static, the agent replaces with `allocatable` or direct argument.

---

## 5. Array of Structures → Structure of Arrays (AoS → SoA + `collapse`)

Derived-type arrays in Fortran are stored as **Array of Structures**
(AoS): the fields of one element are contiguous in memory. On GPU, all
threads of a warp access the *same field* on *different elements* —
AoS forces non-coalesced accesses.

```fortran
! AoS — bad for GPU (non-coalesced access)
type :: point_t
  real :: x, y, vx, vy
end type
type(point_t) :: particles(N)

do i = 1, N
  particles(i)%vx = particles(i)%vx + particles(i)%x * dt   ! thread i jumps 4 reals at a time
end do

! SoA — optimal GPU (coalesced access, column-major Fortran)
real :: x(N), y(N), vx(N), vy(N)

!$acc parallel loop
do i = 1, N
  vx(i) = vx(i) + x(i) * dt   ! contiguous threads → one coalesced memory access
end do
```

**For 2D loops — `collapse(2)`:**

Without `collapse`, only the outer `j` is parallelised (NY threads).
With `collapse(2)`, both loops fuse into NX×NY independent threads —
full GPU utilisation.

```fortran
! Without collapse — only NY threads
!$acc parallel loop
do j = 2, NY
  do i = 2, NX                  ! inner loop stays sequential in each thread
    sigma_xx(i,j) = sigma_xx(i,j) + ...
  end do
end do

! With collapse(2) — NX×NY threads, all cells in parallel
!$acc parallel loop collapse(2) private(tmp_dx, tmp_dy)
do j = 2, NY
  do i = 2, NX
    sigma_xx(i,j) = sigma_xx(i,j) + ...
  end do
end do
```

**Transformation rules:**

| Source | OpenACC | JAX |
| ------ | ------- | --- |
| `type(t) :: arr(N)` (AoS) | Split into scalar SoA arrays | `pytree` or separate `jnp.array` fields |
| Independent 2D `do j; do i` | `!$acc parallel loop collapse(2)` | `jax.vmap` on two axes or implicit vectorisation |
| 2D loop with stencil `(i-1,j)` | `collapse(2)` OK if `i-1` is from an array already on GPU | Same — JAX accesses `a[i-1]` as slice |

> ⚠️ Fortran is column-major. Dimension **i** varies fastest in memory.
> For coalesced GPU access, the inner loop must iterate over **i**
> (dimension 1) — typical of FD stencils.

---

## 6. Nested dependencies — non-parallelisable loops

A loop is parallelisable only if each iteration is **independent**.
Dependencies on `i-1` in the *same dimension* break this.

```fortran
! Case 1 — FD stencil (dependency on i-1 of ANOTHER array) → parallelisable
!$acc parallel loop collapse(2)
do j = 2, NY
  do i = 2, NX
    vx(i,j) = vx(i,j) + (sigma_xx(i,j) - sigma_xx(i-1,j)) / dx  ! sigma_xx is read-only
  end do
end do

! Case 2 — recurrence on same array → NOT parallelisable
do i = 2, N
  a(i) = coeff * a(i-1) + source(i)   ! a(i) depends on a(i-1) from previous step

! Case 3 — time loop (temporal dependency) → sequential on host
do it = 1, NSTEP
  call update_stress(...)    ! state it+1 depends on state it
  call update_velocity(...)
end do
```

**Transformation strategies:**

| Dependency type | OpenACC | JAX |
| --------------- | ------- | --- |
| **FD stencil** `a(i,j) ← b(i-1,j)` (different arrays) | `!$acc parallel loop collapse(2)` ✅ | `jax.vmap` or implicit vectorisation ✅ |
| **Recurrence** `a(i) = f(a(i-1))` (same array) | ❌ Not parallelisable — keep sequential or reformulate | `jax.lax.scan` ✅ |
| **Time loop** `u(t+1) = f(u(t))` | `!$acc data` around the loop (GPU kernels, time loop on host) | `jax.lax.scan` with carry ✅ |
| **Reduction** `sum += a(i)` | `!$acc loop reduction(+:sum)` ✅ | `jnp.sum(a)` ✅ |

**Worked example — time loop to JAX:**

```fortran
! Fortran — sequential time loop on host, GPU kernels in parallel
!$acc data copyin(lambda,rho) copy(vx,vy,sigma_xx)
do it = 1, NSTEP                                ! sequential host — temporal dependency
  call update_stress(vx, sigma_xx, ...)         ! GPU kernel (2D collapse)
  call update_velocity(sigma_xx, vx, ...)       ! GPU kernel (2D collapse)
end do
!$acc end data
```

```python
# JAX — time loop → jax.lax.scan (jit-compiled, differentiable)
def time_step(carry, _):
    vx, vy, sigma_xx, sigma_yy, sigma_xy = carry
    sigma_xx, sigma_yy = update_stress(vx, vy, sigma_xx, sigma_yy, ...)
    vx, vy = update_velocity(sigma_xx, sigma_yy, sigma_xy, vx, vy, ...)
    return (vx, vy, sigma_xx, sigma_yy, sigma_xy), None

# Launch NSTEP iterations in one XLA-compiled call
(vx_f, vy_f, *_), _ = jax.lax.scan(time_step, init_carry, xs=None, length=NSTEP)
```

**JAX advantage**: `jax.lax.scan` is differentiable —
`jax.grad(loss)(params)` backpropagates through all NSTEP iterations.
Useful for seismic inversion (FWI) or surrogate training.

---

## 7. `ELEMENTAL` + OpenACC — the right pattern

An `ELEMENTAL` procedure **cannot** contain an OpenACC compute
directive (`!$acc parallel`, `!$acc kernels`) — same constraint as
`PURE`. But it's *perfect* for `!$acc routine seq`: it runs
sequentially in each GPU thread, called from a parent `!$acc parallel
loop`.

```fortran
! Correct pattern — ELEMENTAL + !$acc routine seq
ELEMENTAL function pml_update(psi, field_deriv, b, a, K) result(corrected)
  !$acc routine seq           ! ← allowed inside ELEMENTAL (not a compute construct)
  real(dp), intent(in) :: psi, field_deriv, b, a, K
  real(dp) :: psi_new, corrected
  psi_new   = b * psi + a * field_deriv
  corrected = field_deriv / K + psi_new
end function

! The parallel loop is in the PARENT routine — not in the ELEMENTAL
subroutine update_velocity_x(vx, sigma_xx, psi_dvx, b_x, a_x, K_x, ...)
  !$acc parallel loop collapse(2) private(dvx_dx)
  do j = 2, NY
    do i = 2, NX
      dvx_dx   = (sigma_xx(i,j) - sigma_xx(i-1,j)) / dx
      dvx_dx   = pml_update(psi_dvx(i,j), dvx_dx, b_x(i), a_x(i), K_x(i))  ! ← GPU call
      vx(i,j)  = vx(i,j) + dvx_dx * dt / rho(i,j)
    end do
  end do
  !$acc end parallel
end subroutine
```

**Allowed combinations:**

| Procedure | `!$acc parallel loop` inside | `!$acc routine seq` | Callable from GPU |
| --------- | ---------------------------- | ------------------- | ----------------- |
| Standard `SUBROUTINE` | ✅ | ✅ | With `!$acc routine` |
| `PURE SUBROUTINE` | ❌ (standard) | ✅ | With `!$acc routine seq` |
| `ELEMENTAL FUNCTION` | ❌ | ✅ ← **correct usage** | ✅ from parallel loop |
| `ELEMENTAL SUBROUTINE` | ❌ | ✅ | ✅ from parallel loop |

> 💡 **Rule**: `ELEMENTAL` is the right candidate for *pointwise*
> computations (one stencil point, a PML correction, a source term).
> The `!$acc parallel loop collapse(2)` stays in the parent routine
> iterating over all points.

---

## 8. Explicit types — no compiler-inferred mixed precision

Fortran allows implicit declarations and silent promotions. On GPU,
this ambiguity translates to mixed fp32/fp64 instructions and costly
inter-register conversions.

**Absolute rule before translation**: `IMPLICIT NONE` + explicit
`KIND`-tagged types.

```fortran
! Before — precision left to the compiler
REAL dx, dy                      ! 32 or 64 bits depending on -r8 / -fdefault-real-8?
DOUBLE PRECISION vx(NX, NY)      ! portable but stylistically inconsistent
REAL*8 sigma_xx(NX, NY)          ! non-standard extension (GCC/Intel only)
INTEGER NX                       ! OK — 32-bit integers by default

! After — precision declared explicitly via KIND parameter
integer, parameter :: dp = selected_real_kind(15, 307)   ! IEEE 754 double (64-bit)
integer, parameter :: sp = selected_real_kind(6,  37)    ! IEEE 754 single (32-bit)

real(dp) :: dx, dy               ! 64-bit everywhere — consistent with nvfortran -acc
real(dp) :: vx(NX, NY)
real(dp) :: sigma_xx(NX, NY)
integer  :: NX, NY               ! 32-bit integer — correct
```

**Mixed precision — when it's intentional:**

On A100, fp32 ops are 2× faster than fp64. Hybrid codes may use sp for
working arrays and dp for accumulation:

```fortran
! Explicit mixed precision — compiler infers nothing
real(sp), intent(in)    :: source_term(NX, NY)  ! low-precision input (sensors)
real(dp), intent(inout) :: accumulated(NX, NY)  ! high-precision accumulation

! Explicit conversion required (never let the compiler silently promote)
accumulated(i,j) = accumulated(i,j) + real(source_term(i,j), dp)
```

**Transformation rules:**

| Source pattern | Transformation | Note |
| -------------- | -------------- | ---- |
| `REAL x` | `real(dp) :: x` | Assume dp unless told otherwise |
| `DOUBLE PRECISION x` | `real(dp) :: x` | Normalise the style |
| `REAL*8 x` | `real(dp) :: x` | Non-standard extension → portable |
| `REAL*4 x` | `real(sp) :: x` | Explicit if intended |
| `COMPLEX x` | `complex(dp) :: x` | Visco-elastic, complex acoustics |
| Implicit promotion | `real(x, dp)` explicit | Never leave `x + 1.0` if `x` is dp |
| Literals | `1.0_dp` instead of `1.0d0` | Consistent with KIND parameter |

**For JAX**: `jnp.float64` by default; force with
`jax.config.update("jax_enable_x64", True)`. Mixed precision possible
with explicit `x.astype(jnp.float32)`.

---

## 9. Logical flags `USE_xx` → compile-time directives

Scientific Fortran codes often use `LOGICAL PARAMETER` as feature
switches:

```fortran
LOGICAL, PARAMETER :: USE_PML        = .TRUE.
LOGICAL, PARAMETER :: USE_ATTENUATION = .FALSE.
LOGICAL, PARAMETER :: SAVE_SNAPSHOTS  = .TRUE.
```

**GPU problem**: even if these constants are compile-time, `if
(USE_PML)` branches inside a `!$acc parallel loop` generate dead code
that *some* compilers don't eliminate cleanly → potential warp
divergence.

**Recommended transformation — CPP preprocessor:**

```fortran
! kernel.F90 (extension .F90 = automatic preprocessing with nvfortran/gfortran)

#ifdef USE_PML
  ! PML memory correction — compiled only if -DUSE_PML
  memory_dvx_dx(i,j) = b_x(i) * memory_dvx_dx(i,j) + a_x(i) * dvx_dx
  dvx_dx = dvx_dx / K_x(i) + memory_dvx_dx(i,j)
#endif
#ifdef USE_ATTENUATION
  sigma_xx(i,j) = sigma_xx(i,j) - tau_sigma * memory_sigma(i,j)
#endif
```

```bash
# Compile with features enabled
nvfortran -acc -gpu=cc80 -cpp \
  -DUSE_PML \
  -o seismic_gpu kernel.F90

# Variant without PML (comparative benchmark)
nvfortran -acc -gpu=cc80 -cpp \
  -o seismic_gpu_nopml kernel.F90
```

**Multi-target equivalences:**

| Fortran source | OpenACC / nvfortran | JAX |
| -------------- | ------------------- | --- |
| `LOGICAL, PARAMETER :: USE_PML = .TRUE.` | `#define USE_PML` → `-DUSE_PML` | `USE_PML = True` (Python constant) |
| `if (USE_PML) then ... end if` | `#ifdef USE_PML ... #endif` | `if USE_PML: ...` (jit-trace time) |
| `if (USE_PML)` inside parallel loop | `#ifdef` → **dead-code eliminated** | `jax.lax.cond(USE_PML, f_pml, f_nopml, args)` if differentiable |
| Multi-valued flag `INTEGER, PARAMETER :: SCHEME = 2` | `#if SCHEME == 2 ... #endif` | `if SCHEME == 2: ...` at trace time |

```python
# JAX — Python flags are evaluated once at jit-trace, not on each iteration
USE_PML = True

@jax.jit
def update_velocity(vx, sigma_xx, psi_dvx, ...):
    dvx_dx = (sigma_xx[i,j] - sigma_xx[i-1,j]) / dx
    if USE_PML:               # ← evaluated ONCE at jit, not per iteration
        psi_dvx = b_x * psi_dvx + a_x * dvx_dx
        dvx_dx  = dvx_dx / K_x + psi_dvx
    return vx + dvx_dx * dt / rho[i,j], psi_dvx

# If USE_PML must be differentiable → jax.lax.cond
dvx_dx, psi = jax.lax.cond(
    use_pml_flag,
    lambda args: pml_correction(*args),
    lambda args: (args[0], args[1]),
    (dvx_dx, psi_dvx),
)
```

> ⚠️ **Agent action**: Loki detects `LOGICAL PARAMETER` with pattern
> `USE_*` or `APPLY_*`. The `extractor` converts them into `#ifdef`
> blocks in the generated `.F90` and documents active flags in a header.

---

## 10. MPI halo exchange → GHEX (GPU-to-GPU)

> ⚠️ **Phase 3 scope — not yet implemented.** Documented for planning.

Multi-domain MPI codes exchange **halos** (ghost bands) between
processes at each time step. In the classical scheme, these exchanges
go through CPU memory — even if the arrays are on GPU:

```
GPU (proc 0)          CPU                GPU (proc 1)
   vx_local  ──acc update host──►  vx_host  ──MPI_Send──►  vx_host  ──acc update device──►  vx_local
   (device)          ↑                                                        ↓
                CPU roundtrip                                          CPU roundtrip
```

**Cost**: 2× PCIe transfers + MPI latency per time step → cancels most
GPU gain on multi-node clusters.

**Solution — GHEX (GridTools, ETH Zürich)**: direct GPU-to-GPU
exchanges via RDMA (NVLink or InfiniBand + CUDA-aware MPI), without
CPU roundtrip.

```fortran
! Current pattern — CPU halo exchange (expensive roundtrip)
!$acc update host(vx, vy)                              ! GPU → CPU
call MPI_Sendrecv(vx_send, ..., vx_recv, ..., MPI_COMM_WORLD, ...)
!$acc update device(vx, vy)                            ! CPU → GPU

! GHEX pattern — GPU-to-GPU halo exchange (Phase 3)
! GHEX handles the exchange on device directly
call ghex_exchange(vx_field, vy_field, context)        ! GPU-to-GPU RDMA
! No CPU roundtrip — next kernels see up-to-date halos on device
```

```python
# Python/Cython side — GHEX interface (Phase 3)
import ghex

ctx     = ghex.context(MPI.COMM_WORLD, thread_safe=False)
pattern = ghex.structured_pattern(ctx, domain, halo_width=1)

# In the time loop — transparent GPU-to-GPU exchange
pattern.exchange(vx_field, vy_field).wait()
update_stress(vx, vy, sigma_xx, ...)
```

**Transformation rules:**

| Source pattern | OpenACC + MPI (Phase 1) | OpenACC + GHEX (Phase 3) |
| -------------- | ----------------------- | ------------------------ |
| `MPI_Sendrecv` after `update_stress` | `!$acc update host` + MPI + `!$acc update device` | `ghex.exchange().wait()` |
| Shared halo arrays | `INTENT(INOUT)` + CPU sync | `INTENT(INOUT)` + GPU sync |
| Compute/comm overlap | ❌ Sequential | ✅ async `exchange()` |

**Expected gain**: 3–10× communication reduction on multi-GPU
InfiniBand clusters.

---

## 11. Fortran I/O → xarray / zarr + DLPack

> ⚠️ **Phase 4 scope — not yet implemented.** Documented for planning.

Classical Fortran I/O (`WRITE`, `OPEN`, PostScript) produces
proprietary binary files or text incompatible with the modern data
science ecosystem.

**Problem**: seismic codes write `.pnm` images and `.dat` seismograms —
unreadable directly by Pandas, xarray, or cloud visualisation tools.

### A — DLPack: zero-copy between Fortran GPU and Python

DLPack is a tensor-sharing protocol between frameworks (CUDA, JAX,
PyTorch, CuPy) **without memory copy**. The Cython wrapper exposes GPU
arrays directly via DLPack:

```python
# Current Phase 1 — CPU copy required
vx_np = np.asfortranarray(vx)              # GPU → CPU → NumPy copy

# Phase 4 target — zero-copy via DLPack
from __dlpack__ import from_dlpack
import cupy as cp

vx_gpu = from_dlpack(seismic_module.vx_dlpack())   # direct DLPack view on GPU memory
vx_jax = jax.dlpack.from_dlpack(vx_gpu)            # JAX array without copy
vx_cp  = cp.from_dlpack(vx_gpu)                    # CuPy array without copy
```

```fortran
! Fortran side — device pointer exposure via iso_c_binding (Phase 4)
function vx_device_ptr(vx) result(ptr) bind(C, name="vx_device_ptr")
  use iso_c_binding
  real(dp), device, intent(in) :: vx(:,:)    ! device attribute (nvfortran)
  type(c_ptr) :: ptr
  ptr = c_loc(vx)
end function
```

### B — xarray / zarr outputs (replaces `WRITE` / PostScript)

```python
# Current — Fortran binary + PostScript files
! WRITE(unit=27,...) image_data_2D    → .pnm files
! WRITE(unit=11,...) sisvx(it, irec)  → .dat files

# Phase 4 — cloud-native xarray/zarr outputs
import xarray as xr, zarr, numpy as np

# Build a geophysical Dataset with coordinates
ds = xr.Dataset(
    {
        "vx":       (["x", "z", "time"], vx_history),      # velocity field
        "sigma_xx": (["x", "z", "time"], stress_history),  # normal stress
        "seismo_x": (["receiver", "time"], sisvx),          # seismograms
    },
    coords={
        "x":    np.arange(NX) * DELTAX,
        "z":    np.arange(NY) * DELTAY,
        "time": np.arange(NSTEP) * DELTAT,
    },
    attrs={"source_x": ISOURCE * DELTAX, "source_z": JSOURCE * DELTAY},
)

# Zarr write — compatible with S3-API object storage (Pangeo, Dask, OpenStack Swift)
ds.to_zarr("s3://seismic-results/run_001.zarr", mode="w")

# Read and visualise directly without conversion
import hvplot.xarray
ds["vx"].isel(time=100).hvplot(x="x", y="z", cmap="seismic")
```

**Transformation rules:**

| Fortran source pattern | Phase 1 (Cython) | Phase 4 (xarray/zarr) |
| ---------------------- | ---------------- | --------------------- |
| `WRITE(unit,...) field(NX,NY)` | In-memory NumPy array | `xr.DataArray` with geo coords |
| `OPEN / WRITE / CLOSE` `.dat` file | Python text file | Zarr dataset on S3-API object storage (e.g. OpenStack Swift) |
| `.pnm` image file (PostScript) | `matplotlib.imshow` | Interactive hvPlot / GeoViews |
| Per-receiver seismogram `.dat` | NumPy array | `xr.DataArray` indexed by receiver |
| Snapshot every N steps | Accumulated 3D array | Streaming Zarr append |

---

## See also

- [Architecture](architecture.md) — the activation-order rationale
  behind the multi-stage pipeline.
- [LLM endpoints](llm-endpoints.md) — sovereign-EU and self-hosted
  inference options.
- Open an [issue](https://github.com/maurinl26/fortranspire/issues) if
  you hit a Fortran pattern not on this page — adding a new pattern is
  usually a one-stage change plus a fixture.
