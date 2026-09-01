# Geometry catalogue & software decomposition

*The interactive domain agent (issue
[#88](https://github.com/maurinl26/fortranspire/issues/88)).*

The [typed domain model](gt4py-next-patterns.md) reads the **stencil** from
Fortran — dtypes, axes, offsets, and the **halo** those offsets imply. It
cannot read the **geometry**: whether the horizontal mesh is an octahedral
reduced Gaussian grid, a HEALPix pixelisation, a cubed sphere, or an
icosahedral mesh. That is a modelling choice, which is why the domain agent
is **interactive** — you give the geometry and resolution, the stencil halo
comes from the kernel, and the agent proposes the software decomposition.

## The geometry catalogue

Point-count formulas verified against ECMWF Atlas (Gaussian family) and
standard references; `N` is each family's resolution parameter.

| Geometry | Naming | Points | Structure | Mesh library |
| -------- | ------ | ------ | --------- | ------------ |
| Octahedral reduced Gaussian | `O<N>` — N Gaussian latitudes | `4N² + 36N` | unstructured | ECMWF Atlas |
| Regular Gaussian | `F<N>` — nx=4N, ny=2N | `8N²` | structured | ECMWF Atlas |
| HEALPix | `nside` (power of 2) | `12·nside²` | unstructured | healpy / Atlas |
| Cubed sphere | `C<N>` — N per face edge | `6N²` | block-structured | FV3 / GFDL |
| Icosahedral | `R2B<k>` — root 2, k bisections | `80·4^k` | unstructured | ICON |
| Regular lat-lon | `<dlon>x<dlat>` degrees | `(360/dlon)(180/dlat)` | structured | — |

```bash
fortranspire domain --list                         # the catalogue
fortranspire domain O1280 --ranks 1024             # a decomposition
fortranspire domain nside=1024 --ranks 512 --kernel src/stencil.f90
```

## The decomposition it proposes

Given a geometry + resolution + rank count, and the **halo read from the
kernel's stencil**, the agent proposes:

```
# stencil halo 1 read from src/stencil.f90

# Decomposition — Octahedral reduced Gaussian O1280
  horizontal points     6,599,680
  vertical levels       137
  MPI ranks             1024
  points / rank         ~6,445
  stencil halo          1 row(s)
  memory / rank         ~0.07 GiB
  partitioning          equal-regions (Atlas), one-element halo
  halo exchange         MPI neighbour exchange over the partition boundary
```

The halo is the link between the two features: the [typed domain
model](gt4py-next-patterns.md) computes it from the stencil offsets
(`t(k+1)`/`t(k-1)` → halo 1), and the decomposition sizes the halo exchange
and the per-rank memory from it. The vertical axis stays structured; the
mesh is horizontal-only.

## Why interactive

Geometry is not in the Fortran. A kernel `t(k+1) - t(k-1)` has a halo of 1
whatever the mesh — but whether that mesh is octahedral or HEALPix, and how
many ranks it runs on, is the modeller's call. So the agent asks, rather
than guessing, and the catalogue is the vocabulary it asks in.

## See also

- [gt4py-next-patterns](gt4py-next-patterns.md) — the typed domain model and
  the halo it computes
- The GT4Py driver/halo work (issue #82) — where the halo also feeds the
  generated mesh driver
