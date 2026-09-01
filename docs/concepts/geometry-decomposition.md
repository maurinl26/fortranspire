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

## Spectral truncation is an alias, not a geometry

`TCo1279`, `TL1279` are **spectral truncation** labels — they show the
spectral resolution, and they resolve to a Gaussian grid, which is what
imposes the decomposition. So the agent treats them as aliases, not as a
catalogue family:

- `TCo<T>` (cubic octahedral) → `O<T+1>` — verified against ECMWF
  (`TCo1279` = `O1280`);
- `TL<T>` (linear) → `O<(T+1)/2>` — the linear grid ≈ reduced Gaussian
  `N<(T+1)/2>` (`TL1279` ≈ `N640`).

`fortranspire domain TCo1279 --ranks 1024` works and notes the resolution.

## The grid imposes the decomposition

The dependency runs one way: **software-defined by the grid**. The rank
count is not a free knob — the grid topology decides which decompositions
exist, and a requested count is snapped to the nearest one the grid allows.

- A **cubed sphere** has 6 faces tiled `p×p`, so ranks are `6·p²`. Ask for
  1000 on C768 and the agent returns **1014** (`6 faces × 13×13`), telling
  you it snapped.
- **HEALPix nested** imposes the `12·4^d` hierarchy that keeps neighbours
  local. Ask for 500 and it returns **768** (`12·4³`).
- **Octahedral** (Atlas equal-regions) and **icosahedral** (METIS) are
  flexible partitioners — they take (almost) any count — but it is still the
  grid's partitioner that allows it, not a blind `total / ranks`.

The per-rank points and memory are computed from the grid-allowed count, not
the wish — so the estimate matches what would actually run.

## Interactive from an agent (MCP)

The domain agent is exposed as two deterministic MCP tools, so an LLM agent
(Claude Code, mistral-vibe, Le Chat) can drive the conversation:

- `domain_geometries` — the catalogue, for the agent to present.
- `domain_decomposition(resolution, n_ranks, source, …)` — the proposal, and
  the **nudge**: with no geometry it returns the catalogue and asks the user
  to choose; with a geometry but no ranks it reads the stencil halo and asks
  for the ranks; with everything it proposes.

The flow the agent runs:

1. read the kernel's halo (pass `source`);
2. **ask the user** which geometry + resolution and how many ranks — these
   are not in the Fortran;
3. propose.

Both tools are deterministic — no LLM, no token — so they are safe on the
public (Le Chat) surface too.

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
