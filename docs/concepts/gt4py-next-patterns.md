# Fortran → gt4py.next — pattern correspondences

*Specification for the GT4Py transformation target (issue
[#42](https://github.com/maurinl26/fortranspire/issues/42)).*

This page is the **teaching reference** for the third transformation
target: legacy Fortran → [gt4py.next](https://github.com/GridTools/gt4py),
the functional frontend of GridTools for Python. It sits beside the
OpenACC (Phase 1) and JAX (Phase 2) rules in
[fortran-patterns](fortran-patterns.md), and it is written to be read
before any code is generated — a contributor, a reviewer, or an NWP
engineer should be able to follow each Fortran construct to its
gt4py.next counterpart and understand *why* the mapping is what it is.

> **Scope note.** GT4Py has two frontends. The older **gtscript**
> (`@gtscript.stencil`, `with computation(FORWARD)`, `interval(...)`,
> `Field[IN, FloatField]`) is imperative and Cartesian-only. **gt4py.next**
> (`@field_operator`, `@program`, `Field[Dims[...], dtype]`,
> `FieldOffset`) is functional and handles both Cartesian and unstructured
> grids. **This target is gt4py.next.** Where the original #42 scope refers
> to gtscript constructs, this document supersedes it.

---

## 1. Why gt4py.next is the right shape for this pipeline

gt4py.next is **functional**, and that is the whole reason it fits.

A gt4py.next *field operator* is a pure function from fields to fields. It
does not loop over the grid — it describes the computation **at one point**,
and the framework applies it across the iteration domain. There are no
in-place writes, no hidden state, no `i, j, k` loop indices in the body.

That is the same functional contract our **Phase 2 (JAX)** target already
enforces. Concretely, the deterministic `functionalize` node built for
Phase 2 — which turns an imperative subroutine's INTENT map into an
explicit "these are inputs, these are returned values" interface and issues
a purity verdict — is **reused unchanged** for gt4py.next. Only the
emission node and the portability rules differ.

```
Phase 1 (OpenACC):  parse → extract → pure_elemental → openacc → cython → validate
Phase 2 (JAX):      parse → extract → functionalize → jax_kernel   → gradcheck
GT4Py (this):       parse → extract → functionalize → gt4py_kernel → type_check
                                      └── shared ──┘   └── new ──┘
```

The reuse is not a convenience; it is the argument. The same purity
analysis that says "this routine can become a JAX function" is exactly what
says "this routine can become a field operator." A routine that fails it
(I/O in the body, no observable output, hidden `SAVE` state) cannot become
either, and the pipeline reports it rather than emitting something that
compiles but is wrong.

---

## 2. The one mental-model shift

Everything below follows from a single change of stance.

> **Fortran describes the loop over the grid. gt4py.next describes the
> computation at a point, and the domain is applied from outside.**

```fortran
! Fortran: the routine OWNS the iteration.
subroutine add(a, b, c, n)
  real, intent(in)    :: a(n), b(n)
  real, intent(out)   :: c(n)
  integer, intent(in) :: n
  integer :: i
  do i = 1, n
     c(i) = a(i) + b(i)      ! explicit index, explicit loop
  end do
end subroutine
```

```python
# gt4py.next: the field operator is the point-wise rule. No loop, no index.
import gt4py.next as gtx
from gt4py.next import Dims, float64

Cell = gtx.Dimension("Cell")

@gtx.field_operator
def add(a: gtx.Field[Dims[Cell], float64],
        b: gtx.Field[Dims[Cell], float64]) -> gtx.Field[Dims[Cell], float64]:
    return a + b          # 'a + b' means 'a[p] + b[p]' at every point p

# The domain lives outside, where the operator is called:
add(a, b, out=c, offset_provider={})
```

The `do i = 1, n` disappears. It is not translated — it is *removed*,
because the iteration is the framework's job. This is the pattern that most
often confuses a Fortran reader, so it leads every example that follows.

---

## 3. Pattern correspondence table

| # | Fortran | gt4py.next | Notes |
| - | ------- | ---------- | ----- |
| 1 | `do i = 1, n` grid loop | *(removed)* — the field operator is point-wise | §2 |
| 2 | `intent(in) a` | field argument `a: Field[...]` | §4 |
| 3 | `intent(out) / intent(inout)` | **return value** of the operator | §4 |
| 4 | `c(i) = a(i) + b(i)` | `return a + b` | §2 |
| 5 | `a(k+1)`, `a(k-1)` vertical shift | `a(Koff[1])`, `a(Koff[-1])` | §5 |
| 6 | `a(i+1,j)` structured stencil | `a(Ioff[1])` via Cartesian `FieldOffset` | §5 |
| 7 | `if (cond) x = p else x = q` | `where(cond, p, q)` | §6 |
| 8 | `sum` over neighbours (unstructured) | `neighbor_sum(a(E2C), axis=E2CDim)` | §7 |
| 9 | vertical recurrence `x(k)=f(x(k-1))` | `@scan_operator` | §8 |
| 10 | `real, dimension(:,:)` ranks | `Field[Dims[I, J], ...]` explicit dims | §9 |
| 11 | `real(kind=8)` | `float64` (explicit dtype) | §9 |
| — | I/O, `SAVE`, recursion, pointer aliasing | **does not map** — reported, not emitted | §10 |

---

## 4. INTENT → arguments and returns (shared with Phase 2)

gt4py.next cannot mutate an argument, so the mapping is identical to the
JAX one: `INTENT(IN)` becomes an argument, `INTENT(OUT)`/`INTENT(INOUT)`
becomes a **returned** field. This is exactly what the `functionalize`
node already computes from the Loki INTENT map.

```fortran
subroutine diffuse(phi, phi_new, kdiff, n)
  real, intent(in)    :: phi(n), kdiff
  real, intent(out)   :: phi_new(n)
  ...
end subroutine
```

```python
@gtx.field_operator
def diffuse(phi: gtx.Field[Dims[Cell], float64],
            kdiff: float64) -> gtx.Field[Dims[Cell], float64]:
    #                    ^ a scalar INTENT(IN) stays a scalar
    ...
    return phi_new       # INTENT(OUT) becomes the return value
```

Multiple outputs return a tuple, mirroring the JAX rule:

```fortran
real, intent(out) :: u_new(n), v_new(n)
```
```python
-> tuple[gtx.Field[Dims[Cell], float64], gtx.Field[Dims[Cell], float64]]
```

`INTENT(INOUT)` is read *and* written: it appears as an argument **and** in
the returned tuple. A routine with no `INTENT(OUT)`/`INTENT(INOUT)` writes
global state — it cannot be a field operator, and `functionalize` already
flags it `blocked`.

---

## 5. Neighbour access — the heart of a stencil

A Fortran stencil reads shifted array elements: `a(k+1)`, `a(i-1,j)`. In
gt4py.next you do **not** index with an integer. You declare a
`FieldOffset` once, then shift the field by it.

### Vertical shift (the common NWP case: column physics on `K`)

```fortran
! Fortran: vertical average onto layer k
do k = 2, nlev
   avg(k) = 0.5 * (t(k) + t(k-1))
end do
```

```python
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)   # verified (§11)
Koff = gtx.FieldOffset("Koff", source=K, target=(K,))    # a shift along K itself

@gtx.field_operator
def vertical_avg(t: gtx.Field[Dims[K], float64]) -> gtx.Field[Dims[K], float64]:
    return 0.5 * (t + t(Koff[-1]))     # t(Koff[-1]) is 't at k-1'
#                        ^^^^^^^^^ no index arithmetic — a declared shift
```

`t(Koff[-1])` reads the value one layer below. `Koff[+1]` reads one above.
At call time the shift needs an offset provider; on the embedded backend that is a `gtx.CartesianConnectivity` (§11). Validation type-checks the operator rather than running it, so it does not depend on this — a driver that executes the operator does.

### Horizontal structured stencil (Cartesian `I`, `J`)

```fortran
! Fortran: 2-D five-point Laplacian
lap(i,j) = -4*u(i,j) + u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1)
```

```python
I = gtx.Dimension("I")
J = gtx.Dimension("J")
Ioff = gtx.FieldOffset("Ioff", source=I, target=(I,))
Joff = gtx.FieldOffset("Joff", source=J, target=(J,))

@gtx.field_operator
def laplacian(u: gtx.Field[Dims[I, J], float64]) -> gtx.Field[Dims[I, J], float64]:
    return (-4.0 * u
            + u(Ioff[1]) + u(Ioff[-1])
            + u(Joff[1]) + u(Joff[-1]))
```

Each Fortran offset `(i±1, j±1)` becomes a `FieldOffset` shift. The
constant-offset structure is what makes a routine a *stencil* — and what
makes it portable. A non-constant offset (`a(idx(i))`, indirect addressing)
is **not** a Cartesian shift; it is either an unstructured connectivity
(§7) or unportable (§10).

> **Note (§11):** the vertical `Koff` form is verified against gt4py.next
> 1.2.1. The horizontal `Ioff`/`Joff` form uses the identical `FieldOffset`
> mechanism (`source`/`target` on the same dimension); confirm a Cartesian
> *execution* example per backend when a driver runs it — the type-check
> does not execute, so emission is not blocked on it.

---

## 6. Conditionals → `where`, never `if`

A data-dependent branch in a field operator is a `where`, not a Python
`if`. This is the same rule as JAX (`jnp.where`), for the same reason: the
operator is traced, so a branch on a field value must be expressed as a
field-level select.

```fortran
! Fortran: clip negatives to zero
do i = 1, n
   if (q(i) < 0.0) then
      q_clip(i) = 0.0
   else
      q_clip(i) = q(i)
   end if
end do
```

```python
from gt4py.next import where

@gtx.field_operator
def clip(q: gtx.Field[Dims[Cell], float64]) -> gtx.Field[Dims[Cell], float64]:
    return where(q < 0.0, 0.0, q)
```

`where(mask, a, b)` selects `a` where `mask` is true, `b` elsewhere, and
supports tuples: `where(mask, (a1, a2), (b1, b2))`. As with JAX, **both
branches are evaluated**, so a branch that guards an unsafe operation
(a `sqrt` of a possibly-negative value, a division by a possibly-zero
denominator) has to be made safe first — the guard rules from the JAX work
(`jax_smooth`-style) carry over conceptually.

A branch on a *loop index* (`if (k == 1)`, a boundary layer) is different:
that is a **vertical interval**, handled by the domain or by a
`scan_operator`, not by `where`.

---

## 7. Unstructured neighbours → connectivity + `neighbor_sum`

This is where gt4py.next goes beyond OpenACC and beyond a Cartesian
stencil, and it is the reason the NWP community (ICON, FVM) uses it. On an
unstructured mesh, a cell's neighbours are given by a **connectivity
table**, not by an index offset.

```fortran
! Fortran: sum the cells adjacent to each edge (edge-to-cell table)
do e = 1, nedges
   flux(e) = 0.0
   do c = 1, 2                      ! each edge touches 2 cells
      flux(e) = flux(e) + cellval(e2c(e, c))
   end do
end do
```

```python
Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)   # the 'local' neighbour axis
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))

@gtx.field_operator
def edge_flux(cellval: gtx.Field[Dims[Cell], float64]) -> gtx.Field[Dims[Edge], float64]:
    return neighbor_sum(cellval(E2C), axis=E2CDim)
```

The connectivity table is supplied as an offset provider at call time:

```python
E2C_provider = gtx.as_connectivity([Edge, E2CDim], codomain=Cell,
                                    data=edge_to_cell_table, skip_value=-1)
edge_flux(cellval, out=flux, offset_provider={"E2C": E2C_provider})
```

Detecting this from Fortran is harder than a constant stencil: the pipeline
must recognise the **indirection** `e2c(e, c)` as a connectivity access
rather than plain array indexing. Structured stencils (§5) are the first
milestone; unstructured connectivity is a second, and the detection rubric
(§9) scores them separately.

---

## 8. Vertical recurrence → `scan_operator`

A loop-carried dependency along the vertical — `x(k)` depends on `x(k-1)` —
is a recurrence, not a stencil. gt4py.next expresses it with a
`@scan_operator`, the direct analogue of the `lax.scan` mapping in Phase 2.
This is the same `FORT004` (loop-carried dependency) finding, routed to a
different construct per target.

```fortran
! Fortran: a downward vertical integral (each layer needs the one above)
column(1) = source(1)
do k = 2, nlev
   column(k) = column(k-1) * decay + source(k)
end do
```

```python
@gtx.scan_operator(axis=K, forward=True, init=0.0)
def integrate(carry: float64, source: float64) -> float64:
    return carry * decay + source
```

`forward=True` corresponds to `do k = 1, nlev` (top-down); `forward=False`
to `do k = nlev, 1, -1`. The `init` is the boundary value. The parser
already detects the loop-carried dependency; the target decides whether it
becomes `lax.scan` (JAX) or `scan_operator` (gt4py.next).

> **Note (§11):** `gtx.scan_operator` is confirmed present in 1.2.1; its
> `axis` / `forward` / `init` parameters and carry typing are pinned there.

---

## 9. Dimensions, dtypes, and the portability score

### Structured vs unstructured

gt4py.next needs the **named dimensions** of every field. Fortran gives
them only positionally (`a(:,:,:)`), so the pipeline must infer them:

- a 1-D column kernel → `Dims[K]`
- a 2-D horizontal kernel → `Dims[I, J]` (Cartesian) or `Dims[Cell]`
  (unstructured)
- a 3-D kernel → `Dims[I, J, K]` or `Dims[Cell, K]`

The NWP idiom helps: physics parameterisations are **column** kernels
(`Dims[Cell, K]` with all real work on `K`), which is the strongest fit and
the first target. Dynamical cores are horizontal stencils or unstructured.

### dtype

`real(kind=8)` → `float64`, `real(kind=4)` → `float32`, `integer` →
`int32`/`int64`. The explicit `KIND` (the `FORT007` pattern) is what makes
this unambiguous; a `real` with no `KIND` is flagged, as it already is for
the other targets.

### `FORT032` — GT4Py portability score

Mirroring `FORT030` (JAX portability) and `FORT031` (non-smooth
constructs), `analyze` and `explain` gain a per-routine GT4Py verdict, so a
user sees the fit **before** paying for an LLM port:

| Score | Meaning | Signals |
| ----- | ------- | ------- |
| **5 — field operator** | pure point-wise or constant-offset stencil | no I/O, no `SAVE`, explicit `INTENT`, constant loop bounds, offsets are literals |
| **3 — needs a construct** | maps, but not to a plain operator | data-dependent branch (`where`), loop-carried vertical dep (`scan_operator`), interval boundaries |
| **1 — unstructured / hard** | indirect addressing, connectivity | `a(idx(i))` indirection, ragged neighbour loops |
| **0 — does not map** | outside the DSL | I/O in body, recursion, pointer aliasing, `SAVE` state — see §10 |

The score reuses the `functionalize` purity verdict as its floor: a
`blocked` routine is at most `0`, a `threaded` routine at most `3`.

---

## 10. What does not map — and must be reported, not emitted

gt4py.next is a restricted DSL. The following have **no** gt4py.next
counterpart, and the pipeline must refuse them with a reason rather than
emit something that silently does the wrong thing — the same discipline as
the `blocked` verdict in Phase 2:

- **Fortran I/O in the body** (`WRITE`, `READ`, `PRINT`) — a side effect;
  `FORT001`.
- **`SAVE` / module state** — hidden persistence; a field operator is pure.
- **Recursion** — not expressible.
- **Pointer aliasing** — the framework assumes fields do not alias.
- **Arbitrary control flow** — `GOTO`, computed jumps, `while` with a
  data-dependent trip count.
- **Non-constant, non-connectivity indexing** — `a(perm(i))` where `perm`
  is arbitrary.

Each is already a finding the analyzer produces for the other targets;
`FORT032` simply lowers the score to 0 when one is present, and the
`gt4py_kernel` node skips the routine with a message.

---

## 11. Verified against gt4py.next 1.2.1

The API surface below was confirmed by installing gt4py.next 1.2.1 and
running each construct, not read from memory:

- **Imports & decorators** — `import gt4py.next as gtx`;
  `@gtx.field_operator`, `@gtx.scan_operator`, `@gtx.program`. ✓
- **`DimensionKind`** — `HORIZONTAL`, `LOCAL`, `VERTICAL`. ✓
- **`FieldOffset`** — `gtx.FieldOffset("Koff", source=K, target=(K,))` for
  a shift; `field(Koff[-1])` reads one step back. ✓
- **`where`, `neighbor_sum`** — `from gt4py.next import where,
  neighbor_sum`; `where(mask, a, b)` runs point-wise (verified end to end);
  `neighbor_sum(a(E2C), axis=E2CDim)`. ✓
- **Backends** — `run_gtfn`, `run_gtfn_gpu` under
  `gt4py.next.program_processors.runners.gtfn`. ✓

Two facts learned by running it, both of which shape the pipeline:

1. **A field operator must live in a real `.py` file.** gt4py reads the
   operator's source with `inspect.getsourcelines`, so a body built by
   `exec` (the way the JAX validator loads its output) raises "could not
   get source code". The GT4Py validator writes the emitted module to a
   file and imports it.
2. **Validation is a frontend type-check, not execution.** Forcing an
   operator's `.foast_stage` runs gt4py's own type checker — it verifies
   the body against the annotated signature and raises `DSLError` on a
   mismatch (a scalar returned where a field is declared, an illegal
   construct) — with no offset providers and no backend compile. That is
   what `type_check` does, and it is stronger than the JAX tracing
   check.

The type-check validates the **operator**, not the geometric **domain**.
The `domain=` a program writes, the offset providers, and halo/boundary
handling live in the *driver* and are a separate, harder problem — the one
icon4py and Pace frame explicitly. That is tracked in issue #82; this
target currently emits and type-checks the operator, and stops there.

One nuance still open, and it is an *execution* detail, not a
correctness one: running a Cartesian-offset operator on the embedded
backend needs a `gtx.CartesianConnectivity` offset provider whose exact
form varies by version. Validation type-checks rather than executes, so it
does not depend on this; a driver that runs the operator does, and pins it
per backend.

## 12. Where this plugs into the pipeline

| Piece | Status | Reuse |
| ----- | ------ | ----- |
| `parser`, `extractor` | exists | shared with Phase 1 & 2 |
| `functionalize` (INTENT → returns, purity) | exists (Phase 2) | **shared unchanged** |
| `FORT032` detection in `analyze`/`explain` | to build | mirrors `FORT030`/`FORT031` |
| `gt4py_kernel` emission node | to build | mirrors `jax_kernel` |
| `type_check` (frontend type-check) | to build | mirrors `gradcheck`'s structure |
| `prompts/gt4py_kernel/{en,fr}/v1.md` | to build | mirrors `jax_kernel` prompts |
| `fortranspire gt4py` CLI verb | to build | mirrors `translate` |

The next concrete step after this specification is the **detection**
(`FORT032`): it is deterministic, needs no LLM, and — like the port-cost
estimate — lets a user see the GT4Py fit for their kernel before any
generation. The emission node and prompts follow.

---

## See also

- [fortran-patterns](fortran-patterns.md) — the OpenACC and JAX rules
- [jax-optimization](jax-optimization.md) — the functional target this
  reuses
- [architecture](architecture.md) — the pipeline graph
- gt4py.next upstream:
  [QuickstartGuide](https://github.com/GridTools/gt4py/blob/main/docs/user/next/QuickstartGuide.md)
