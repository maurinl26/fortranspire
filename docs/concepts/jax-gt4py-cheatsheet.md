# JAX ↔ gt4py.next — the two functional targets

*Companion to [gt4py-next-patterns](gt4py-next-patterns.md) and
[jax-optimization](jax-optimization.md). Both Phase 2 (JAX) and GT4Py emit
**functional** code from the same analysis; this page sets them side by
side, so you can see where they agree, where they diverge, and — the part
that matters — **which one to target**.*

---

## Why they share a front half

Both targets take the same route out of imperative Fortran:

```
parse → extract → functionalize → ┬─ jax_kernel   → gradcheck        (JAX)
                                  └─ gt4py_kernel → domain_validate  (GT4Py)
```

The `functionalize` node is **the same node**. It turns the INTENT map into
an explicit interface — `INTENT(IN)` an argument, `INTENT(OUT)`/`INTENT(INOUT)`
a return value — and issues one purity verdict (`pure` / `threaded` /
`blocked`). A routine that cannot be a pure function is `blocked` for
**both**: neither a JAX function nor a field operator can do I/O, hold
`SAVE` state, or write invisible globals. So the "does this map at all?"
question has one answer across the two.

They diverge only in the back half — how the point-wise body is written, and
how it is validated.

---

## Side-by-side correspondence

| Concept | JAX (Phase 2) | gt4py.next (GT4Py) |
| ------- | ------------- | ------------------ |
| decorator / form | plain `def` + `jax.jit` | `@gtx.field_operator` |
| grid loop `do i=1,n` | whole-array `jnp` expression (vectorised) | *(removed — operator is point-wise)* |
| `intent(out)` | return value | return value |
| element `a(i)` | `a[i]` / whole-array `a` | `a` (point-wise; no index) |
| shift `a(k+1)` | `jnp.roll` / slice / index | `a(Koff[1])` via `FieldOffset` |
| conditional `if` | `jnp.where(c, p, q)` | `where(c, p, q)` |
| in-place write | `a.at[i].set(v)` | not needed — pure return |
| loop-carried dep | `jax.lax.scan` | `@gtx.scan_operator` |
| reduction (sum) | `jnp.sum` | `jnp.sum` (axis) / `neighbor_sum` (mesh) |
| dtype | `float64` (x64 on) | `float64` |
| dimensions | array **shape** (positional) | named **`Dimension`s** (explicit) |
| **differentiable?** | **yes** — `jax.grad` | **no** — a stencil DSL |
| execution | `jit` → XLA (CPU/GPU/TPU) | `gtfn_cpu` / `gtfn_gpu` |
| validation | gradient vs finite differences | frontend type-check (`.foast_stage`) |

The `where` spelling is identical — the one line you can copy across. Nearly
everything else has the same *shape* and a different surface.

---

## The three that look alike but aren't

### 1. The grid loop

JAX keeps the array and vectorises: the loop becomes a whole-array
expression that still *has* a shape. gt4py.next **removes** the loop — the
operator is the rule at one point, and the iteration domain lives outside
the operator entirely.

```python
# Fortran:  do i=1,n;  c(i) = a(i) + b(i);  end do

# JAX — a whole-array expression, shape (n,)
def add(a, b):
    return a + b

# gt4py.next — a point-wise operator, no shape in the body
@gtx.field_operator
def add(a: gtx.Field[Dims[Cell], float64],
        b: gtx.Field[Dims[Cell], float64]) -> gtx.Field[Dims[Cell], float64]:
    return a + b
```

They read almost the same, but the JAX `a + b` is an operation on arrays of
shape `(n,)`, while the gt4py `a + b` is `a[p] + b[p]` at an unindexed point
`p`. That difference is why gt4py needs **named dimensions** and JAX needs
only shapes.

### 2. The vertical recurrence — the cleanest analogue

A loop-carried dependency (`x(k)` needs `x(k-1)`) is the same finding
(`FORT004`) routed to the two targets' equivalent constructs:

```python
# Fortran:  do k=2,n;  x(k) = x(k-1)*d + s(k);  end do

# JAX
def step(carry, s):
    x = carry * d + s
    return x, x
_, x = jax.lax.scan(step, init, s)

# gt4py.next
@gtx.scan_operator(axis=K, forward=True, init=0.0)
def integrate(carry: float64, s: float64) -> float64:
    return carry * d + s
```

Same carry, same forward/backward direction, same boundary `init`. This is
the correspondence to lead with when explaining either target to someone
who knows the other.

### 3. Reductions

JAX reduces along an **axis** of an array. gt4py.next reduces the same way
*within* a field, **but** its distinctive reduction is over a mesh
connectivity — `neighbor_sum` over a `FieldOffset` — which has no JAX
counterpart, because JAX has no notion of an unstructured neighbour table.

```python
# structured — both the same idea:
jnp.sum(a, axis=-1)                        # JAX
# gt4py: a reduction along a Dimension, similar

# unstructured — gt4py only:
neighbor_sum(cellval(E2C), axis=E2CDim)    # sum over each edge's cells
```

---

## The divergence that decides the target

Everything above is surface. **One difference is not:**

> JAX is **differentiable**. gt4py.next is **not**.

`jax.grad` gives the exact gradient of the kernel; that is the entire reason
Phase 2 exists (adjoints, sensitivity, UQ, ML integration — the `mf6adj` /
TORAX class of need). gt4py.next is a performance DSL for structured and
unstructured **stencils**; it has no autodiff, and its value is
performance-portable execution on CPU/GPU from one source, plus the mesh
model the NWP community (ICON, FVM) is built on.

So the choice is not about which is "better":

| Target | Choose it when you need… |
| ------ | ------------------------ |
| **JAX** | gradients / adjoints, differentiable simulation, ML coupling, autodiff to replace hand-written or Tapenade-generated derivative code |
| **gt4py.next** | a fast structured or **unstructured** stencil, NWP dycore / physics, performance portability without autodiff, the mesh/connectivity model |

`FORT030` (JAX portability) and `FORT032` (GT4Py portability) both surface in
`analyze` / `explain`, so a kernel can be scored for **both** before you
pick. A pure column stencil often scores well for each; a routine whose
value is its derivative points at JAX; a routine on an unstructured mesh
points at gt4py.next.

---

## Validation — two philosophies

Because the two DSLs promise different things, they are validated
differently, and both checks are **blocking**:

- **JAX → `gradcheck`.** The output must be *differentiable and correct*:
  `jax.grad` is compared against central finite differences. A kernel that
  traces but has a wrong gradient is caught. (It cannot see a locally-flat
  transform like `floor` — a documented limit.)
- **gt4py.next → `domain_validate`.** The output must be a *well-typed field
  operator*: forcing `.foast_stage` runs gt4py's own frontend type checker,
  which raises `DSLError` on a scalar returned where a field is declared, an
  illegal construct, undefined names. No execution, no GPU.

The JAX check verifies *meaning* (the gradient); the GT4Py check verifies
*form* (the types). Neither substitutes for numerical equivalence against
the original Fortran, which is the next validation layer for both.

---

## What is identical

Worth stating, because it is where the shared `functionalize` node shows:

- the interface — `INTENT(OUT)` → return value, multiple outputs → a tuple;
- the purity gate — `blocked` for one is `blocked` for the other;
- `where` for a data-dependent branch — same name, same rule (both branches
  evaluated, so guard an unsafe one first);
- the "does not map" set — I/O, `SAVE`, recursion, pointer aliasing;
- explicit `float64`.

---

## See also

- [jax-optimization](jax-optimization.md) — the JAX target in depth
- [gt4py-next-patterns](gt4py-next-patterns.md) — the GT4Py spec
- [gt4py-next-cheatsheet](gt4py-next-cheatsheet.md) — Fortran → gt4py.next
- [fortran-patterns](fortran-patterns.md) — all targets from the Fortran side
