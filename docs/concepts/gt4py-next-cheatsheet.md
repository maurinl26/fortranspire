# Fortran → gt4py.next — visual cheat-sheet

*Companion to [gt4py-next-patterns](gt4py-next-patterns.md). The full
reasoning is there; this page is the fast, side-by-side lookup — mirroring
the [Fortran patterns](fortran-patterns.md) summary table.*

---

## The one shift, in one line

> Fortran writes the **loop over the grid**. gt4py.next writes the
> **computation at a point**, and the domain is applied from outside.

The `do i = 1, n` is not translated — it is **removed**.

> Prefer the **unstructured mesh** model (Cell/Edge + connectivity,
> `neighbor_sum`) — the mature, high-value path (icon4py, Pace). The
> Cartesian horizontal grid (`Ioff`/`Joff`) is the older, procedural
> alternative. The vertical `Koff` / `scan_operator` is universal.

---

## Correspondence table

| Fortran | gt4py.next | Rule |
| ------- | ---------- | ---- |
| `do i = 1, n … end do` | *(removed — the operator is point-wise)* | the loop is the framework's job |
| `c(i) = a(i) + b(i)` | `return a + b` | whole-field expression |
| `intent(in) a` | `a: gtx.Field[Dims[Cell], float64]` | argument |
| `intent(out) c` | `-> gtx.Field[...]` **return value** | can't mutate an argument |
| `intent(inout) v` | argument **and** return | read and written |
| two outputs | `-> tuple[Field[...], Field[...]]` | tuple return |
| `real(kind=8)` | `float64` | explicit dtype |
| `a(k+1)` / `a(k-1)` | `a(Koff[1])` / `a(Koff[-1])` | vertical `FieldOffset` |
| `sum(a(e2c(e,:)))` | `neighbor_sum(a(E2C), axis=E2CDim)` | **unstructured connectivity (the primary model)** |
| `if (c) x=p else x=q` | `where(c, p, q)` | no `if` in an operator |
| `a(i+1,j)` | `a(Ioff[1])` | Cartesian `FieldOffset` (legacy horizontal grid) |
| `x(k)=f(x(k-1))` | `@scan_operator(axis=K, forward=True)` | vertical recurrence |
| `write` / `read` / `print` | — | **does not map** |
| `save` / module state | — | **does not map** |
| recursion, `goto`, aliasing | — | **does not map** |

---

## Two worked examples

### Point-wise (score 5)

```fortran
subroutine add(a, b, c, n)
  real, intent(in)  :: a(n), b(n)
  real, intent(out) :: c(n)
  integer :: i
  do i = 1, n
     c(i) = a(i) + b(i)
  end do
end subroutine
```
```python
@gtx.field_operator
def add(a: gtx.Field[Dims[Cell], float64],
        b: gtx.Field[Dims[Cell], float64]) -> gtx.Field[Dims[Cell], float64]:
    return a + b
```

### Vertical average — a shift (score 5)

```fortran
do k = 2, nlev
   avg(k) = 0.5 * (t(k) + t(k-1))
end do
```
```python
Koff = gtx.FieldOffset("Koff", source=K, target=(K,))

@gtx.field_operator
def vertical_avg(t: gtx.Field[Dims[K], float64]) -> gtx.Field[Dims[K], float64]:
    return 0.5 * (t + t(Koff[-1]))
```

---

## The `FORT032` portability score

`fortranspire analyze` and `explain` score every routine before you pay for
a port. Run it with no LLM, no token:

```bash
fortranspire analyze --no-toolchain-check src/kernel.f90   # FORT032 notes
fortranspire explain  src/                                 # score in the triage report
```

| Score | Verdict | What it means | Typical signal |
| :---: | ------- | ------------- | -------------- |
| **5** | 🟢 field operator | port it — a clean operator or constant-offset stencil | point-wise, explicit INTENT, literal offsets |
| **3** | 🟡 needs a construct | maps, but not to a plain operator | `if` → `where`, recurrence → `scan_operator`, `SAVE` state |
| **1** | 🟠 unstructured / hard | connectivity, or unportable | indirect indexing `a(idx(i))` |
| **0** | 🔴 does not map | outside the DSL | I/O in the body, no output, recursion, aliasing |

A score-5 routine gets **no** finding — silence means "clean, port it".
Only routines below 5 are annotated, so a good stencil never clutters the
report.

The score's floor is the **same purity verdict** the JAX target uses: a
routine that cannot be a pure function (`blocked`) is 0 for both, a
`threaded` routine (hidden `SAVE`) is capped at 3. The two targets never
disagree about whether a routine can be pure.

---

## Backends (at call time)

```python
from gt4py.next.program_processors.runners import gtfn

add(a, b, out=c, offset_provider={})                 # default (embedded)
add.with_backend(gtfn.run_gtfn)(a, b, out=c, …)      # gtfn_cpu
add.with_backend(gtfn.run_gtfn_gpu)(a, b, out=c, …)  # gtfn_gpu
```

> The API above is verified against gt4py.next 1.2.1
> ([patterns §11](gt4py-next-patterns.md)). One execution detail stays
> version-dependent: running a Cartesian-offset operator needs a
> `CartesianConnectivity` offset provider. Validation type-checks the
> operator (no execution), so it does not depend on that; a driver that
> runs it pins the provider per backend.

---

## See also

- [gt4py-next-patterns](gt4py-next-patterns.md) — the full spec with the reasoning
- [fortran-patterns](fortran-patterns.md) — the OpenACC / JAX rules
- gt4py.next: [QuickstartGuide](https://github.com/GridTools/gt4py/blob/main/docs/user/next/QuickstartGuide.md)
