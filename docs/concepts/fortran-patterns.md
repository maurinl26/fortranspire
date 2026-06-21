# Fortran patterns handled

Nine recurring Fortran 90 patterns — drawn from production seismic and
atmospheric codes — are documented and covered by fixture kernels in the
test suite.

| #  | Pattern                       | What the pipeline does                                                   |
| -- | ----------------------------- | ------------------------------------------------------------------------ |
| 1  | Missing `INTENT`              | Inferred from data-flow analysis; `INTENT(IN|OUT|INOUT)` added           |
| 2  | `SAVE` (implicit or explicit) | Promoted to an explicit `INTENT(INOUT)` argument of the kernel           |
| 3  | `COMMON` blocks               | Replaced by an explicit argument list; the driver passes them through    |
| 4  | `POINTER` / `TARGET`          | Detected and preserved; flagged when they prevent `PURE`/`ELEMENTAL`     |
| 5  | Implicit typing               | `IMPLICIT NONE` enforced; every variable gets an explicit declaration    |
| 6  | Fixed-form continuation       | Free-form `&` continuation emitted in the rewritten module               |
| 7  | Derived types                 | Carried through unchanged; `iso_c_binding` mapping generated for Cython  |
| 8  | `MODULE PROCEDURE` interfaces | Preserved when the kernel exports a generic name                         |
| 9  | Module-private state          | Surfaced as INTENT(INOUT) at the kernel boundary, never hidden as SAVE   |

These patterns cover the overwhelming majority of legacy scientific
Fortran encountered in seismic imaging and numerical weather prediction
codes. If you hit a pattern not on this list, please open an issue with a
minimal reproducer — adding a new pattern is usually a one-stage change
plus a fixture.

See [`README.md` §8](https://github.com/maurinl26/fortranspire#8--patterns-fortran--règles-de-transformation)
for worked examples of each pattern.
