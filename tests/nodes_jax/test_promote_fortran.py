"""Promoted-Fortran generator for the whole-program equivalence twin (#99)."""
from fortranspire.agent.nodes_jax.promote_fortran import generate_equivalence_fortran


def _kernel(**over):
    k = {
        "routine_name": "scale_it",
        "fortran_code": (
            "SUBROUTINE scale_it(x, y)\n"
            "  USE netdata\n"
            "  IMPLICIT NONE\n"
            "  REAL(KIND=8), INTENT(IN) :: x(:)\n"
            "  REAL(KIND=8), INTENT(OUT) :: y(:)\n"
            "  INTEGER :: i\n"
            "  DO i=1,nn\n"
            "    y(i) = coef(i)*x(i)\n"
            "  END DO\n"
            "END SUBROUTINE scale_it\n"
        ),
        "free_reads": ["coef", "nn"],
        "free_writes": [],
        "resolved": {"coef": {"dtype": "real", "rank": 1},
                     "nn": {"dtype": "integer", "rank": 0}},
    }
    k.update(over)
    return k


def test_promotes_module_state_and_drops_use():
    code, dim = generate_equivalence_fortran(_kernel())
    assert dim == "nfs"
    assert "USE netdata" not in code and "use netdata" not in code
    # promoted symbols appended to the argument list:
    assert "scale_it(x, y, coef, nn, nfs)" in code
    # declared with resolved types:
    assert "INTENT(IN) :: coef(:)" in code
    assert "INTEGER, INTENT(IN) :: nn" in code
    assert "INTEGER, INTENT(IN) :: nfs" in code
    # the OUT array made explicit in nfs (assumed-shape OUT is unbuildable):
    assert "y(nfs)" in code


def test_refuses_self_contained_routine():
    code, reason = generate_equivalence_fortran(_kernel(free_reads=[], free_writes=[]))
    assert code is None and "self-contained" in reason


def test_refuses_unresolved_symbol():
    code, reason = generate_equivalence_fortran(_kernel(resolved={"coef": {"dtype": "real", "rank": 1}}))
    assert code is None and "unresolved" in reason
    assert "nn" in reason


def test_refuses_external_call():
    k = _kernel()
    k["fortran_code"] = k["fortran_code"].replace(
        "  INTEGER :: i\n", "  INTEGER :: i\n  CALL special_rates(x)\n")
    code, reason = generate_equivalence_fortran(k)
    assert code is None
    assert "external" in reason and "special_rates" in reason.lower()
