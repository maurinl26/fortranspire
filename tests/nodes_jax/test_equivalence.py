"""Phase 2 numerical equivalence — JAX vs the original Fortran.

The comparison logic is tested with mock callables (always runs); the real
f2py compile path is guarded by the toolchain (gfortran + meson + ninja).
"""
import shutil

import pytest

jax = pytest.importorskip("jax")

from fortranspire.agent.nodes_jax.equivalence import (  # noqa: E402
    check_equivalence,
    compile_fortran,
    _toolchain_env,
)


def _kernel(**over):
    k = {
        "routine_name": "k",
        "inputs": ["n", "t"],
        "outputs": ["avg"],
        "intent_map": {"n": "IN", "t": "IN", "avg": "OUT"},
        "dimensions": {"t": ["n"], "avg": ["n"]},   # symbolic → n and t stay consistent
        "arg_dtypes": {"n": "integer", "t": "real", "avg": "real"},
    }
    k.update(over)
    return k


# ── comparison logic (mock callables, no compiler) ──────────────────────────

def test_agreeing_outputs_pass():
    import numpy as np

    fort = lambda n, t: np.asarray(t) * 2.0          # noqa: E731
    jaxf = lambda n, t: t * 2.0                       # noqa: E731
    report = check_equivalence(_kernel(), jaxf, fort)
    assert report["status"] == "pass"
    assert report["max_abs_err"] < 1e-12


def test_disagreeing_outputs_fail():
    import numpy as np

    fort = lambda n, t: np.asarray(t) * 2.0          # noqa: E731
    jaxf = lambda n, t: t * 3.0                       # noqa: E731 - wrong
    report = check_equivalence(_kernel(), jaxf, fort)
    assert report["status"] == "fail"
    assert report["mismatches"][0]["arg"] == "avg"


def test_shape_mismatch_is_reported():
    import numpy as np

    fort = lambda n, t: np.asarray(t)                # noqa: E731  shape (4,)
    jaxf = lambda n, t: t[:2]                          # noqa: E731  shape (2,)
    report = check_equivalence(_kernel(), jaxf, fort)
    assert report["status"] == "fail"
    assert report["mismatches"][0]["kind"] == "shape"


# ── real f2py compile + compare (needs the Fortran toolchain) ───────────────

_VERTICAL_AVG = """
subroutine vertical_avg(n, t, avg)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: t(n)
  real(8), intent(out) :: avg(n)
  integer :: k
  avg(1) = t(1)
  do k = 2, n
     avg(k) = 0.5d0 * (t(k) + t(k-1))
  end do
end subroutine vertical_avg
"""


def _toolchain_ready() -> bool:
    path = _toolchain_env()["PATH"]
    return all(shutil.which(t, path=path) for t in ("gfortran", "meson", "ninja"))


@pytest.mark.skipif(not _toolchain_ready(), reason="gfortran/meson/ninja not available")
def test_compile_and_match_real_fortran():
    import jax.numpy as jnp

    fn, reason = compile_fortran(_VERTICAL_AVG, "vertical_avg")
    assert fn is not None, reason

    def vertical_avg(n, t):
        return jnp.concatenate([t[:1], 0.5 * (t[1:] + t[:-1])])

    report = check_equivalence(_kernel(), vertical_avg, fn)
    assert report["status"] == "pass"
    assert report["max_abs_err"] < 1e-9
