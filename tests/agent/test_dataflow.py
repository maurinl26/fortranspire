"""Free-variable (use-def) analysis for module-state promotion (issue #5).

The Loki-dependent test skips where the [gpu] extra is absent; the promotion
test is pure and always runs.
"""
import os
import tempfile

import pytest

from fortranspire.agent.nodes_jax.functionalize import _promote_free_state


# ── the pure promotion logic (no Loki) ──────────────────────────────────────

def test_reads_promote_to_inputs_only():
    kernel = {"free_reads": ["RKI", "IRM2"], "free_writes": []}
    inputs, outputs, carried, promoted = _promote_free_state(
        kernel, ["yin"], ["ydot"], [])
    assert inputs == ["yin", "RKI", "IRM2"]      # declared arg first, then state
    assert outputs == ["ydot"]                   # reads never become outputs
    assert promoted == ["RKI", "IRM2"]


def test_writes_promote_to_inout():
    kernel = {"free_reads": [], "free_writes": ["ACC"]}
    inputs, outputs, carried, promoted = _promote_free_state(
        kernel, ["x"], [], [])
    assert inputs == ["x", "ACC"]                # written state is also read
    assert outputs == ["ACC"]                    # …and returned
    assert "ACC" in carried
    assert promoted == ["ACC"]


def test_no_double_promotion_when_already_an_argument():
    kernel = {"free_reads": ["yin"], "free_writes": []}  # already an input
    inputs, outputs, _, promoted = _promote_free_state(
        kernel, ["yin"], ["ydot"], [])
    assert inputs == ["yin"]
    assert promoted == []


def test_empty_free_state_is_a_noop():
    kernel = {}
    inputs, outputs, carried, promoted = _promote_free_state(
        kernel, ["a"], ["b"], [])
    assert (inputs, outputs, promoted) == (["a"], ["b"], [])


# ── the Loki free-variable analysis (skips without the extra) ────────────────

_ROUTINE = """
subroutine rhs(n, y, dydt)
  use network_data          ! provides kmat (read) and nreac (read)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: y(n)
  real(8), intent(out) :: dydt(n)
  integer :: i
  do i = 1, nreac
     dydt(i) = kmat(i) * y(i)
  end do
end subroutine rhs
"""


def _parse_routine(src: str):
    from loki import Sourcefile

    fd, path = tempfile.mkstemp(suffix=".f90")
    try:
        os.write(fd, src.encode())
        os.close(fd)
        return Sourcefile.from_file(path).routines[0]
    finally:
        os.remove(path)


def test_free_symbols_finds_unresolved_module_state():
    pytest.importorskip("loki")
    from fortranspire.agent.dataflow import free_symbols

    routine = _parse_routine(_ROUTINE)
    fs = free_symbols(routine)
    got = {r.lower() for r in fs.reads}
    # kmat and nreac come from `use network_data`, are read, never declared:
    assert {"kmat", "nreac"} <= got
    # declared names and the loop index are bound, never free:
    assert not ({"n", "y", "dydt", "i"} & got)
    assert fs.writes == []


# ── dtype inference & index detection (issue #4) ─────────────────────────────

_GATHER = """
subroutine gather(n, idx, src, dst)
  implicit none
  integer, intent(in) :: n, idx(n)
  real(8), intent(in) :: src(n)
  real(8), intent(out) :: dst(n)
  integer :: i
  do i = 1, n
     dst(i) = src(idx(i))
  end do
end subroutine gather
"""


def test_infer_dtypes_types_bounds_integer_and_payload_real():
    pytest.importorskip("loki")
    from fortranspire.agent.dataflow import infer_dtypes

    dt = infer_dtypes(_parse_routine(_ROUTINE))
    assert dt.get("n") == "integer"           # declared integer
    assert dt.get("nreac") == "integer"       # module scalar used as a loop bound
    assert dt.get("y") == "real"
    assert dt.get("kmat") != "integer"        # numeric payload, never an index


def test_integer_index_args_flags_a_lookup_table():
    pytest.importorskip("loki")
    from fortranspire.agent.dataflow import integer_index_args

    idx = {a.lower() for a in integer_index_args(_parse_routine(_GATHER))}
    assert "idx" in idx                        # integer array used as a subscript
    assert "src" not in idx and "dst" not in idx
