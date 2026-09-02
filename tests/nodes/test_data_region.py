"""Deterministic driver data region — copyin/copyout/copy derived from INTENT.

A wrong data clause is silent GPU corruption; deriving it from INTENT is correct
by construction (and optimal). Replaces an LLM that guessed without the INTENT.
"""
from fortranspire.agent.nodes.data_region import (
    derive_data_clauses,
    extract_kernel_calls,
    insert_data_region,
    render_data_pragma,
)

_KERNELS = {
    "update_v": {"intent_map": {"vx": "INOUT", "sxx": "IN", "n": "IN"},
                 "dimensions": {"vx": ["n"], "sxx": ["n"]}},
    "update_s": {"intent_map": {"sxx": "OUT", "vx": "IN", "n": "IN"},
                 "dimensions": {"sxx": ["n"], "vx": ["n"]}},
}
_DRIVER = (
    "program main\n  integer :: it, n\n  real(8) :: vx(n), sxx(n)\n"
    "  do it = 1, nstep\n"
    "     call update_v(vx, sxx, n)\n"
    "     call update_s(sxx, vx, n)\n"
    "  end do\nend program main\n"
)


def test_in_only_is_copyin_out_only_is_copyout():
    kernels = {"k": {"intent_map": {"a": "IN", "b": "OUT", "s": "IN"},
                     "dimensions": {"a": ["n"], "b": ["n"]}}}
    clauses = derive_data_clauses([("k", ["a", "b", "s"])], kernels)
    assert clauses["copyin"] == ["a"]
    assert clauses["copyout"] == ["b"]
    assert clauses["copy"] == []


def test_mixed_role_array_is_copy_and_scalars_excluded():
    clauses = derive_data_clauses(extract_kernel_calls(_DRIVER, set(_KERNELS)), _KERNELS)
    assert set(clauses["copy"]) == {"sxx", "vx"}   # sxx IN+OUT, vx INOUT+IN → copy
    assert clauses["copyin"] == [] and clauses["copyout"] == []
    # `n` is a scalar (no dimensions) → never in a data clause
    for kind in clauses:
        assert "n" not in clauses[kind]


def test_render_acc_and_omp():
    clauses = {"copyin": ["a"], "copyout": ["b"], "copy": ["c"]}
    op, cl = render_data_pragma(clauses, "acc")
    assert op == "!$acc data copyin(a) copyout(b) copy(c)" and cl == "!$acc end data"
    op2, _ = render_data_pragma(clauses, "omp")
    assert "map(to: a)" in op2 and "map(from: b)" in op2 and "map(tofrom: c)" in op2


def test_region_wraps_the_time_loop():
    op, cl = render_data_pragma(
        derive_data_clauses(extract_kernel_calls(_DRIVER, set(_KERNELS)), _KERNELS), "acc")
    out = insert_data_region(_DRIVER, op, cl, set(_KERNELS))
    lines = [l.strip() for l in out.splitlines()]
    di, doi, edi, eddi = (lines.index("!$acc data copy(sxx, vx)"),
                          lines.index("do it = 1, nstep"),
                          lines.index("!$acc end data"),
                          [i for i, l in enumerate(lines) if l == "end do"][0])
    assert di < doi < eddi < edi                    # data … do … end do … end data


# ── liveness (optimal clauses) ───────────────────────────────────────────────

from fortranspire.agent.nodes.data_region import (  # noqa: E402
    analyse_liveness, array_actuals, clauses_from_liveness, find_region_bounds,
)

_KERNELS_LT = {
    "k1": {"intent_map": {"a": "IN", "tmp": "OUT", "n": "IN"},
           "dimensions": {"a": ["n"], "tmp": ["n"]}},
    "k2": {"intent_map": {"tmp": "IN", "b": "OUT", "n": "IN"},
           "dimensions": {"tmp": ["n"], "b": ["n"]}},
}
_DRIVER_LT = (
    "program main\n  integer :: it, n\n  real(8) :: a(n), b(n), tmp(n)\n"
    "  a = 1.0d0\n"
    "  do it = 1, nstep\n"
    "     call k1(a, tmp, n)\n"
    "     call k2(tmp, b, n)\n"
    "  end do\n"
    "  print *, b\nend program main\n"
)


def _live():
    kn = set(_KERNELS_LT)
    calls = extract_kernel_calls(_DRIVER_LT, kn)
    bounds = find_region_bounds(_DRIVER_LT, kn)
    arrays = array_actuals(calls, _KERNELS_LT)
    return analyse_liveness(_DRIVER_LT, arrays, bounds[0], bounds[1])


def test_loop_local_temporary_becomes_create():
    live = _live()
    assert live["tmp"] == (False, False)          # never touched on the host
    assert live["a"] == (True, False)             # written before the loop
    assert live["b"] == (False, True)             # read after the loop
    clauses = clauses_from_liveness(live)
    assert clauses["create"] == ["tmp"]           # zero transfer — the win
    assert clauses["copyin"] == ["a"]
    assert clauses["copyout"] == ["b"]
    assert clauses["copy"] == []


def test_declarations_do_not_count_as_host_use():
    # `tmp` appears only in its declaration before the loop → still create.
    assert _live()["tmp"] == (False, False)


def test_create_renders_in_acc_and_omp():
    op, _ = render_data_pragma({"copyin": ["a"], "copyout": ["b"],
                                "copy": [], "create": ["tmp"]}, "acc")
    assert "create(tmp)" in op
    op2, _ = render_data_pragma({"copyin": [], "copyout": [], "copy": [],
                                 "create": ["tmp"]}, "omp")
    assert "map(alloc: tmp)" in op2
