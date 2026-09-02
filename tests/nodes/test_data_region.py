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
