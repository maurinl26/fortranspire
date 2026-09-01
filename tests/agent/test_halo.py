"""FORT033 stencil-halo scan — must not fire on integer index tables (#6).

`IRM2(NRK, NP+3, NCS)` in a chemistry mechanism is index arithmetic into a
lookup table, not a spatial field stencil, so the `+3` must not be read as a
halo. A real field stencil (`t(k-1)`) still must.
"""
from fortranspire.agent.analyze import _gt4py_halo


def test_real_field_stencil_still_reports_a_halo():
    assert _gt4py_halo("avg(k) = 0.5*(t(k+2) + t(k-1))") == 2


def test_integer_index_table_shift_is_not_a_halo():
    chem = "isp1 = irm2(nrk, np+3, ncs)"
    assert _gt4py_halo(chem) == 3                              # no type info: unchanged
    assert _gt4py_halo(chem, frozenset({"irm2"})) == 0        # irm2 integer → skipped


def test_a_real_stencil_beside_an_integer_table_is_still_seen():
    src = "y(i) = a(i-1) + coef(idx(i+4))"
    # `a` is a real field (i-1 → halo 1); `idx` is an integer table (i+4 ignored).
    assert _gt4py_halo(src, frozenset({"idx"})) == 1
