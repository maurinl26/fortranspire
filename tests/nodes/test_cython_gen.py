"""Deterministic Cython/C wrapper generation (replaces the LLM path).

The value is that the three artifacts are consistent *by construction*: same
C name, same argument count and types across shim / header / .pyx, and the
INTENT(OUT/INOUT) arguments come back.
"""
import re

from fortranspire.agent.nodes.cython_gen import (
    generate_cython,
    render_header,
    render_pyx,
    render_shim,
)


def _kernel(**over):
    k = {
        "routine_name": "axpy",
        "intent_map": {"n": "IN", "a": "IN", "x": "IN", "y": "IN", "z": "OUT"},
        "arg_ctypes": {"n": "int", "a": "double", "x": "double",
                       "y": "double", "z": "double"},
        "dimensions": {"x": ["n"], "y": ["n"], "z": ["n"]},
        "has_io": False,
    }
    k.update(over)
    return k


def test_header_declares_the_bindc_name_with_all_args_as_pointers():
    h = render_header([_kernel()])
    assert "void axpy_c(int* n, double* a, double* x, double* y, double* z);" in h
    assert "extern \"C\"" in h


def test_shim_is_bindc_and_forwards_into_the_kernels_module():
    s = render_shim([_kernel()], "kern_kernels")
    assert "use kern_kernels" in s
    assert 'bind(c, name="axpy_c")' in s
    assert "call axpy(n, a, x, y, z)" in s
    # scalar vs array declarations follow the rank
    assert "integer(c_int), intent(in) :: n" in s
    assert "real(c_double), intent(in) :: x(*)" in s
    assert "real(c_double), intent(out) :: z(*)" in s


def test_pyx_scalars_by_ref_arrays_fortran_contiguous_out_returned():
    p = render_pyx([_kernel()], "kern")
    assert "cdef int n_c = n" in p and "&n_c" in p          # scalar by reference
    assert "np.asfortranarray(x, dtype=np.float64)" in p    # array Fortran layout
    assert "cnp.PyArray_DATA(x_arr)" in p
    assert "return z_arr" in p                               # OUT arg comes back


def test_kind_maps_to_c_type():
    k = _kernel(arg_ctypes={"n": "long", "a": "float", "x": "float",
                            "y": "float", "z": "float"},
                dimensions={"x": ["n"], "y": ["n"], "z": ["n"]})
    h = render_header([k])
    assert "long* n" in h and "float* a" in h                # int8 / real4 honoured


def test_three_artifacts_agree_on_arity():
    art = generate_cython([_kernel()], "kern")
    n_args = 5
    sig = next(l for l in art["header"].splitlines() if l.startswith("void axpy_c"))
    assert sig.count("*") == n_args                           # one pointer per arg
    assert art["pyx"].count("*>") == 3                        # three array casts
    # the shim has one declaration line per argument
    decls = [l for l in art["shim"].splitlines() if "intent(" in l]
    assert len(decls) == n_args


def test_inout_argument_is_both_input_and_returned():
    k = _kernel(intent_map={"n": "IN", "v": "INOUT"},
                arg_ctypes={"n": "int", "v": "double"},
                dimensions={"v": ["n"]})
    p = render_pyx([k], "kern")
    assert "def axpy(n, v):" in p
    assert "return v_arr" in p
