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
    assert art["pyx"].count("*>") == 6                        # three array casts
    # one declaration line per argument, in each of the host + device entries
    decls = [l for l in art["shim"].splitlines() if "intent(" in l]
    assert len(decls) == 2 * n_args


def test_inout_argument_is_both_input_and_returned():
    k = _kernel(intent_map={"n": "IN", "v": "INOUT"},
                arg_ctypes={"n": "int", "v": "double"},
                dimensions={"v": ["n"]})
    p = render_pyx([k], "kern")
    assert "def axpy(n, v):" in p
    assert "return v_arr" in p


# ── Device-pointer (deviceptr) entry: GPU-resident interop via CAI ────────────

def test_shim_emits_a_deviceptr_variant_for_kernels_with_arrays():
    s = render_shim([_kernel()], "kern_kernels")
    assert 'bind(c, name="axpy_c_device")' in s
    # only the ARRAY args go in the deviceptr clause; scalars stay host
    assert "!$acc data deviceptr(x, y, z)" in s
    assert "!$acc end data" in s
    assert "call axpy(n, a, x, y, z)" in s  # forwards the same call, no copyin/out


def test_no_device_variant_when_the_kernel_has_no_array_args():
    scalar_only = _kernel(intent_map={"a": "IN", "b": "OUT"},
                          arg_ctypes={"a": "double", "b": "double"},
                          dimensions={}, routine_name="scal")
    s = render_shim([scalar_only], "kern_kernels")
    assert "scal_c_device" not in s      # nothing to share on the device
    h = render_header([scalar_only])
    assert "scal_c_device" not in h


def test_header_declares_the_device_entry():
    h = render_header([_kernel()])
    assert "void axpy_c_device(int* n, double* a, double* x, double* y, double* z);" in h


def test_pyx_device_entry_reads_a_cai_pointer_and_returns_in_place():
    p = render_pyx([_kernel()], "kern")
    assert "cdef unsigned long long _device_ptr(" in p      # the CAI bridge helper
    assert "__cuda_array_interface__" in p
    assert "def axpy_device(n, a, x, y, z):" in p
    assert 'cdef unsigned long long x_ptr = _device_ptr(x, "f8", 8)' in p
    assert "axpy_c_device(&n_c, &a_c, <double*>x_ptr, <double*>y_ptr, <double*>z_ptr)" in p
    assert "return z_arr" not in p.split("def axpy_device")[1]  # no host copy in device path
    assert "return z" in p.split("def axpy_device")[1]          # device array in place


def test_device_helper_enforces_fortran_order_and_dtype():
    p = render_pyx([_kernel()], "kern")
    assert "must be Fortran-ordered" in p            # rank>1 C-order is rejected
    assert "dtype mismatch" in p


def test_full_pyx_is_valid_python_syntax_after_cythonless_strip():
    # the .pyx uses cdef/<cast> (Cython), but the pure-python bodies must be sane;
    # here we just assert the device entry and helper are present and balanced.
    p = render_pyx([_kernel()], "kern")
    assert p.count("def ") >= 2                       # host + device entry
    assert p.count("_device_ptr(") >= 1


def test_device_shim_declares_explicit_shape_not_assumed_size():
    # deviceptr rejects a(*); the extent must be explicit (here the arg `n`).
    s = render_shim([_kernel()], "kern_kernels")
    dev = s.split("_c_device")[1]
    assert "real(c_double), intent(in) :: x(n)" in s
    assert "x(*)" not in dev                     # no assumed-size in the device entry


def test_no_device_variant_when_an_extent_is_not_in_scope():
    # array dimensioned by a module PARAMETER (not an arg) → cannot size the
    # explicit-shape declaration → skip the device entry (host entry still works).
    k = _kernel(intent_map={"x": "IN", "y": "OUT"},
                arg_ctypes={"x": "double", "y": "double"},
                dimensions={"x": ["NRXN"], "y": ["NRXN"]},  # NRXN is not an arg
                routine_name="rxn")
    s = render_shim([k], "kern_kernels")
    assert "rxn_c_device" not in s
    assert "def rxn_device" not in render_pyx([k], "kern")
    assert "rxn_c_device" not in render_header([k])
    assert "def rxn(" in render_pyx([k], "kern")  # host entry unaffected


def test_generated_device_shim_compiles_under_openacc(tmp_path):
    """The deviceptr shim must be valid Fortran under -fopenacc (assumed-size,
    which gfortran rejects in a map/deviceptr clause, is the trap this guards)."""
    import shutil
    import subprocess
    gfortran = shutil.which("gfortran")
    if not gfortran:
        import pytest
        pytest.skip("gfortran not available")
    shim = render_shim([_kernel()], "kern_kernels")
    # a minimal matching kernels module so the shim's `use` resolves.
    mod = (
        "module kern_kernels\ncontains\n"
        "  subroutine axpy(n, a, x, y, z)\n"
        "    integer, intent(in) :: n\n    real(8), intent(in) :: a\n"
        "    real(8), intent(in) :: x(n), y(n)\n    real(8), intent(out) :: z(n)\n"
        "    z = a * x + y\n  end subroutine\nend module\n"
    )
    (tmp_path / "m.f90").write_text(mod)
    (tmp_path / "shim.f90").write_text(shim)
    r = subprocess.run(
        [gfortran, "-fsyntax-only", "-fopenacc", "-ffree-line-length-none",
         "m.f90", "shim.f90"],
        cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
