"""Schema-level tests for the structured outputs added in #4.

No LLM is fired here — just confirms the Pydantic shapes accept what the
prompt promises to deliver and reject malformed payloads.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fortranspire.agent.schemas import (
    CythonHeaderOutput,
    CythonPyxOutput,
    DocRoutineOutput,
    ExtractorOutput,
    OpenACCDriverOutput,
    OpenACCKernelOutput,
)


def test_doc_routine_minimum():
    out = DocRoutineOutput(
        short_summary="Updates the horizontal velocity field on the interior grid",
        detailed=(
            "Reads sigma_xx at (i, j) and (i-1, j). Writes vx in place. "
            "INTENT(INOUT) on vx is required because the time loop accumulates updates."
        ),
        params={"vx": "horizontal velocity (m/s), modified in place"},
    )
    assert out.short_summary.startswith("Updates")
    assert out.params["vx"].startswith("horizontal")


def test_doc_routine_missing_required_raises():
    with pytest.raises(ValidationError):
        DocRoutineOutput(short_summary="x")   # 'detailed' missing


def test_doc_routine_short_summary_length_capped():
    long = "x" * 500
    with pytest.raises(ValidationError):
        DocRoutineOutput(short_summary=long, detailed="d")


def test_doc_routine_params_optional():
    out = DocRoutineOutput(short_summary="s", detailed="d")
    assert out.params == {}


def test_extractor_shape():
    out = ExtractorOutput(
        module_fortran="module m\ncontains\nend module",
        driver_fortran="program p\nend program",
    )
    assert "module m" in out.module_fortran
    assert "program p" in out.driver_fortran


def test_openacc_kernel_shape():
    out = OpenACCKernelOutput(annotated_fortran="subroutine k\n!$acc parallel loop\nend subroutine")
    assert "!$acc" in out.annotated_fortran


def test_openacc_driver_shape():
    out = OpenACCDriverOutput(annotated_fortran="!$acc data\nprogram\nend program\n!$acc end data")
    assert "data" in out.annotated_fortran


def test_cython_pyx_shape():
    out = CythonPyxOutput(pyx_code="cdef extern from 'kernel_c.h': pass")
    assert "cdef extern" in out.pyx_code


def test_cython_header_shape():
    out = CythonHeaderOutput(header_code="#ifndef KERNEL_C_H\n#define KERNEL_C_H\n#endif")
    assert "#define" in out.header_code
