"""Pydantic schemas for the LLM-emitted structured outputs of each pipeline stage.

Used with ``langchain``'s ``llm.with_structured_output(SchemaModel)`` so the
agent never has to regex-parse free-form LLM responses. When the underlying
model (Mistral La Plateforme, Codestral, a self-hosted vLLM build) supports
JSON-schema mode, LangChain wires it natively; otherwise it falls back to
JSON-mode + Pydantic validation. Either way, the agent receives a typed
object on every call.

These schemas are deliberately narrow: each one mirrors what a single node
needs from its LLM, no more. Bumping a schema in a breaking way → bump the
prompt version too (`v1` → `v2`) so the pair stays in sync.
"""
from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class DocRoutineOutput(BaseModel):
    """LLM-emitted documentation for a single Fortran routine.

    Two audiences, one round-trip:

    - ``short_summary`` — stakeholder view, ≤ 100 chars, no trailing period.
    - ``detailed`` — developer view, 2–4 sentences (INTENT semantics,
      invariants, gotchas like hidden ``SAVE`` or COMMON-block leakage).
    - ``params`` — per-argument physical meaning, units when obvious.
    """

    short_summary: str = Field(
        ...,
        max_length=120,
        description="One sentence summary for non-developer stakeholders. "
                    "No trailing period. ≤ 100 characters preferred.",
    )
    detailed: str = Field(
        ...,
        description="2–4 plain-text sentences for HPC engineers. No Markdown. "
                    "Mention INTENT semantics, invariants, and gotchas "
                    "(hidden SAVE, COMMON-block leakage, index ordering).",
    )
    params: Dict[str, str] = Field(
        default_factory=dict,
        description="`{arg_name: physical meaning + role + units}` for every "
                    "argument of the routine. Empty dict if no arguments.",
    )


class ExtractorOutput(BaseModel):
    """LLM-emitted MODULE + DRIVER split from a monolithic Fortran PROGRAM."""

    module_fortran: str = Field(
        ...,
        description="Self-contained Fortran MODULE containing only the "
                    "extracted compute kernels (subroutines).",
    )
    driver_fortran: str = Field(
        ...,
        description="Fortran PROGRAM that USEs the module and replaces each "
                    "inline 2D loop nest with a CALL to the corresponding "
                    "kernel subroutine.",
    )


class OpenACCKernelOutput(BaseModel):
    """LLM-emitted Fortran subroutine annotated with `!$acc parallel loop`."""

    annotated_fortran: str = Field(
        ...,
        description="The input subroutine with !$acc parallel loop "
                    "collapse(...) inserted before the outermost 2D loop "
                    "nest. PURE / ELEMENTAL keywords stripped if present.",
    )


class OpenACCDriverOutput(BaseModel):
    """LLM-emitted Fortran driver wrapped in an `!$acc data` region."""

    annotated_fortran: str = Field(
        ...,
        description="The original PROGRAM with !$acc data copyin/copy "
                    "around the time loop and !$acc update host directives "
                    "before periodic I/O. Existing code preserved verbatim.",
    )


class CythonPyxOutput(BaseModel):
    """LLM-emitted Cython `.pyx` wrapper."""

    pyx_code: str = Field(
        ...,
        description="Complete .pyx source — `cdef extern from \"kernel_c.h\"`, "
                    "cpdef wrappers with NumPy typed memoryviews, "
                    "`np.asfortranarray` for column-major guarantees.",
    )


class CythonHeaderOutput(BaseModel):
    """LLM-emitted C header for the Fortran iso_c_binding ABI."""

    header_code: str = Field(
        ...,
        description="C header (kernel_c.h) with include guards, `extern \"C\"` "
                    "block, and one prototype per kernel subroutine.",
    )
