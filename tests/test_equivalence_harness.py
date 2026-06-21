"""Tests for the equivalence-harness node added by issue #11."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from fortranspire.agent.nodes.equivalence_harness import (
    _kernel_test_body,
    _render_test_file,
    equivalence_harness_agent,
)


def _make_kernel(name: str = "update_vx", *, has_io: bool = False) -> dict:
    return {
        "routine_name": name,
        "fortran_code": "",
        "pure_elemental_code": "",
        "openacc_code": "",
        "intent_map": {
            "vx":       "INOUT",
            "sigma_xx": "IN",
            "dx":       "IN",
            "nx":       "IN",
            "ny":       "IN",
        },
        "is_pure": False,
        "is_elemental": False,
        "has_io": has_io,
        "has_save": False,
        "loops": [],
        "dimensions": {
            "vx":       ["nx", "ny"],
            "sigma_xx": ["nx", "ny"],
        },
        "status": "extracted",
        "error_log": "",
    }


def test_kernel_body_contains_assertion_for_inout_only():
    body = _kernel_test_body(_make_kernel("update_vx"))
    # Allocations
    assert "vx_cpu" in body and "vx_gpu" in body
    assert "sigma_xx_cpu" in body and "sigma_xx_gpu" in body
    # Both backends called
    assert "cpu_mod.update_vx(" in body
    assert "gpu_mod.update_vx(" in body
    # Only INTENT(INOUT)/OUT trigger assertions — vx is INOUT, sigma_xx is IN.
    assert "assert_allclose(vx_gpu, vx_cpu" in body
    assert "assert_allclose(sigma_xx_gpu" not in body


def test_render_skips_when_io_routines_only(tmp_path: Path):
    state = {
        "fortran_filepath": str(tmp_path / "foo.f90"),
        "kernel_results": [_make_kernel("printer", has_io=True)],
        "executed_agents": [],
    }
    out = equivalence_harness_agent(state)
    assert "equivalence_harness" in out["executed_agents"]
    # No test file emitted when nothing eligible.
    assert not (tmp_path / "output" / "tests").exists() \
        or not list((tmp_path / "output" / "tests").glob("test_*"))


def test_full_render_produces_runnable_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "fortran_filepath": str(tmp_path / "wave.f90"),
        "kernel_results": [_make_kernel("update_vx"), _make_kernel("update_sigma")],
        "executed_agents": [],
    }
    out = equivalence_harness_agent(state)

    test_path = tmp_path / "output" / "tests" / "test_wave_equivalence.py"
    readme    = tmp_path / "output" / "tests" / "README.md"
    assert test_path.is_file()
    assert readme.is_file()

    text = test_path.read_text()
    # File should at least be syntactically valid Python.
    compile(text, str(test_path), "exec")

    # Covers both routines.
    assert "def test_update_vx_equivalence" in text
    assert "def test_update_sigma_equivalence" in text

    # Carries the importorskip fallbacks (so the test self-skips when builds missing).
    assert "f2py CPU wrapper not built" in text
    assert "Cython GPU wrapper not built" in text

    # Tolerances come from env-var defaults.
    assert "ATOL" in text and "RTOL" in text

    assert "equivalence_harness" in out["executed_agents"]


def test_tolerances_overridable_via_env(monkeypatch: pytest.MonkeyPatch):
    # The env vars are read at module import time — reload to pick them up.
    monkeypatch.setenv("FORTRANSPIRE_TOLERANCE_ATOL", "1e-3")
    monkeypatch.setenv("FORTRANSPIRE_TOLERANCE_RTOL", "1e-2")
    import importlib
    from fortranspire.agent.nodes import equivalence_harness as mod
    importlib.reload(mod)
    text = mod._render_test_file(
        [_make_kernel("update_vx")],
        module_name="foo",
        original_fortran="foo.f90",
        atol=mod._DEFAULT_ATOL,
        rtol=mod._DEFAULT_RTOL,
    )
    assert "ATOL = 0.001" in text
    assert "RTOL = 0.01" in text


def test_graph_includes_new_node():
    # Smoke-import — confirms the LangGraph wiring picks up the new node
    # and the public re-exports stay clean.
    from fortranspire.agent.translation_graph_phase1 import translation_app_phase1
    assert translation_app_phase1 is not None
    from fortranspire.agent.nodes import equivalence_harness_agent
    assert callable(equivalence_harness_agent)
