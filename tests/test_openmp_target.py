"""Tests for the OpenMP target alternative emission — issue #18.

Verifies that:
  * The new OpenMP prompts load correctly (EN + FR, v2 only).
  * The openacc node dispatches to the right prompt based on
    ``state["gpu_pragma"]``.
  * The CLI `--gpu-pragma {acc,omp}` flag plumbs through correctly.

End-to-end LLM behaviour is not tested here (would need a real key);
we patch the LLM client to capture which system prompt was sent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from fortranspire.prompts.loader import clear_cache, load_prompt


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    clear_cache()
    yield
    clear_cache()


# ── Prompt files exist + look like the right family ────────────────────────

@pytest.mark.parametrize("name,version,lang,expected_words", [
    ("openmp_kernel", "v2", "en", ["!$omp target", "OpenMP target-offload"]),
    ("openmp_driver", "v2", "en", ["!$omp target data", "OpenMP target-offload"]),
    ("openmp_kernel", "v2", "fr", ["!$omp target", "expert OpenMP"]),
    ("openmp_driver", "v2", "fr", ["!$omp target data", "expert OpenMP"]),
])
def test_openmp_prompt_loads(name: str, version: str, lang: str, expected_words: list[str]):
    text = load_prompt(name, version=version, lang=lang)
    for word in expected_words:
        assert word in text, f"{name}/{lang}/{version}: expected {word!r}"


def test_openmp_prompts_differ_from_openacc():
    omp_kernel = load_prompt("openmp_kernel", version="v2", lang="en")
    acc_kernel = load_prompt("openacc_kernel", version="v2", lang="en")
    assert omp_kernel != acc_kernel
    assert "!$acc" not in omp_kernel
    assert "!$omp" not in acc_kernel


# ── Node dispatch ───────────────────────────────────────────────────────────

class _CapturingLLM:
    """Stub that records every (system_prompt, user_prompt) it sees.

    Returns a dummy `annotated_fortran` so the node thinks the call
    succeeded. Bypasses the structured-output wrapper by accepting
    `.with_structured_output(...)` and returning self.
    """

    def __init__(self):
        self.system_prompts: list[str] = []
        self.model_name = "stub-model"

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages, *, config=None):
        # First message is the SystemMessage holding our prompt.
        sys_msg = messages[0].content if messages else ""
        self.system_prompts.append(sys_msg)
        class _Result:
            annotated_fortran = "subroutine k\nend subroutine\n"
        return _Result()


def _patch_get_llm(monkeypatch: pytest.MonkeyPatch, stub: _CapturingLLM):
    import fortranspire.llm as llm_mod
    monkeypatch.setattr(llm_mod, "get_llm", lambda *_a, **_kw: stub)


def _minimal_state(*, gpu_pragma: str) -> dict:
    return {
        "fortran_filepath": "/tmp/x.f90",
        "fortran_code": "",
        "ast_info": {},
        "kernel_results": [
            {
                "routine_name": "k",
                "fortran_code": (
                    "subroutine k(n, x)\n"
                    "  implicit none\n"
                    "  integer, intent(in) :: n\n"
                    "  real(8), intent(inout) :: x(n)\n"
                    "  integer :: i\n"
                    "  do i = 1, n\n"
                    "     x(i) = x(i) * 2.0d0\n"
                    "  end do\n"
                    "end subroutine k\n"
                ),
                "pure_elemental_code": "",
                "openacc_code": "",
                "intent_map": {"x": "INOUT"},
                "is_pure": False,
                "is_elemental": False,
                "has_io": False,
                "has_save": False,
                "loops": ["1,n"],
                "dimensions": {},
                "status": "extracted",
                "error_log": "",
            }
        ],
        "schema": {},
        "is_program": False,
        "module_fortran": "",
        "driver_fortran": "",
        "kernel_names": ["k"],
        "pure_elemental_fortran": "",
        "openacc_fortran": "",
        "cython_pyx": "",
        "cython_header": "",
        "cython_setup": "",
        "validation_passed": False,
        "validation_log": "",
        "gpu_pragma": gpu_pragma,
        "executed_agents": [],
    }


def test_node_uses_openacc_prompt_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)  # outputs go under tmp
    stub = _CapturingLLM()
    _patch_get_llm(monkeypatch, stub)

    from fortranspire.agent.nodes.openacc import openacc_insert_agent
    out = openacc_insert_agent(_minimal_state(gpu_pragma="acc"))

    # Kernel pragma is now derived deterministically — dispatch shows in the code.
    code = out["kernel_results"][0]["openacc_code"]
    assert "!$acc parallel loop" in code
    assert "!$omp" not in code


def test_node_uses_openmp_prompt_when_omp_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    stub = _CapturingLLM()
    _patch_get_llm(monkeypatch, stub)

    from fortranspire.agent.nodes.openacc import openacc_insert_agent
    out = openacc_insert_agent(_minimal_state(gpu_pragma="omp"))

    code = out["kernel_results"][0]["openacc_code"]
    assert "!$omp target teams distribute parallel do" in code
    assert "!$acc parallel loop" not in code


def test_node_falls_back_to_acc_on_unknown_pragma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    monkeypatch.chdir(tmp_path)
    stub = _CapturingLLM()
    _patch_get_llm(monkeypatch, stub)

    from fortranspire.agent.nodes.openacc import openacc_insert_agent
    result = openacc_insert_agent(_minimal_state(gpu_pragma="cuda"))  # nonsense

    printed = capsys.readouterr().out
    assert "unknown gpu_pragma" in printed
    assert "!$acc parallel loop" in result["kernel_results"][0]["openacc_code"]


# ── CLI flag plumbing ───────────────────────────────────────────────────────

def test_cli_gpu_pragma_flag_propagates_to_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """`fortranspire gpu --gpu-pragma omp file.f90` must put 'omp' into state."""
    captured_state: dict = {}

    class _StubApp:
        def invoke(self, state):
            captured_state.update(state)
            return {"validation_passed": True, "validation_log": ""}

    import fortranspire.agent.translation_graph_phase1 as mod
    monkeypatch.setattr(mod, "translation_app_phase1", _StubApp())

    fixture = tmp_path / "k.f90"
    fixture.write_text("subroutine k\nend subroutine\n")

    monkeypatch.setattr(sys, "argv", [
        "agent-gpu", str(fixture), "--gpu-pragma", "omp",
    ])
    from fortranspire.agent.cli import run_translate_gpu
    run_translate_gpu()
    assert captured_state.get("gpu_pragma") == "omp"


def test_cli_gpu_pragma_defaults_to_acc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    captured_state: dict = {}

    class _StubApp:
        def invoke(self, state):
            captured_state.update(state)
            return {"validation_passed": True, "validation_log": ""}

    import fortranspire.agent.translation_graph_phase1 as mod
    monkeypatch.setattr(mod, "translation_app_phase1", _StubApp())

    fixture = tmp_path / "k.f90"
    fixture.write_text("subroutine k\nend subroutine\n")

    monkeypatch.setattr(sys, "argv", ["agent-gpu", str(fixture)])
    from fortranspire.agent.cli import run_translate_gpu
    run_translate_gpu()
    assert captured_state.get("gpu_pragma") == "acc"
