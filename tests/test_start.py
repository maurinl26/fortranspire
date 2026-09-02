"""The didactic `fortranspire start` entry point: triage + toolchain doctor + guide."""
import pytest

from fortranspire.agent.recon import RoutineTarget
from fortranspire.agent.start import (
    _VERB,
    _capabilities,
    _guide,
    _render_toolchain,
    main,
)


def _t(name, file, rank=5.0, reason="JAX score 5/5; leaf", role="kernel", jax=5):
    return RoutineTarget(name=name, file=file, role=role, target="jax",
                         jax_score=jax, rank=rank, reason=reason)


def test_guide_ranks_targets_and_gives_the_exact_next_command():
    targets = [_t("dotp", "/r/reduce.f90"), _t("saxpy", "/r/saxpy.f90", rank=4.0)]
    text, best, tc = _guide(targets, "/r", top=5)
    assert "Top 2 target(s), best first:" in text
    assert best.name == "dotp"                              # highest rank first
    assert "why  : JAX score 5/5; leaf" in text            # the reason is shown
    assert "next : fortranspire gpu" in text               # deterministic GPU default
    assert "fortranspire translate" in text                # JAX offered as Phase 2


def test_guide_reports_the_toolchain_and_capabilities():
    text, _best, tc = _guide([_t("k", "/r/k.f90")], "/r", top=5)
    assert "Toolchain on this machine:" in text
    assert "What you can do here:" in text
    # Phase-1 generation is always available regardless of what's installed.
    assert "Generate Phase-1 GPU" in text
    assert isinstance(tc, dict) and "nvfortran" in tc


def test_capabilities_gate_on_installed_tools():
    caps = dict((c, ok) for c, ok, _n in _capabilities(
        {"gfortran": True, "jax": True, "meson": True, "ninja": True,
         "nvfortran": False, "nvidia-smi": False}))
    assert caps["Generate Phase-1 GPU (OpenACC) + Cython wrapper"] is True
    assert caps["Translate to JAX + check gradients"] is True
    assert caps["Numerically validate JAX vs Fortran (f2py)"] is True
    assert caps["Compile & run the OpenACC GPU port"] is False   # no nvfortran/GPU


def test_capability_jax_equivalence_needs_the_full_toolchain():
    caps = dict((c, ok) for c, ok, _n in _capabilities(
        {"gfortran": True, "jax": True, "meson": True, "ninja": False,
         "nvfortran": False, "nvidia-smi": False}))
    assert caps["Numerically validate JAX vs Fortran (f2py)"] is False   # ninja missing


def test_render_toolchain_marks_present_and_missing():
    text, tc = _render_toolchain()
    assert text.count("✓") + text.count("✗") >= 10        # one mark per probed tool
    assert set(tc.values()) <= {True, False}


def test_no_portable_kernel_points_to_recon_and_explain():
    driver = _t("main_driver", "/r/main.f90", rank=0.0, role="driver")
    text, best, _tc = _guide([driver], "/r", top=5)
    assert best is None
    assert "fortranspire recon /r" in text
    assert "fortranspire explain /r" in text


def test_target_to_verb_mapping():
    assert _VERB == {"gpu": "gpu", "jax": "translate"}


def test_start_main_on_a_missing_path_is_clean(capsys):
    rc = main(["/nonexistent/dir/xyz"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "path not found" in err


def test_start_main_on_an_empty_dir_renders_the_guide(tmp_path, capsys):
    rc = main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fortranspire start" in out
    assert "Toolchain on this machine:" in out
