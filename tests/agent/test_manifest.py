"""Run manifest — provenance and the reproducibility verdict (#agentic-3)."""
from fortranspire.agent.manifest import build_manifest


def _state(**over):
    s = {
        "kernel_results": [
            {"routine_name": "k", "purity": "pure", "status": "success",
             "jax_code": "def k(): pass", "derived_deterministically": True,
             "gradcheck": {"status": "pass", "max_abs_err": 1e-12},
             "equivalence": {"status": "pass", "max_abs_err": 0.0}},
        ],
        "gradcheck_passed": True, "equivalence_passed": True,
        "executed_agents": ["parser", "jax_kernel", "gradcheck"],
    }
    s.update(over)
    return s


def test_derived_only_run_is_reproducible():
    m = build_manifest(_state(), input_path="/tmp/x.f90", target="jax")
    assert m["llm_used"] is False
    assert m["reproducible"] is True
    assert "no LLM" in m["reproducible_reason"]
    assert m["input"]["digest"].startswith("sha256:") or m["input"]["digest"] == "unknown"


def test_llm_run_reproducible_only_with_temp0_and_pinned(monkeypatch):
    llm_kernel = {"routine_name": "k", "status": "success",
                  "jax_code": "def k(): pass"}  # no derived flag → LLM
    # temp 0 + moving `-latest` → NOT reproducible
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-latest")
    import importlib, fortranspire.config as cfg
    importlib.reload(cfg)
    m = build_manifest(_state(kernel_results=[llm_kernel]), input_path="/tmp/x.f90", target="jax")
    assert m["llm_used"] is True
    assert m["reproducible"] is False          # moving tag

    monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-2411")   # pinned
    importlib.reload(cfg)
    m2 = build_manifest(_state(kernel_results=[llm_kernel]), input_path="/tmp/x.f90", target="jax")
    assert m2["reproducible"] is True          # temp 0 + pinned
    importlib.reload(cfg)


def test_manifest_never_leaks_the_api_key():
    m = build_manifest(_state(), input_path="/tmp/x.f90", target="jax")
    blob = str(m).lower()
    assert "api_key" not in blob and "authorization" not in blob
    assert m["model"]["endpoint_host"]  # host only, no scheme/key
