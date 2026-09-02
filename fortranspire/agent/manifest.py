"""Run manifest — provenance written next to every port (#agentic-3).

Without a record of *how* a port was produced — which model (and version), at
what temperature, with which tolerances and tool version, over which input — the
output cannot be reproduced, audited, or trusted downstream. This writes that
record as ``manifest.json`` beside the output, and states plainly whether the run
is reproducible: a run is reproducible when every kernel was *derived*
deterministically, or when the LLM ran at temperature 0 against a **pinned**
(non-``-latest``) model.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _tool_version() -> str:
    try:
        from importlib.metadata import version
        return version("fortranspire")
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256(path: str) -> str:
    try:
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def _model_info() -> Dict[str, Any]:
    from fortranspire.config import config

    model = os.getenv("MISTRAL_MODEL_REASONING") or os.getenv("MISTRAL_MODEL") \
        or config.model_name
    endpoint = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
    # host only — never the API key.
    host = endpoint.split("//", 1)[-1].split("/", 1)[0]
    pinned = "latest" not in model.lower()
    return {"model": model, "pinned": pinned, "endpoint_host": host,
            "temperature": config.temperature}


def build_manifest(final_state: Dict[str, Any], *, input_path: str,
                   target: str) -> Dict[str, Any]:
    """Assemble the provenance manifest from the finished pipeline state."""
    model = _model_info()
    kernels = []
    all_derived = True
    for k in final_state.get("kernel_results", []):
        gc = k.get("gradcheck") or {}
        eq = k.get("equivalence") or {}
        # "derived" = produced by the deterministic skeleton (no LLM); we mark
        # it on the kernel status/log — infer from the emission tag absence of
        # LLM is not stored, so treat a passing gradcheck with no repairs as best
        # effort; the reliable signal is whether an LLM was consulted at all.
        derived = bool(k.get("derived_deterministically"))
        if not derived and k.get("jax_code"):
            all_derived = False
        kernels.append({
            "name": k.get("routine_name"),
            "purity": k.get("purity"),
            "status": k.get("status"),
            "derived": derived,
            "gradcheck": gc.get("status"),
            "gradcheck_max_abs_err": gc.get("max_abs_err"),
            "equivalence": eq.get("status"),
            "equivalence_max_abs_err": eq.get("max_abs_err"),
        })

    # GPU (Phase 1): the deterministic passes use no LLM; only the extractor does,
    # and only for a monolithic PROGRAM. So the LLM is used iff `is_program`.
    if target == "gpu":
        llm_used = bool(final_state.get("is_program"))
    else:
        llm_used = not all_derived
    reproducible = (not llm_used) or (model["temperature"] == 0.0 and model["pinned"])

    gpu_section = None
    if target == "gpu":
        gpu_section = {
            "validation_level": final_state.get("validation_level", "generated"),
            "numerically_validated": bool(final_state.get("numerically_validated")),
            # Parallel GPU reductions/atomics are FP non-associative: the sum
            # order depends on threads/scheduling, so the result is NOT
            # bit-reproducible run-to-run or vs the CPU — only within a tolerance.
            "runtime_bit_reproducible": False,
            "runtime_note": ("parallel GPU reductions are FP non-associative → not "
                             "bit-reproducible; the equivalence harness compares "
                             "within (atol, rtol), not bit-exact"),
        }

    manifest = {
        "tool": "fortranspire",
        "tool_version": _tool_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,                       # "jax" | "gpu" | "gt4py"
        "input": {"path": str(input_path), "digest": _sha256(input_path)},
        "model": model,
        "llm_used": llm_used,
        "reproducible": reproducible,
        "reproducible_reason": (
            "deterministic generation, no LLM" if not llm_used
            else "temperature 0 + pinned model" if reproducible
            else "LLM at non-zero temperature or a moving `-latest` model — "
                 "pin MISTRAL_MODEL and set LLM_TEMPERATURE=0 to make it reproducible"
        ),
        "verification": {
            "gradcheck_passed": final_state.get("gradcheck_passed"),
            "gradcheck_unverified": final_state.get("gradcheck_unverified") or [],
            "equivalence_passed": final_state.get("equivalence_passed"),
        },
        "executed_agents": final_state.get("executed_agents", []),
        "kernels": kernels,
    }
    if gpu_section is not None:
        manifest["gpu"] = gpu_section
    return manifest


def write_manifest(out_path: str, manifest: Dict[str, Any]) -> None:
    Path(out_path).write_text(json.dumps(manifest, indent=2, default=str))
