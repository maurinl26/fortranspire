"""Tests for the JSONL tracer + per-model pricing — issue #5."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fortranspire.observability.pricing import (
    clear_cache as clear_pricing_cache,
    estimate_cost_usd,
)
from fortranspire.observability.tracer import Span, Tracer


# ── Pricing ─────────────────────────────────────────────────────────────────

def test_pricing_known_model():
    # mistral-large: 2 USD input / 6 USD output per 1M tokens
    cost = estimate_cost_usd("mistral-large-latest", 1_000_000, 0)
    assert cost == pytest.approx(2.0)
    cost = estimate_cost_usd("mistral-large-latest", 0, 1_000_000)
    assert cost == pytest.approx(6.0)


def test_pricing_unknown_model_falls_back_to_default():
    # Default is (1.0, 3.0) — never silently zero.
    cost = estimate_cost_usd("brand-new-model-xyz", 1_000_000, 1_000_000)
    assert cost == pytest.approx(4.0)


def test_pricing_case_insensitive():
    a = estimate_cost_usd("mistral-large-latest", 1000, 1000)
    b = estimate_cost_usd("MISTRAL-LARGE-LATEST", 1000, 1000)
    assert a == b


def test_pricing_none_model_uses_default():
    # Robust when model name couldn't be detected from the LLM client.
    cost = estimate_cost_usd(None, 1000, 1000)
    assert cost > 0


def test_pricing_override_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    override = tmp_path / "prices.json"
    override.write_text(json.dumps({"mistral-large-latest": [99.0, 99.0]}))
    monkeypatch.setenv("FORTRANSPIRE_PRICING_FILE", str(override))
    clear_pricing_cache()
    cost = estimate_cost_usd("mistral-large-latest", 1_000_000, 0)
    assert cost == pytest.approx(99.0)


# ── Tracer / Span ───────────────────────────────────────────────────────────

def test_tracer_writes_jsonl(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    tr = Tracer(output_path=out, tenant_id="t-001", enabled=True)
    with tr.span(node="doc_routine", model="codestral-latest") as span:
        span.record_tokens(prompt=120, completion=80)
        span.annotate(routine="update_vx")

    assert out.is_file()
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tenant_id"] == "t-001"
    assert record["node"] == "doc_routine"
    assert record["model"] == "codestral-latest"
    assert record["prompt_tokens"] == 120
    assert record["completion_tokens"] == 80
    assert record["cost_usd"] > 0
    assert record["error"] is None
    assert record["extra"]["routine"] == "update_vx"


def test_tracer_appends_records(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    tr = Tracer(output_path=out, enabled=True)
    for i in range(3):
        with tr.span(node=f"node_{i}", model="codestral-latest") as span:
            span.record_tokens(prompt=10, completion=20)
    assert len(out.read_text().splitlines()) == 3


def test_tracer_captures_errors(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    tr = Tracer(output_path=out, enabled=True)
    with pytest.raises(RuntimeError):
        with tr.span(node="extractor", model="mistral-large-latest"):
            raise RuntimeError("boom")
    record = json.loads(out.read_text().splitlines()[0])
    assert record["node"] == "extractor"
    assert record["error"] == "RuntimeError: boom"


def test_tracer_disabled_writes_nothing(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    tr = Tracer(output_path=out, enabled=False)
    with tr.span(node="doc_routine") as span:
        span.record_tokens(prompt=999, completion=999)
    assert not out.exists()


def test_tracer_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "custom_path.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_TRACE_PATH", str(target))
    monkeypatch.setenv("FORTRANSPIRE_TENANT_ID", "my-tenant")
    monkeypatch.setenv("FORTRANSPIRE_TRACE", "1")
    tr = Tracer()
    assert tr.output_path == target
    assert tr.tenant_id == "my-tenant"
    assert tr.enabled is True


def test_tracer_disable_via_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORTRANSPIRE_TRACE", "0")
    tr = Tracer()
    assert tr.enabled is False


def test_span_record_tokens_accepts_zero_or_none():
    # Span is module-level safe to instantiate without a Tracer for unit tests.
    span = Span(node="x")
    span.record_tokens(prompt=0, completion=0)
    span.record_tokens(prompt=None, completion=None)  # type: ignore[arg-type]
    assert span.prompt_tokens == 0
    assert span.completion_tokens == 0


def test_tracer_does_not_crash_on_readonly_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # Telemetry must never fail the pipeline.
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)  # r-x; can't create file
    try:
        target = readonly / "traces.jsonl"
        tr = Tracer(output_path=target, enabled=True)
        with tr.span(node="doc_routine"):
            pass
        err = capsys.readouterr().err
        # Either succeeded somehow, or printed a warning — neither raises.
        assert "fortranspire.observability" in err or target.exists()
    finally:
        readonly.chmod(0o700)  # restore so pytest can clean up
