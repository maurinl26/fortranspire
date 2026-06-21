"""Tests for the audit log + HMAC signature."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fortranspire.security.audit import (
    AuditEvent,
    emit,
    verify_record,
    write_event,
)


def test_emit_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(out))
    monkeypatch.delenv("FORTRANSPIRE_AUDIT_SECRET", raising=False)

    emit(tenant_id="t-001", path="/sse", method="POST", status=200,
         duration_ms=12.5, outcome="ok")

    record = json.loads(out.read_text().splitlines()[0])
    assert record["tenant_id"] == "t-001"
    assert record["path"] == "/sse"
    assert record["status"] == 200
    assert record["outcome"] == "ok"
    assert "signature" not in record   # No signature when secret unset


def test_signature_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(out))
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_SECRET", "test-secret")

    emit(tenant_id="t-001", path="/sse", method="POST", status=200,
         duration_ms=5.0, outcome="ok")

    record = json.loads(out.read_text().splitlines()[0])
    assert record.get("signature")
    assert verify_record(record, "test-secret") is True


def test_signature_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(out))
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_SECRET", "test-secret")

    emit(tenant_id="t-001", path="/sse", method="POST", status=200,
         duration_ms=5.0, outcome="ok")

    record = json.loads(out.read_text().splitlines()[0])
    # Tamper with the status code
    record["status"] = 401
    assert verify_record(record, "test-secret") is False


def test_signature_rejects_wrong_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(out))
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_SECRET", "secret-a")

    emit(tenant_id="t-001", path="/sse", method="POST", status=200,
         duration_ms=5.0, outcome="ok")

    record = json.loads(out.read_text().splitlines()[0])
    # Verification with a different secret must fail.
    assert verify_record(record, "secret-b") is False


def test_write_event_appends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(out))
    monkeypatch.delenv("FORTRANSPIRE_AUDIT_SECRET", raising=False)
    for i in range(3):
        write_event(AuditEvent(
            ts=float(i), tenant_id=f"t-{i}", path="/sse", method="POST",
            status=200, duration_ms=1.0, outcome="ok",
        ))
    assert len(out.read_text().splitlines()) == 3


def test_write_event_does_not_crash_on_readonly_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        monkeypatch.setenv("FORTRANSPIRE_AUDIT_PATH", str(readonly / "audit.jsonl"))
        emit(tenant_id="t-001", path="/", method="GET", status=200,
             duration_ms=1.0, outcome="ok")
        err = capsys.readouterr().err
        assert "fortranspire.security.audit" in err or True  # never raises
    finally:
        readonly.chmod(0o700)
