"""`graph` must degrade like `analyze`/`explain` when Loki is unavailable (#71).

The bug: `analyze`/`explain` absorbed a Loki import failure and produced their
report, but `graph` let the traceback kill the process. These pin the graceful
path — a parse error becomes a reported degradation, never a crash.
"""
from fortranspire.agent.call_graph import (
    FileGraph,
    main,
    render_mermaid,
    render_report,
)


def test_render_mermaid_handles_a_parse_error():
    out = render_mermaid(FileGraph(file="x.f90", parse_error="loki unavailable: boom"))
    assert "parse error" in out.lower()
    assert "boom" in out


def test_render_report_handles_a_parse_error():
    out = render_report([FileGraph(file="x.f90", parse_error="loki unavailable: boom")])
    assert "Parse failure" in out
    assert "boom" in out


def test_main_returns_cleanly_on_an_unparseable_file(tmp_path, capsys, monkeypatch):
    # Force the per-file extraction to fail as a Loki-unavailable import would,
    # and assert the command still returns 0 with a degraded report.
    import fortranspire.agent.call_graph as cg

    monkeypatch.setattr(
        cg, "_extract_one",
        lambda path: FileGraph(file=path, parse_error="loki unavailable: simulated"),
    )
    f = tmp_path / "k.f90"
    f.write_text("subroutine k(a, b)\n  b = a\nend subroutine\n")

    rc = main([str(f)])
    assert rc == 0                                   # degrades, never crashes
    assert "simulated" in capsys.readouterr().out
