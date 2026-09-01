"""Inline-source MCP tools for hosted use (issue #51).

The path-taking tools assume the client and server share a filesystem —
true for a local stdio IDE, false for a hosted SSE endpoint. A Le Chat
user's Fortran file is on their machine, not on ours. These variants take
the source text, so the hosted connector is useful rather than merely
reachable, and they are the only tools safe to face a public directory:
deterministic, no LLM, no token, nothing persisted.
"""
from __future__ import annotations

import glob
import os

from fortranspire.config import config
from fortranspire.server import (
    analyze_source,
    build_call_graph_source,
    explain_source,
)

FREE = """subroutine axpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a, x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
     y(i) = y(i) + a * x(i)
  end do
end subroutine axpy
"""

# Fixed-form, uppercase suffix, with a cpp conditional whose body is I/O.
FIXED_CPP = """      SUBROUTINE ROS(N, Y)
      IMPLICIT NONE
      INTEGER N
      REAL*8 Y(N)
      INTEGER I
#ifdef dbg
      WRITE(6,*) 'debug'
#endif
      DO I = 1, N
         Y(I) = MAX(Y(I), 0.0D0)
      END DO
      END SUBROUTINE ROS
"""


class TestDeterministicOutput:
    def test_analyze_reports_findings(self):
        out = analyze_source(FREE)
        assert "rc=0" in out
        assert "0 error(s)" in out

    def test_explain_returns_the_port_cost_report(self):
        assert "port-cost estimate" in explain_source(FREE)

    def test_graph_returns_mermaid(self):
        assert "mermaid" in build_call_graph_source(FREE)


class TestDialectHandling:
    def test_uppercase_suffix_source_is_preprocessed(self):
        """`.F` means run cpp first; without it Loki finds nothing."""
        out = analyze_source(FIXED_CPP, filename="ros.F")
        assert "FORT009" not in out, "parse failure on source that only needed cpp"

    def test_inactive_branch_is_not_flagged_as_io(self):
        """The WRITE sits in an undefined #ifdef — it must not raise FORT001.

        Otherwise the hosted tool refuses a portable kernel over I/O that
        the build would have compiled out.
        """
        out = analyze_source(FIXED_CPP, filename="ros.F")
        assert "FORT001" not in out

    def test_non_smooth_construct_is_detected(self):
        assert "FORT031" in analyze_source(FIXED_CPP, filename="ros.F")

    def test_a_lowercase_default_is_free_form(self):
        """No filename → `.f90`, so free-form source parses."""
        assert "rc=0" in analyze_source(FREE)


class TestHostedSafety:
    def test_oversize_source_is_refused(self):
        """A hosted endpoint must not let one request paste a whole tree."""
        out = analyze_source("x = 1\n" * 400_000)
        assert "rc=2" in out and "over the" in out

    def test_no_temp_files_are_left_behind(self):
        analyze_source(FREE)
        explain_source(FREE)
        build_call_graph_source(FREE)
        assert glob.glob(os.path.join(config.workspace_dir, "lechat-*")) == []

    def test_a_directory_in_filename_does_not_steer_the_path(self):
        """A hosted caller must not choose where the temp file lands."""
        # `../../etc/passwd.f90` must not escape; only the suffix is used.
        out = analyze_source(FREE, filename="../../../etc/evil.f90")
        assert "rc=0" in out
        assert glob.glob(os.path.join(config.workspace_dir, "lechat-*")) == []

    def test_source_tools_call_no_llm(self):
        """The whole point of the public surface — verified structurally."""
        import inspect

        from fortranspire import server

        for name in ("analyze_source", "explain_source", "build_call_graph_source"):
            src = inspect.getsource(getattr(server, name))
            assert "get_llm" not in src
            assert "_get_agent" not in src


class TestRegistration:
    def test_source_tools_are_in_the_canonical_surface(self):
        from fortranspire.server import _TOOL_NAMES

        for name in ("analyze_source", "explain_source", "build_call_graph_source"):
            assert name in _TOOL_NAMES


class TestConnectorSurface:
    """The Le Chat connector is a public directory listing (issue #51)."""

    def _connector(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        return json.loads((root / "integration" / "le-chat-connector.json").read_text())

    def test_connector_exposes_no_token_spending_tool(self):
        """A public endpoint must not run the operator's LLM key on demand."""
        spends_tokens = {
            "translate_kernel_gpu", "translate_kernel", "profile_kernels",
            "ask_agent", "generate_docs",
        }
        declared = {t["id"] for t in self._connector()["tools"]}
        assert not (declared & spends_tokens), (
            f"public connector exposes token-spending tools: {declared & spends_tokens}"
        )

    def test_connector_exposes_no_writing_tool(self):
        for tool in self._connector()["tools"]:
            assert tool.get("side_effects", []) == [], f"{tool['id']} writes files"
            assert tool.get("llm_calls_per_invocation", 0) == 0

    def test_connector_tools_all_take_inline_source(self):
        """A hosted server shares no filesystem with the caller."""
        for tool in self._connector()["tools"]:
            names = {i["name"] for i in tool["inputs"]}
            assert "source" in names, f"{tool['id']} takes a path, not source"
            assert "path" not in names

    def test_connector_tools_exist_on_the_server(self):
        from fortranspire.server import _TOOL_NAMES

        for tool in self._connector()["tools"]:
            assert tool["id"] in _TOOL_NAMES
