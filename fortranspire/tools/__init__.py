"""Tool exports for the LangChain ReAct agent.

`run_shell` is **opt-in** since fortranspire 0.2.1. Even though the agent
loop ostensibly drives it from a trusted prompt, the same tool surface is
reachable through the `ask_agent` MCP tool — which means an
HTTP-deployed MCP server (`fortranspire mcp` without `--stdio`) would
expose unauthenticated shell execution if any auth middleware bypass
were ever introduced. Default-off closes that blast radius.

To re-enable shell access, set ``FORTRANSPIRE_ENABLE_SHELL=1`` in the
process environment. Suitable for local CLI use (`fortranspire mcp
--stdio` spawned by your IDE), never recommended for a public network
deployment.
"""
import os

from fortranspire.tools.file_tools import read_file, write_file, list_directory
from fortranspire.tools.search_tools import search_code
from fortranspire.tools.shell_tools import run_shell

_BASE_TOOLS = [read_file, write_file, list_directory, search_code]

if os.getenv("FORTRANSPIRE_ENABLE_SHELL") == "1":
    ALL_TOOLS = [*_BASE_TOOLS, run_shell]
else:
    ALL_TOOLS = list(_BASE_TOOLS)

__all__ = [
    "read_file", "write_file", "list_directory", "run_shell",
    "search_code", "ALL_TOOLS",
]
