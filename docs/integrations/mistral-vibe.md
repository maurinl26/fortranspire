# Use fortranspire from mistral-vibe

[`mistral-vibe`](https://github.com/mistralai/mistral-vibe) — Mistral's
CLI coding agent — speaks the
[Model Context Protocol](https://modelcontextprotocol.io) natively.
fortranspire ships a FastMCP server with both **stdio** (vibe spawns it
per session — zero ports, zero auth) and **HTTP/SSE** (a permanent
service for CI / LAN / Claude Desktop) transports.

The recommended path on a developer Mac (or a Mac mini acting as your
self-hosted runner) is stdio: vibe boots the server when you `cd` into a
trusted project, the tools appear in your toolbox, and they vanish when
you exit. No daemon to babysit.

This is the strongest demonstration of the sovereignty story (see
[`Architecture vs LLM`](../concepts/llm-endpoints.md)): Loki + the
deterministic harness absorb 60-70% of the work, and the LLM (Mistral
on mistral-vibe, Claude on Claude Code) is a fungible component.

## Prerequisites

```bash
brew install mistral-vibe                       # installs `vibe` at /opt/homebrew/opt/mistral-vibe/bin/vibe
pip install fortranspire                        # installs `fortranspire` console script
fortranspire mcp --stdio < /dev/null            # sanity-check: returns immediately, no banner
```

The third command should exit cleanly with no output — FastMCP is
waiting for JSON-RPC on stdin and shuts down on EOF.

## Self-hosted runner on a Mac mini (stdio)

This is the path you want if vibe and the MCP server live on the same
host. Edit `~/.vibe/config.toml` and append:

```toml
[[mcp_servers]]
name      = "fortranspire"
transport = "stdio"
command   = "fortranspire"
args      = ["mcp", "--stdio"]

# Optional — the server reads these to call out to Mistral when you
# invoke an LLM verb (translate_kernel_gpu, generate_docs --with-llm).
[mcp_servers.env]
MISTRAL_API_KEY = "${MISTRAL_API_KEY}"
MISTRAL_MODEL   = "codestral-latest"
```

> **Path resolution.** If `which fortranspire` is empty in vibe's
> environment (Homebrew shells out without your full profile), pin
> the absolute path instead — e.g.
> `command = "/Users/you/.venvs/fortranspire/bin/fortranspire"`.

Trust the project directory so vibe will let the MCP server see your
sources:

```bash
vibe --trust --workdir ~/work/phyex            # one-shot
# or persist:
# vibe                                          # then `/trust` inside the TUI
```

Verify the registration from inside a vibe session:

```
/mcp                                           # lists discovered servers
/tools                                         # the 9 fortranspire_* tools should appear
```

## Talking to it in natural language

Once the server is registered, vibe's planner can call the tools from
plain French/English. A typical "is this kernel portable?" session:

```
> Regarde phyex/src/MNH/rain_ice.f90 et estime le coût d'un portage GPU.
```

vibe will call `fortranspire_explain_port_cost`, then summarise the
table it gets back — routine count, control-flow complexity, GPU
portability score, expected token spend for a Phase-1 port. No source
file leaves the process; only the rendered report flows to the model.

A few prompts that map cleanly onto the exposed tools:

| Intent                                                       | Tool that fires            |
| ------------------------------------------------------------ | -------------------------- |
| "Audite ce fichier" / "any GPU-unfriendly patterns?"         | `analyze_kernels`          |
| "Combien coûterait un port ?" / "is it worth porting?"       | `explain_port_cost`        |
| "Dessine le graphe d'appels"                                 | `build_call_graph`         |
| "Documente ces routines"                                     | `generate_docs`            |
| "Génère le wrapper OpenACC + Cython"                         | `translate_kernel_gpu`     |
| "Prototype une version JAX"                                  | `translate_kernel`         |

The full surface and arguments:

| Tool                     | What it does                                          | LLM call?       |
| ------------------------ | ----------------------------------------------------- | --------------- |
| `analyze_kernels`        | Loki AST analysis, optional SARIF output              | No              |
| `explain_port_cost`      | Pre-flight cost & risk estimate                       | No              |
| `build_call_graph`       | Mermaid call-graph                                    | No              |
| `generate_docs`          | `!>` docstring injection (no-LLM or LLM-driven)       | Optional        |
| `translate_kernel_gpu`   | Phase 1 — Fortran → OpenACC + Cython                  | Yes             |
| `translate_kernel`       | Phase 2 — Fortran → JAX (experimental)                | Yes             |
| `profile_kernels`        | Performance benchmarking                              | No              |
| `ask_agent`              | Natural-language query against the code               | Yes             |
| `agent_status`           | Dump server config                                    | No              |

## 5-minute demo on PHYEX

[PHYEX](https://github.com/UMR-CNRM/PHYEX) bundles the Météo-France
physics schemes (microphysics, turbulence, convection). The kernels
are small, self-contained, and ideal for a guided demo.

```bash
git clone https://github.com/UMR-CNRM/PHYEX ~/work/phyex
cd ~/work/phyex
vibe --trust .
```

Then, in the vibe session, a guided three-step demo:

**1. Triage — is this kernel portable?**

```
> Compare ces deux fichiers pour un portage GPU :
>   src/common/turb/mode_compute_function_thermo.F90
>   src/common/turb/mode_tke_eps_sources.F90
> Lequel est portable tel quel ? Lequel demande un refactor d'abord ?
```

vibe calls `fortranspire_explain_port_cost` sur chaque fichier. La
sortie surface `FORT001 — I/O in kernel candidate` sur le second (un
`PRINT` dans `TKE_EPS_SOURCES` qui bloque le port GPU), et un coût
propre (~$0.01) sur le premier.

**2. Visualiser la structure du portable**

```
> Dessine le call-graph de mode_compute_function_thermo.F90.
```

`fortranspire_build_call_graph` rend un Mermaid `flowchart LR`
embarquable directement dans une PR ou un mdBook.

**3. Générer le wrapper OpenACC + Cython**

```
> Génère le port OpenACC + Cython pour mode_compute_function_thermo.F90.
```

`fortranspire_translate_kernel_gpu` enchaîne le pipeline Phase-1 et
écrit `output/fortran_gpu/*.f90` + `output/cython/*.pyx` à côté du
`cwd`. vibe propose ensuite la compilation + validation via gfortran.

> **Tip.** Toujours commencer par `explain_port_cost` avant d'appeler
> un verbe LLM — ça évite de griller des tokens sur un fichier qui se
> fera retoquer en validation pour une raison structurelle (I/O,
> `COMMON`, pointeurs).

## Alternative — permanent HTTP service (CI / LAN access)

When you want the same server reachable from Claude Code Desktop, from
your laptop hitting the Mac mini over the LAN, or from a CI runner:

```bash
fortranspire mcp                                # default: SSE on $MCP_HOST:$MCP_PORT
# or
MCP_HOST=0.0.0.0 MCP_PORT=8000 fortranspire mcp
```

The server listens on SSE at `http://<host>:8000/sse`. Auth is opt-in:

```bash
export FORTRANSPIRE_TOKENS_FILE=/etc/fortranspire/tokens.json
export FORTRANSPIRE_AUDIT_SECRET=<random-256-bit-secret>
fortranspire mcp
```

See [security](../security.md) for the token registry format
and the HMAC-signed audit log.

vibe's HTTP registration looks like:

```toml
[[mcp_servers]]
name      = "fortranspire"
transport = "streamable-http"
url       = "http://mac-mini.local:8000/sse"

[mcp_servers.auth]
type        = "static"
api_key_env = "FORTRANSPIRE_TOKEN"
```

To keep the server alive across reboots, drop a LaunchAgent under
`~/Library/LaunchAgents/fr.fortranspire.mcp.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>fr.fortranspire.mcp</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/fortranspire</string>
    <string>mcp</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MCP_HOST</key><string>0.0.0.0</string>
    <key>MCP_PORT</key><string>8000</string>
    <key>MISTRAL_API_KEY</key><string>...</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/fortranspire-mcp.out.log</string>
  <key>StandardErrorPath</key><string>/tmp/fortranspire-mcp.err.log</string>
</dict>
</plist>
```

Then `launchctl load -w ~/Library/LaunchAgents/fr.fortranspire.mcp.plist`.

## Why both Claude Code and mistral-vibe?

| Surface                            | LLM                  | Sovereignty                                   |
| ---------------------------------- | -------------------- | --------------------------------------------- |
| Claude Code                        | Claude (US)          | Mixed — code stays local, model is US-hosted  |
| mistral-vibe                       | Mistral (EU)         | Full — code + inference both EU-resident      |
| `fortranspire mcp` self-hosted     | Mistral self-hosted  | Air-gapped possible                           |

Same agent, three sovereignty postures. Choose per project / per
customer requirement.

## Known gaps

- The MCP server returns plain-text responses (Markdown-friendly).
  Structured-output support (Pydantic JSON schemas) is on the roadmap
  — track progress in the [issue tracker](https://github.com/maurinl26/fortranspire/issues).
- The `translate_kernel_gpu` tool runs the full Phase-1 LangGraph in
  the same process as the MCP server. On large kernels (>2k LoC) this
  can hold vibe's UI for a few minutes — prefer running the CLI
  `fortranspire gpu file.f90` for batch work.
