# Quickstart with mistral-vibe (60 seconds)

This page is the fastest path from "I just heard about fortranspire" to
"a Mistral model just told me how much it would cost to port a real
Fortran kernel to GPU." It assumes nothing — no API key, no Mac mini,
no Météo-France codebase on disk. By the end, you will have:

1. `fortranspire` installed via `pip` (one step, no submodule dance);
2. `mistral-vibe` installed via Homebrew and registered with the
   `fortranspire` MCP server over stdio;
3. A real port-cost estimate on a PHYEX kernel, printed in your
   terminal, from a French/English natural-language prompt.

The wall-clock target is 60 seconds. The actual blocker is going to be
`pip install` (~20 s) and `brew install mistral-vibe` (~10 s on a warm
cache). Everything else is configuration.

## Prerequisites

```bash
# uv (single command, manages Python + venvs for you)
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
# Homebrew on macOS (the only supported way to install mistral-vibe today)
brew --version
# A Mistral API key — generate one at https://console.mistral.ai/api-keys
# (Scale plan recommended; the Experiment free tier rate-limits you out
# of multi-tool sessions in under a minute).
echo $MISTRAL_API_KEY
```

## Step 1 — Install fortranspire and mistral-vibe

```bash
uv tool install fortranspire          # installs the `fortranspire` console
                                      # script in an isolated venv, on PATH
brew install mistral-vibe             # `vibe` console script, brew-managed
```

`uv tool install` is the recommended path: it avoids polluting any
existing virtualenv and puts the `fortranspire` binary on your `PATH`.
If you're already inside a project venv, `uv pip install fortranspire`
works too.

## Step 2 — Register fortranspire in `~/.vibe/config.toml`

First, find the absolute path of the installed binary — vibe is a
brew subprocess and doesn't always inherit your full shell `PATH`:

```bash
which fortranspire           # e.g. /Users/you/.local/bin/fortranspire
```

Then edit `~/.vibe/config.toml` and append:

```toml
[[mcp_servers]]
name      = "fortranspire"
transport = "stdio"
command   = "/Users/you/.local/bin/fortranspire"   # paste the path from `which`
args      = ["mcp", "--stdio"]

# Forward Mistral creds + model picks to the spawned MCP process so the
# LLM verbs (translate_kernel_gpu, generate_docs --with-llm, ask_agent)
# can reach api.mistral.ai. ${VAR} substitution reads from your shell env.
[mcp_servers.env]
MISTRAL_API_KEY  = "${MISTRAL_API_KEY}"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1"
MISTRAL_MODEL    = "codestral-latest"
```

If your config already declares `mcp_servers = []` (the default), delete
or comment that line first — TOML refuses to mix the inline-empty form
with the array-of-tables form.

## Step 3 — Pick a Fortran target (PHYEX is the demo)

```bash
git clone https://github.com/UMR-CNRM/PHYEX ~/PHYEX
cd ~/PHYEX
```

[PHYEX](https://github.com/UMR-CNRM/PHYEX) bundles Météo-France physics
schemes (microphysics, turbulence, convection). The kernels are short,
self-contained, and CeCILL-C licensed — a fair demo target that exists
publicly on GitHub.

## Step 4 — Launch vibe and talk

```bash
/opt/homebrew/opt/mistral-vibe/bin/vibe --trust
```

Then, in the interactive session, type:

```
Donne-moi le coût estimé de portage GPU pour
src/common/turb/mode_compute_function_thermo.F90.
Pas d'appel LLM côté fortranspire, juste l'estimation.
```

Within 2 seconds, vibe routes the request to the
`fortranspire_explain_port_cost` MCP tool and renders the table:

```
• Fichiers analysés : 1/1 avec succès
• Routines détectées : 1
• Coût estimé pour un passage complet de portage : 0,04 USD
• Modèle de raisonnement   : mistral-large-latest
• Modèle de génération     : codestral-latest
• Risques structurels      : aucun
```

No LLM tokens were consumed by fortranspire to produce this estimate —
the analysis is deterministic, driven by [Loki](https://github.com/ecmwf-ifs/loki)
AST parsing. The only Mistral call in this flow is vibe's own planner
that decided which tool to invoke (~$0.005 on codestral).

## Step 5 — Push it further

The natural-language surface above keeps working for the full pipeline:

```
Maintenant lance la Phase-1 GPU sur ce même fichier.
Génère le wrapper OpenACC + Cython.
```

That dispatches `fortranspire_translate_kernel_gpu` and ~$0.04 of
codestral tokens later you have `output/fortran_gpu/kernel_pure.f90`,
`output/fortran_gpu/kernel_gpu.f90`, and `output/cython/*.pyx` on disk.

For the full integration surface (9 MCP tools, HTTP/SSE alternative
deployment, LaunchAgent service template, OAuth-backed remote MCPs),
see [Integrations — mistral-vibe](../integrations/mistral-vibe.md).

## Why this matters

This is the first toolchain that lets you steer **legacy Fortran HPC
porting from inside a sovereign EU-resident LLM session**. The codebase
never leaves your disk (the MCP server is a local stdio subprocess),
inference runs on Mistral's EU infrastructure (or your self-hosted
endpoint), and the deterministic structural work (Loki AST analysis,
SARIF reporting, call-graph extraction) burns zero tokens.

The pivot to sovereignty is not aesthetic — it's the reason `Météo-France`,
`ECMWF`, and the major French and European HPC centres can adopt this
class of tooling at all. US-hosted closed models are off the table for
several of these institutions; codestral + fortranspire is a
demonstrable alternative that works **today**.

## Next steps

- [`integrations/mistral-vibe.md`](../integrations/mistral-vibe.md) —
  full PHYEX walkthrough (triage / call-graph / Phase-1).
- [`integrations/claude-code.md`](../integrations/claude-code.md) —
  same agent, Claude Code as the surface (mixed sovereignty posture).
- [Roadmap](https://github.com/maurinl26/fortranspire/blob/main/ROADMAP.md)
  — what's next: gt4py.next codegen track, OpenStack / EWC managed
  deployment, JOSS paper milestone.
- Hire or collaborate: [maurin.loic.ac@gmail.com](mailto:maurin.loic.ac@gmail.com).
