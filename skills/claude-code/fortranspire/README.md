# fortranspire — Claude Code skill

A drop-in [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)
that lets Claude port, analyze, document, and benchmark legacy Fortran 90
code through the `fortranspire` CLI — without leaving the IDE.

## Install

### One-time setup

```bash
pip install fortranspire
```

You can stop here if you only need `analyze`, `explain`, `format`,
`graph`, and `doc --no-llm`. For LLM-driven verbs (`gpu`, `port-batch`,
`translate`, `doc --with-llm`) add the GPU extra:

```bash
pip install "fortranspire[gpu]"
```

…and configure an OpenAI-compatible endpoint (Mistral La Plateforme by
default — see
[docs/getting-started/installation.md](../../../docs/getting-started/installation.md)).

### Add the skill to Claude Code

Copy the `fortranspire/` directory into your project's `.claude/skills/`:

```bash
# In your project root (the one Claude Code is operating on)
mkdir -p .claude/skills
cp -r <path-to-fortranspire-repo>/skills/claude-code/fortranspire .claude/skills/
```

Or, if you cloned the fortranspire repo locally:

```bash
git clone https://github.com/maurinl26/fortranspire
ln -s "$PWD/fortranspire/skills/claude-code/fortranspire" .claude/skills/fortranspire
```

Restart Claude Code (or reload the project) — `/fortranspire` should now
appear in the slash-command menu, and Claude will auto-invoke the skill
when it sees Fortran-porting intent in your messages.

## How it works

The skill is a [`SKILL.md`](./SKILL.md) file with a frontmatter
`description` that tells Claude when to use it. The body documents every
`fortranspire` subcommand, prerequisites to check, and a quick decision
flow (e.g., "always run `explain` before paying for `gpu` tokens").

Claude calls into `fortranspire` via plain `Bash` tool calls — no MCP
plumbing needed for the local install. For remote / hosted usage, see
[docs/integrations/claude-code.md](../../../docs/integrations/claude-code.md).

## Verify

After installing, ask Claude:

> Can you analyze the Fortran file `path/to/kernel.f90`?

Claude should invoke `fortranspire analyze path/to/kernel.f90` and
summarize the toolchain probe + findings. If it doesn't, check that
`.claude/skills/fortranspire/SKILL.md` exists and Claude Code has been
reloaded.

## Issues / contributions

File issues at <https://github.com/maurinl26/fortranspire/issues>. The
SKILL.md lives in-repo so changes ship with each fortranspire release.
