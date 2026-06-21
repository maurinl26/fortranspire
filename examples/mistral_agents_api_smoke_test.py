"""Smoke test — confirm a Mistral API key works against the agent's stack.

Three checks, in increasing scope:

1. **/v1/models** — the key is valid and the EU endpoint is reachable.
2. **Chat completion** — a direct `mistral-large` call (the model used by the
   reasoning stages of the GPU porting pipeline).
3. **Agents API + MCP** — register the local `fortranspire` MCP server as a
   tool, ask the agent to port a tiny Fortran kernel, and print the result.
   This step requires the MCP server to be reachable from Mistral — set
   `--mcp-url https://<public>/sse` and run the server publicly first.

Run:

    uv run python examples/mistral_agents_api_smoke_test.py \\
        --key $MISTRAL_API_KEY

    # Step 3 only — needs the MCP server publicly reachable
    uv run python examples/mistral_agents_api_smoke_test.py \\
        --key $MISTRAL_API_KEY \\
        --mcp-url https://your-mcp-host.example.com/sse \\
        --mcp-token $API_KEY \\
        --kernel path/to/kernel.f90

This script **consumes tokens** on your account. Use a low-tier key for
the smoke test and rotate it afterwards if you publish the trace.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://api.mistral.ai/v1"


def _request(url: str, *, token: str, payload: dict | None = None,
             method: str = "GET", timeout: int = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(f"HTTP {exc.code} {exc.reason} — {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error: {exc.reason}") from exc


def check_models(endpoint: str, token: str) -> list[str]:
    print("→ [1/3] GET /v1/models")
    data = _request(f"{endpoint}/models", token=token)
    ids = sorted({m["id"] for m in data.get("data", [])})
    print(f"   ok — {len(ids)} models visible. Sample: {', '.join(ids[:5])}")
    for required in ("mistral-large-latest", "codestral-latest"):
        if required not in ids:
            print(f"   ⚠ {required} not in this account's catalog — pipeline will fall back.")
    return ids


def check_chat(endpoint: str, token: str, model: str) -> None:
    print(f"→ [2/3] POST /v1/chat/completions  (model={model})")
    result = _request(
        f"{endpoint}/chat/completions",
        token=token,
        method="POST",
        payload={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a Fortran HPC reviewer."},
                {"role": "user",
                 "content": "In one sentence, what does `INTENT(INOUT)` mean?"},
            ],
            "max_tokens": 60,
        },
    )
    answer = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    print(f"   ok — {usage.get('total_tokens', '?')} tokens used.")
    print(f"   reply: {answer}")


def check_agents_api(
    endpoint: str, token: str, model: str,
    mcp_url: str, mcp_token: str | None, kernel: str | None,
) -> None:
    print(f"→ [3/3] POST /v1/agents  (MCP server: {mcp_url})")
    print("   This step is a template — Mistral's Agents API schema evolves;")
    print("   adapt the payload below to the current spec from")
    print("   https://docs.mistral.ai/api/#tag/agents before relying on it.")

    payload = {
        "name": "fortran-gpu-port-smoketest",
        "model": model,
        "instructions": (
            "You port Fortran kernels to GPU. When the user gives you a "
            "path, call translate_kernel_gpu with that path and report the "
            "validation status."
        ),
        "tools": [{
            "type": "mcp",
            "server": {
                "url": mcp_url,
                **({"auth": {"type": "bearer", "token": mcp_token}}
                   if mcp_token else {}),
            },
        }],
    }
    print("   request body:")
    print(json.dumps(payload, indent=2))
    if kernel:
        print(f"\n   Next: send a message to the created agent asking it to port {kernel}.")
        print("   (Left to the user — Agents API conversation schema changes; pin to a version first.)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", required=True, help="Mistral API key")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help=f"OpenAI-compatible base URL (default: {DEFAULT_ENDPOINT})")
    p.add_argument("--model", default="mistral-large-latest",
                   help="Model to exercise in the chat / agents calls")
    p.add_argument("--skip-agents", action="store_true",
                   help="Skip step 3 (Agents API) — useful when no public MCP endpoint is up")
    p.add_argument("--mcp-url", default=None,
                   help="Public URL of the fortranspire MCP server (https://.../sse)")
    p.add_argument("--mcp-token", default=None,
                   help="Bearer token expected by the MCP server (API_KEY env var on the server)")
    p.add_argument("--kernel", default=None, help="Path to a .f90 kernel for step 3's prompt")
    args = p.parse_args()

    endpoint = args.endpoint.rstrip("/")
    check_models(endpoint, args.key)
    check_chat(endpoint, args.key, args.model)

    if args.skip_agents:
        print("→ [3/3] skipped (--skip-agents)")
        return 0
    if not args.mcp_url:
        print("→ [3/3] skipped — pass --mcp-url to exercise the Agents API + MCP wiring.")
        return 0

    check_agents_api(endpoint, args.key, args.model,
                     args.mcp_url, args.mcp_token, args.kernel)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
