"""Detect Fortran compilers available on the host and their OpenACC capability.

Used by `agent-analyze` to:
  - confirm the user has at least one Fortran compiler to validate generated code,
  - flag mismatches between the source (uses `!$acc` pragmas) and the
    available toolchain (no OpenACC-capable compiler installed).

Detection is cheap: `shutil.which` + a single `<compiler> --version` per
candidate (~100 ms total). Disable with `--no-toolchain-check`.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Literal

# OpenACC capability per compiler family.
#   native       — first-class OpenACC, including GPU offload
#   experimental — supports the spec but with caveats (limited targets,
#                  older versions, or partial coverage)
#   unsupported  — the vendor does not implement OpenACC at all
#   unknown      — compiler present but capability could not be determined
Capability = Literal["native", "experimental", "unsupported", "unknown"]


@dataclass(frozen=True)
class CompilerInfo:
    name: str            # "gfortran", "nvfortran", "ifx", ...
    family: str          # "gnu", "nvidia", "intel-llvm", "intel-classic", "llvm", "lfortran"
    path: str
    version: str | None
    openacc: Capability
    openacc_flag: str | None    # e.g. "-acc", "-fopenacc"

    def to_dict(self) -> dict:
        return asdict(self)


# (binary name, family, OpenACC capability, flag) — order = priority for
# the "recommended OpenACC compiler" pick.
_CANDIDATES: tuple[tuple[str, str, Capability, str | None], ...] = (
    ("nvfortran", "nvidia",        "native",       "-acc"),
    ("pgfortran", "nvidia",        "native",       "-acc"),    # legacy PGI alias
    ("gfortran",  "gnu",           "experimental", "-fopenacc"),
    ("ifx",       "intel-llvm",    "unsupported",  None),
    ("ifort",     "intel-classic", "unsupported",  None),
    ("flang-new", "llvm",          "unsupported",  None),
    ("flang",     "llvm",          "unsupported",  None),
    ("lfortran",  "lfortran",      "unsupported",  None),
)


_VERSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "gfortran":  re.compile(r"GNU Fortran[^\d]*([\d.]+)"),
    "nvfortran": re.compile(r"nvfortran[^\d]*([\d.]+)"),
    "pgfortran": re.compile(r"pgfortran[^\d]*([\d.]+)"),
    "ifx":       re.compile(r"IFX[^\d]*([\d.]+)", re.IGNORECASE),
    "ifort":     re.compile(r"ifort[^\d]*([\d.]+)", re.IGNORECASE),
    "flang-new": re.compile(r"flang[^\d]*([\d.]+)"),
    "flang":     re.compile(r"flang[^\d]*([\d.]+)"),
    "lfortran":  re.compile(r"LFortran[^\d]*([\d.]+)", re.IGNORECASE),
}


def _query_version(binary_path: str, name: str) -> str | None:
    try:
        out = subprocess.run(
            [binary_path, "--version"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    blob = (out.stdout or "") + "\n" + (out.stderr or "")
    pattern = _VERSION_PATTERNS.get(name)
    if pattern:
        match = pattern.search(blob)
        if match:
            return match.group(1)
    # Fallback: first version-looking token in the output.
    fallback = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", blob)
    return fallback.group(1) if fallback else None


def _refine_openacc(name: str, family: str, version: str | None,
                    base: Capability) -> Capability:
    """Tighten the per-family default with version-based heuristics."""
    if version is None:
        return base
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return base

    if name == "gfortran":
        # `-fopenacc` available since GCC 5; usable in production since GCC 7
        # (PTX offload). Older versions are present but effectively unsupported.
        if major >= 11:
            return "experimental"
        if major < 7:
            return "unsupported"
        return "experimental"

    if name in ("nvfortran", "pgfortran"):
        # NVIDIA HPC SDK 20.x and PGI 19.x both predate Ampere offload; flag
        # as experimental rather than native if very old.
        if major < 20:
            return "experimental"
        return "native"

    return base


def detect_compilers() -> list[CompilerInfo]:
    """Probe PATH for known Fortran compilers. Returns at most one entry per binary."""
    seen_paths: set[str] = set()
    results: list[CompilerInfo] = []

    for name, family, base_capability, flag in _CANDIDATES:
        path = shutil.which(name)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)

        version = _query_version(path, name)
        openacc = _refine_openacc(name, family, version, base_capability)

        results.append(CompilerInfo(
            name=name,
            family=family,
            path=path,
            version=version,
            openacc=openacc,
            openacc_flag=flag,
        ))
    return results


def best_openacc_compiler(compilers: list[CompilerInfo]) -> CompilerInfo | None:
    """Return the strongest OpenACC-capable compiler, or None."""
    ranking = {"native": 3, "experimental": 2, "unknown": 1, "unsupported": 0}
    capable = [c for c in compilers if c.openacc in ("native", "experimental")]
    if not capable:
        return None
    capable.sort(key=lambda c: ranking[c.openacc], reverse=True)
    return capable[0]


def summarize(compilers: list[CompilerInfo]) -> str:
    """Human-readable toolchain summary used by `agent-analyze --format text`."""
    if not compilers:
        return "Toolchain: no Fortran compiler detected on PATH."

    lines = ["Toolchain:"]
    for c in compilers:
        v = c.version or "version?"
        openacc = c.openacc
        if c.openacc_flag and openacc in ("native", "experimental"):
            openacc = f"{openacc} ({c.openacc_flag})"
        lines.append(f"  {c.name:<10} {v:<12} family={c.family:<14} openacc={openacc}")

    pick = best_openacc_compiler(compilers)
    if pick:
        lines.append(f"  → recommended for GPU port: {pick.name} {pick.version or ''}".rstrip())
    else:
        lines.append("  → no OpenACC-capable compiler found.")
    return "\n".join(lines)
