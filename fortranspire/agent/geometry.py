"""Geometry catalogue + software-decomposition proposer (issue #88).

The typed domain model (`domain_model.py`) reads the *stencil* from Fortran
— dtypes, axes, offsets, and the **halo** those offsets imply. What it
cannot read is the **geometry**: whether the horizontal mesh is an
octahedral reduced Gaussian grid, a HEALPix pixelisation, a cubed sphere,
or an icosahedral mesh. That is a modelling choice a human makes, which is
why the domain agent is *interactive*: it asks which geometry and
resolution, then proposes the software decomposition — points per rank,
halo width, partitioning strategy, MPI halo exchange — that the geometry
and the stencil together imply.

This module is the abstraction: a catalogue of the grid families used in
NWP and climate, with their resolution nomenclature and point-count
formulas (verified against ECMWF Atlas for the Gaussian family and standard
references for the others), and a decomposition proposer that combines a
geometry + resolution + rank count + the stencil halo into a concrete plan.

Point-count formulas (N is the family's resolution parameter):

* **Octahedral reduced Gaussian** `O<N>` (ECMWF/Atlas) — points per latitude
  row are ``20 + 4*j`` from the pole, so the total is ``4*N**2 + 36*N``.
  Verified from the Atlas nx-array definition.
* **Regular Gaussian** `F<N>` — ``nx = 4N``, ``ny = 2N`` → ``8*N**2``.
* **HEALPix** `nside` (a power of two) — ``12 * nside**2`` equal-area pixels.
* **Cubed sphere** `C<N>` (GFDL/FV3) — ``6 * N**2`` cells.
* **Icosahedral** `R2B<k>` (ICON) — ``20 * n_root**2 * 4**k`` cells; for the
  common root of 2, ``80 * 4**k`` (R2B4 = 20 480, verified).
* **Regular lat-lon** `<dlon>x<dlat>` degrees — ``(360/dlon) * (180/dlat)``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Geometry:
    """One grid family: how it is named, sized, meshed and decomposed."""

    key: str
    name: str
    naming: str                       # human description of the resolution name
    example: str                      # a canonical resolution string
    count: Callable[[float], int]     # resolution parameter -> horizontal points
    parse: Callable[[str], Optional[float]]  # name -> resolution parameter
    structure: str                    # "unstructured" | "structured" | "block-structured"
    mesh_library: str                 # who provides the mesh
    partitioning: str                 # native domain-decomposition strategy
    notes: str = ""


# ── Resolution parsers ─────────────────────────────────────────────────────

def _parse_prefixed(prefix: str):
    rx = re.compile(rf"^{prefix}\s*(\d+)$", re.IGNORECASE)

    def parse(name: str) -> Optional[float]:
        m = rx.match(name.strip())
        return float(m.group(1)) if m else None
    return parse


def _parse_r2b(name: str) -> Optional[float]:
    m = re.match(r"^R2B\s*(\d+)$", name.strip(), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_healpix(name: str) -> Optional[float]:
    m = re.match(r"^(?:H|nside=?)\s*(\d+)$", name.strip(), re.IGNORECASE)
    if not m:
        return None
    nside = int(m.group(1))
    # nside must be a power of two.
    return float(nside) if nside > 0 and (nside & (nside - 1)) == 0 else None


def _parse_latlon(name: str) -> Optional[float]:
    m = re.match(r"^([\d.]+)\s*[x×]\s*([\d.]+)$", name.strip())
    if m:
        # Encode as dlon; dlat assumed equal for the count unless given.
        return float(m.group(1))
    return None


# ── The catalogue ──────────────────────────────────────────────────────────

GEOMETRIES: dict[str, Geometry] = {
    "octahedral": Geometry(
        key="octahedral",
        name="Octahedral reduced Gaussian",
        naming="O<N> — N Gaussian latitudes pole→equator (O1280 ≈ 9 km)",
        example="O1280",
        count=lambda N: int(4 * N * N + 36 * N),
        parse=_parse_prefixed("O"),
        structure="unstructured",
        mesh_library="ECMWF Atlas",
        partitioning="equal-regions (Atlas), one-element halo",
        notes="ECMWF IFS / FVM native grid; co-located with the spectral grid.",
    ),
    "gaussian_regular": Geometry(
        key="gaussian_regular",
        name="Regular Gaussian",
        naming="F<N> — nx=4N, ny=2N",
        example="F1280",
        count=lambda N: int(8 * N * N),
        parse=_parse_prefixed("F"),
        structure="structured",
        mesh_library="ECMWF Atlas",
        partitioning="latitude bands, or 2-D checkerboard",
    ),
    "healpix": Geometry(
        key="healpix",
        name="HEALPix",
        naming="nside (power of 2) — 12·nside² equal-area pixels (nside=1024 ≈ 6 km)",
        example="nside=1024",
        count=lambda nside: int(12 * nside * nside),
        parse=_parse_healpix,
        structure="unstructured",
        mesh_library="healpy / Atlas",
        partitioning="ring or nested; nested gives a quad-tree partition",
        notes="Equal-area, iso-latitude; ECMWF/nextGEMS output grid of choice.",
    ),
    "cubed_sphere": Geometry(
        key="cubed_sphere",
        name="Cubed sphere",
        naming="C<N> — N cells per face edge, 6·N² cells (C768 ≈ 13 km)",
        example="C768",
        count=lambda N: int(6 * N * N),
        parse=_parse_prefixed("C"),
        structure="block-structured",
        mesh_library="FV3 / GFDL",
        partitioning="6 faces × (i,j) tiles; ranks per face",
        notes="GFDL FV3, SHiELD, Pace. Structured within each face.",
    ),
    "icosahedral": Geometry(
        key="icosahedral",
        name="Icosahedral (triangular)",
        naming="R2B<k> — root 2, k bisections; 80·4^k cells (R2B9 ≈ 5 km)",
        example="R2B9",
        count=lambda k: int(80 * (4 ** int(k))),
        parse=_parse_r2b,
        structure="unstructured",
        mesh_library="ICON grid generator",
        partitioning="graph partition (METIS) or hierarchical; halo of 1–2 rows",
        notes="DWD/MPI ICON. Cell/Edge/Vertex connectivity.",
    ),
    "latlon": Geometry(
        key="latlon",
        name="Regular lat-lon",
        naming="<dlon>x<dlat> degrees",
        example="0.25x0.25",
        count=lambda dlon: int((360.0 / dlon) * (180.0 / dlon)),
        parse=_parse_latlon,
        structure="structured",
        mesh_library="—",
        partitioning="2-D block decomposition",
    ),
}


def identify(name: str) -> Optional[tuple[Geometry, float]]:
    """Match a resolution string to a geometry and its parameter.

    ``"O1280"`` → (octahedral, 1280). ``"nside=1024"`` → (healpix, 1024).
    Returns None when nothing in the catalogue recognises it.
    """
    for geom in GEOMETRIES.values():
        param = geom.parse(name)
        if param is not None:
            return geom, param
    return None


# ── Decomposition proposer ─────────────────────────────────────────────────

@dataclass
class Decomposition:
    """A proposed software decomposition for a geometry + stencil + ranks."""

    geometry: str
    resolution: str
    total_points: int
    n_ranks: int
    points_per_rank: int
    halo: int
    levels: int
    fields: int
    partitioning: str
    halo_exchange: str
    bytes_per_rank: int
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        gib = self.bytes_per_rank / (1024 ** 3)
        lines = [
            f"# Decomposition — {self.geometry} {self.resolution}",
            "",
            f"  horizontal points     {self.total_points:,}",
            f"  vertical levels       {self.levels}",
            f"  MPI ranks             {self.n_ranks}",
            f"  points / rank         ~{self.points_per_rank:,}",
            f"  stencil halo          {self.halo} row(s)",
            f"  fields (3-D)          {self.fields}",
            f"  memory / rank         ~{gib:.2f} GiB  (fields × levels × 8 B, + halo)",
            "",
            f"  partitioning          {self.partitioning}",
            f"  halo exchange         {self.halo_exchange}",
        ]
        for n in self.notes:
            lines.append(f"  note                  {n}")
        return "\n".join(lines) + "\n"


def _halo_exchange(geom: Geometry, halo: int) -> str:
    if geom.structure == "unstructured":
        return (f"MPI neighbour exchange over the partition boundary "
                f"({halo}-cell halo, {geom.mesh_library} handles it)")
    if geom.structure == "block-structured":
        return f"face-edge + corner exchange, {halo}-cell halo per tile"
    return f"2-D halo exchange (N/S/E/W), {halo}-cell halo"


def propose_decomposition(
    resolution: str,
    *,
    n_ranks: int,
    halo: int = 0,
    levels: int = 137,
    fields: int = 10,
) -> Optional[Decomposition]:
    """Propose a decomposition from a resolution string and the stencil halo.

    ``halo`` comes from the typed domain model (the stencil offsets);
    ``levels`` and ``fields`` size the memory estimate. Returns None when the
    resolution is not in the catalogue.
    """
    hit = identify(resolution)
    if hit is None:
        return None
    geom, param = hit

    total = geom.count(param)
    per_rank = math.ceil(total / max(n_ranks, 1))

    # Memory: an interior tile plus its halo. For an unstructured/quasi-2-D
    # partition the halo scales with the perimeter ~ sqrt(per_rank); a
    # structured tile similarly. A coarse but honest estimate.
    perimeter = math.ceil(math.sqrt(max(per_rank, 1)))
    halo_points = 4 * perimeter * halo if halo else 0
    bytes_per_rank = (per_rank + halo_points) * levels * fields * 8

    notes: list[str] = []
    if halo == 0:
        notes.append("point-wise (no stencil) — no halo needed")
    if geom.structure == "unstructured":
        notes.append("column physics stays structured on the vertical; the "
                     "mesh is horizontal-only")
    if geom.key == "healpix" and halo:
        notes.append("nested ordering keeps neighbours local — good for the halo")

    return Decomposition(
        geometry=geom.name,
        resolution=resolution,
        total_points=total,
        n_ranks=n_ranks,
        points_per_rank=per_rank,
        halo=halo,
        levels=levels,
        fields=fields,
        partitioning=geom.partitioning,
        halo_exchange=_halo_exchange(geom, halo),
        bytes_per_rank=bytes_per_rank,
        notes=notes,
    )


def catalogue_table() -> str:
    """Render the geometry catalogue as a reference table."""
    rows = ["| Geometry | Naming | Points | Structure | Partitioning |",
            "| -------- | ------ | ------ | --------- | ------------ |"]
    for g in GEOMETRIES.values():
        rows.append(f"| {g.name} | `{g.example}` | {g.count(g.parse(g.example)):,} "
                    f"| {g.structure} | {g.partitioning} |")
    return "\n".join(rows)
