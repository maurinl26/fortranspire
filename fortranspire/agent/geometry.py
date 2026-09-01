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


@dataclass
class Partition:
    """A grid-imposed decomposition: the grid decides, the software follows.

    The requested rank count is a wish; `ranks` is what the grid actually
    allows (a cubed sphere has 6 faces, so ranks are 6·nx·ny; HEALPix nested
    imposes the 12·4^d hierarchy). `exact` says whether the wish was already
    grid-compatible.
    """

    requested: int
    ranks: int
    exact: bool
    shape: str
    note: str = ""


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
    # The grid imposes the decomposition: requested ranks -> what the grid
    # actually allows. This is the dependency direction — software-defined
    # by the grid, not a free knob.
    decompose: "Callable[[int], Partition]" = None  # set below
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


# ── Grid-imposed decomposition rules ───────────────────────────────────────
# Each returns the decomposition the *grid* allows for a requested rank
# count — the software is defined by the grid, so the rank count is snapped
# to what the topology permits, not accepted blindly.

def _nearest_square(n: int) -> tuple[int, int]:
    """Factor n into the most square-ish (a, b) with a*b closest to n, a<=b."""
    best = (1, n)
    for a in range(1, int(math.isqrt(n)) + 1):
        b = round(n / a)
        if abs(a * b - n) <= abs(best[0] * best[1] - n):
            best = (a, b)
    return best


def _decompose_cubed_sphere(requested: int) -> Partition:
    # 6 faces, each tiled p×p (the natural FV3 layout is square per face),
    # so ranks are 6·p². Snap to the nearest such count.
    p = max(1, round(math.sqrt(max(requested, 6) / 6)))
    ranks = 6 * p * p
    return Partition(requested, ranks, ranks == requested,
                     f"6 faces × {p}×{p} tiles",
                     "cubed sphere: ranks must be 6·p² (square tiles per face)")


def _decompose_healpix(requested: int) -> Partition:
    # Nested: the sphere splits into 12·4^d equal regions that keep
    # neighbours local. Snap to the nearest such count.
    best_d, best = 0, 12
    d = 0
    while 12 * (4 ** d) <= max(requested, 12) * 4:
        if abs(12 * 4 ** d - requested) < abs(best - requested):
            best, best_d = 12 * 4 ** d, d
        d += 1
    return Partition(requested, best, best == requested,
                     f"nested: 12·4^{best_d} = {best} regions",
                     "HEALPix nested imposes 12·4^d — keeps neighbours local")


def _decompose_latlon(requested: int) -> Partition:
    nx, ny = _nearest_square(requested)
    ranks = nx * ny
    return Partition(requested, ranks, ranks == requested,
                     f"{nx}×{ny} blocks",
                     "lat-lon: a 2-D block factorisation of the rank count")


def _decompose_flexible(strategy: str):
    """Atlas equal-regions and METIS take (almost) any rank count."""
    def decompose(requested: int) -> Partition:
        return Partition(requested, requested, True, strategy,
                         "flexible partitioner — the grid allows this count")
    return decompose


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
        decompose=_decompose_flexible("equal-regions (Atlas)"),
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
        decompose=_decompose_latlon,
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
        decompose=_decompose_healpix,
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
        decompose=_decompose_cubed_sphere,
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
        decompose=_decompose_flexible("graph partition (METIS)"),
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
        decompose=_decompose_latlon,
    ),
}


# ── Spectral-truncation aliases ────────────────────────────────────────────
# TCo / TL / TQ are *spectral truncation* labels, not geometries — they show
# the spectral resolution and resolve to a Gaussian grid, which is what
# imposes the decomposition. So they are aliases, not a catalogue family.
#
# Verified against ECMWF: TCo1279 = O1280 (cubic octahedral, N = T+1). The
# linear grid TL<T> pairs with a reduced Gaussian N=(T+1)/2 (TL1279 = N640);
# the modern operational grid is octahedral, so the octahedral-equivalent is
# used for the estimate, flagged.
_SPECTRAL_RE = re.compile(r"^T(Co|L|Q)\s*(\d+)$", re.IGNORECASE)


def resolve_spectral(name: str) -> Optional[tuple[str, str]]:
    """A spectral truncation name → (grid resolution string, note).

    ``TCo1279`` → ``("O1280", "cubic octahedral, N=T+1")``. Returns None
    when the name is not a spectral truncation label.
    """
    m = _SPECTRAL_RE.match(name.strip())
    if not m:
        return None
    kind, t = m.group(1).lower(), int(m.group(2))
    if kind == "co":                         # cubic octahedral
        return f"O{t + 1}", f"spectral TCo{t} → octahedral O{t + 1} (cubic, N=T+1)"
    if kind == "l":                          # linear
        n = (t + 1) // 2
        return f"O{n}", (f"spectral TL{t} → linear grid ≈ reduced Gaussian N{n}; "
                         f"octahedral-equivalent O{n} used for the estimate")
    # quadratic — between the two; flagged as approximate.
    n = (3 * (t + 1)) // 4
    return f"O{n}", f"spectral TQ{t} → quadratic grid ≈ O{n} (approximate)"


def identify(name: str) -> Optional[tuple[Geometry, float]]:
    """Match a resolution string to a geometry and its parameter.

    ``"O1280"`` → (octahedral, 1280). ``"nside=1024"`` → (healpix, 1024).
    Returns None when nothing in the catalogue recognises it.
    """
    spectral = resolve_spectral(name)
    if spectral is not None:
        name = spectral[0]   # resolve to the underlying grid
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
    requested_ranks: int = 0          # what the user asked for
    ranks_exact: bool = True          # was it grid-compatible?
    partition_shape: str = ""         # the grid-imposed tiling
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        gib = self.bytes_per_rank / (1024 ** 3)
        rank_line = f"  MPI ranks             {self.n_ranks}"
        if not self.ranks_exact:
            rank_line += (f"   (requested {self.requested_ranks} — snapped to what "
                          f"the grid allows)")
        lines = [
            f"# Decomposition — {self.geometry} {self.resolution}",
            "",
            f"  horizontal points     {self.total_points:,}",
            f"  vertical levels       {self.levels}",
            rank_line,
            f"  partition             {self.partition_shape}",
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
    spectral = resolve_spectral(resolution)
    hit = identify(resolution)
    if hit is None:
        return None
    geom, param = hit

    total = geom.count(param)

    # The grid imposes the decomposition: snap the requested rank count to
    # what the topology allows (a cubed sphere has 6 faces; HEALPix nested
    # imposes 12·4^d). This is the dependency direction — software-defined
    # by the grid, not a free choice.
    partition = geom.decompose(n_ranks) if geom.decompose else \
        Partition(n_ranks, n_ranks, True, geom.partitioning)
    effective_ranks = partition.ranks
    per_rank = math.ceil(total / max(effective_ranks, 1))

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

    if spectral is not None:
        notes.append(spectral[1])
    if not partition.exact:
        notes.append(partition.note)

    return Decomposition(
        geometry=geom.name,
        resolution=resolution,
        total_points=total,
        n_ranks=effective_ranks,
        points_per_rank=per_rank,
        halo=halo,
        levels=levels,
        fields=fields,
        partitioning=geom.partitioning,
        halo_exchange=_halo_exchange(geom, halo),
        bytes_per_rank=bytes_per_rank,
        requested_ranks=n_ranks,
        ranks_exact=partition.exact,
        partition_shape=partition.shape,
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
