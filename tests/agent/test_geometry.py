"""Geometry catalogue + decomposition proposer (issue #88).

Geometry can't be read from a kernel — it is a modelling choice — so the
domain agent is interactive: the user gives the geometry/resolution, the
stencil halo comes from the typed domain model, and the decomposition
follows. These tests pin the point-count formulas (verified against ECMWF
Atlas and standard references) and the decomposition logic.
"""
from __future__ import annotations

import pytest

from fortranspire.agent.geometry import (
    GEOMETRIES,
    identify,
    propose_decomposition,
)


class TestPointCounts:
    @pytest.mark.parametrize("name,expected", [
        ("O1280", 4 * 1280**2 + 36 * 1280),   # octahedral, Atlas formula
        ("O16", 4 * 16**2 + 36 * 16),
        ("F640", 8 * 640**2),                 # regular Gaussian
        ("nside=1024", 12 * 1024**2),         # HEALPix
        ("C768", 6 * 768**2),                 # cubed sphere
        ("R2B9", 80 * 4**9),                  # icosahedral (root 2)
        ("R2B4", 20_480),                     # ICON R2B4, a known value
    ])
    def test_count(self, name, expected):
        geom, param = identify(name)
        assert geom.count(param) == expected


class TestIdentify:
    @pytest.mark.parametrize("name,key", [
        ("O1280", "octahedral"),
        ("nside=1024", "healpix"),
        ("H512", "healpix"),
        ("C768", "cubed_sphere"),
        ("R2B9", "icosahedral"),
        ("F1280", "gaussian_regular"),
        ("0.25x0.25", "latlon"),
    ])
    def test_matches_the_right_family(self, name, key):
        geom, _ = identify(name)
        assert geom.key == key

    def test_unknown_returns_none(self):
        assert identify("wibble") is None

    def test_healpix_rejects_non_power_of_two(self):
        """HEALPix nside must be a power of two."""
        assert identify("nside=1000") is None
        assert identify("nside=1024") is not None


class TestDecomposition:
    def test_points_per_rank_and_memory(self):
        d = propose_decomposition("O1280", n_ranks=1024, halo=1, levels=137, fields=10)
        assert d.total_points == 6_599_680
        assert d.points_per_rank == pytest.approx(6_599_680 / 1024, rel=0.01)
        assert d.bytes_per_rank > 0
        assert "Atlas" in d.partitioning

    def test_halo_zero_notes_no_exchange_needed(self):
        d = propose_decomposition("C768", n_ranks=256, halo=0)
        assert d.halo == 0
        assert any("point-wise" in n or "no halo" in n for n in d.notes)

    def test_halo_grows_the_memory_estimate(self):
        no_halo = propose_decomposition("O1280", n_ranks=1024, halo=0)
        with_halo = propose_decomposition("O1280", n_ranks=1024, halo=2)
        assert with_halo.bytes_per_rank > no_halo.bytes_per_rank

    def test_unstructured_uses_a_neighbour_exchange(self):
        d = propose_decomposition("R2B9", n_ranks=2048, halo=1)
        assert "neighbour" in d.halo_exchange.lower()

    def test_unknown_resolution_returns_none(self):
        assert propose_decomposition("nope", n_ranks=10) is None

    def test_render_is_readable(self):
        d = propose_decomposition("nside=1024", n_ranks=512, halo=1)
        text = d.render()
        assert "HEALPix" in text
        assert "points / rank" in text
        assert "halo exchange" in text


class TestCatalogueCoverage:
    def test_every_family_round_trips_its_example(self):
        """Each catalogue entry's example must parse and count."""
        for g in GEOMETRIES.values():
            param = g.parse(g.example)
            assert param is not None, g.key
            assert g.count(param) > 0


class TestCliWiring:
    def test_domain_verb_is_dispatched(self):
        from fortranspire.cli import _DISPATCH

        assert _DISPATCH["domain"] == ("fortranspire.agent.cli", "_domain_main")

    def test_halo_flows_from_the_domain_model(self):
        """The stencil halo the decomposition uses is the one the typed
        model reads from the Fortran — the two features connect."""
        from fortranspire.agent.domain_model import build_domain_model

        src = ("subroutine s(t, o, n)\n  integer :: n\n"
               "  real(8), intent(in) :: t(n)\n  real(8), intent(out) :: o(n)\n"
               "  integer :: k\n  do k=2,n-1\n    o(k) = t(k+1) - t(k-1)\n  end do\n"
               "end subroutine s\n")
        halo = build_domain_model(src).max_halo
        d = propose_decomposition("O1280", n_ranks=1024, halo=halo)
        assert d.halo == 1
