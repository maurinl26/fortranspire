"""Interactive domain agent — MCP tools (issue #88).

Geometry is a modelling choice a kernel does not carry, so the agent is
interactive: the tools surface what is missing at each step (which geometry?
how many ranks?) and the LLM asks the user. The tools themselves are
deterministic — no LLM, no token — so they are safe on the public surface.
"""
from __future__ import annotations

from fortranspire.server import domain_decomposition, domain_geometries

KERNEL = """subroutine diff(t, o, n)
  integer, intent(in) :: n
  real(8), intent(in) :: t(n)
  real(8), intent(out) :: o(n)
  integer :: k
  do k=2,n-1
     o(k) = t(k+1) - t(k-1)
  end do
end subroutine diff
"""


class TestCatalogue:
    def test_lists_the_families(self):
        text = domain_geometries()
        for family in ("Octahedral", "HEALPix", "Cubed sphere", "Icosahedral"):
            assert family in text

    def test_tells_the_agent_to_ask(self):
        assert "Ask the user" in domain_geometries()


class TestInteractiveNudges:
    def test_no_geometry_returns_the_catalogue_and_asks(self):
        r = domain_decomposition(source=KERNEL)
        assert "Ask the user which geometry" in r
        assert "Octahedral" in r  # the catalogue, to choose from

    def test_unknown_geometry_is_named_and_reprompted(self):
        r = domain_decomposition(resolution="wibble", n_ranks=100)
        assert "not a known grid" in r
        assert "Ask the user" in r

    def test_geometry_but_no_ranks_reads_halo_and_asks_for_ranks(self):
        r = domain_decomposition(resolution="O1280", source=KERNEL)
        assert "halo 1 read" in r
        assert "how many MPI ranks" in r


class TestProposal:
    def test_full_call_proposes_and_reads_halo(self):
        r = domain_decomposition(resolution="O1280", n_ranks=1024, source=KERNEL)
        assert "halo 1 read" in r
        assert "Octahedral reduced Gaussian O1280" in r
        assert "points / rank" in r

    def test_grid_imposes_the_rank_count(self):
        r = domain_decomposition(resolution="C768", n_ranks=1000)
        assert "1014" in r  # snapped to 6·13²
        assert "snapped to what the grid allows" in r

    def test_no_source_means_no_halo(self):
        r = domain_decomposition(resolution="O1280", n_ranks=512)
        assert "stencil halo          0" in r


class TestSafety:
    def test_tools_are_deterministic_no_llm(self):
        import inspect

        from fortranspire import server

        for name in ("domain_geometries", "domain_decomposition"):
            src = inspect.getsource(getattr(server, name))
            assert "get_llm" not in src and "_get_agent" not in src

    def test_registered_in_the_canonical_surface(self):
        from fortranspire.server import _TOOL_NAMES

        assert "domain_geometries" in _TOOL_NAMES
        assert "domain_decomposition" in _TOOL_NAMES
