"""Regression tests for the extractor's INTENT-declaration parser.

Surfaced by the first real end-to-end CMAQ port (RBFEVAL → JAX): a
declaration like ``INTENT(IN) :: rki(numcells,nrxns), yin(numcells,ischan)``
was split on *every* comma, including those inside the array dimension
``(...)``. The argument name became ``rki(numcells``, which then propagated
into the derived JAX signature and the emitted kernel's keyword arguments,
so gradcheck failed with ``unexpected keyword argument 'rki(numcells'``.

The parser must (1) split entities on top-level commas only, and (2) key
the INTENT map by the bare argument name, not its shape.
"""
import pytest

from fortranspire.agent.nodes.extractor import _split_entities, _entity_name


@pytest.mark.parametrize(
    "decl, expected",
    [
        # the CMAQ RBFEVAL case: two multi-dim arrays, one list
        ("rki(numcells,nrxns), yin(numcells,ischan)", ["rki", "yin"]),
        # assumed-shape arrays with spaces around the colons
        ("YIN(  :, : ), YDOT( :, : )", ["YIN", "YDOT"]),
        # a scalar, a 2-D array, a 1-D array — mixed
        ("a(n,m), b, c(3)", ["a", "b", "c"]),
        # a bare scalar
        ("NCSP", ["NCSP"]),
        # initialisers must not leak into the name
        ("x = 0.0, y", ["x", "y"]),
    ],
)
def test_entities_split_on_top_level_commas(decl, expected):
    names = [_entity_name(e) for e in _split_entities(decl)]
    assert names == expected


def test_dimension_comma_is_not_a_separator():
    """The exact failure mode: an inner comma must never make a second name."""
    entities = _split_entities("rki(numcells,nrxns)")
    assert len(entities) == 1
    assert _entity_name(entities[0]) == "rki"
