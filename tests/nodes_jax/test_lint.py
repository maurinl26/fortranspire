"""Static lint: a Python branch on array-derived (traced) data."""
from fortranspire.agent.nodes_jax.lint import data_dependent_branches


def test_flags_if_on_scalar_read_from_array():
    code = """
import jax.numpy as jnp
def k(count, x, i):
    n = count[i]
    if n == 1:
        return x
    return x * 2
"""
    hits = data_dependent_branches(code)
    assert len(hits) == 1
    assert hits[0]["names"] == ["n"]
    assert "n == 1" in hits[0]["snippet"]


def test_flags_multiway_switch_once_each():
    code = """
def k(nreact, i, a, b):
    n = nreact[i]
    if n == 1:
        return a
    elif n == 2:
        return b
    return a + b
"""
    # the `elif` is a nested If → a second distinct line
    lines = {h["line"] for h in data_dependent_branches(code)}
    assert len(lines) == 2


def test_ignores_branch_on_a_plain_scalar_argument():
    # `n` is a scalar parameter (a bound/flag), not read from an array.
    code = """
import jax.numpy as jnp
def k(n, x):
    if n > 0:
        return jnp.where(x > 0, x, 0.0)
    return x
"""
    assert data_dependent_branches(code) == []


def test_ignores_branch_on_array_shape():
    code = """
def k(x):
    s = x.shape[0]
    if s > 1:
        return x
    return x
"""
    assert data_dependent_branches(code) == []


def test_no_hits_on_clean_vectorised_kernel():
    code = """
import jax.numpy as jnp
def k(a, b):
    return jnp.where(a > b, a, b)
"""
    assert data_dependent_branches(code) == []


def test_malformed_code_does_not_raise():
    assert data_dependent_branches("def k(:\n") == []
