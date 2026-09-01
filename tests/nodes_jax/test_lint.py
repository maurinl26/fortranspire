"""Static lint: a Python branch on array-derived (traced) data."""
from fortranspire.agent.nodes_jax.lint import (
    data_dependent_branches,
    data_dependent_loops,
    disallowed_imports,
    missing_imports,
)


def test_missing_import_of_used_module_root():
    # uses `jax.lax` but only imported `jnp`
    code = "import jax.numpy as jnp\ndef k(x):\n    return jax.lax.switch(0, [lambda: x], x)\n"
    assert missing_imports(code) == [{"name": "jax"}]


def test_all_used_roots_imported_is_clean():
    code = ("import jax\nimport jax.numpy as jnp\n"
            "def k(x):\n    return jax.lax.switch(0, [lambda: x], jnp.sum(x))\n")
    assert missing_imports(code) == []


def test_disallowed_import_of_a_fortran_module():
    code = "from RXNS_FUNCTION import SPECIAL_RATES\nimport jax.numpy as jnp\ndef k(x):\n    return x\n"
    hits = disallowed_imports(code)
    assert len(hits) == 1 and hits[0]["module"] == "RXNS_FUNCTION"


def test_allowed_imports_are_not_flagged():
    code = ("from __future__ import annotations\nimport jax.numpy as jnp\n"
            "from jax import lax\nfrom fortranspire.jax_smooth import smooth_max\n"
            "def k(x):\n    return x\n")
    assert disallowed_imports(code) == []


def test_data_dependent_range_loop_is_flagged():
    code = ("def k(nuserat, ncsp, x):\n"
            "    for i in range(nuserat[ncsp]):\n"
            "        x = x + i\n"
            "    return x\n")
    hits = data_dependent_loops(code)
    assert len(hits) == 1
    assert "range(nuserat[ncsp])" in hits[0]["snippet"]


def test_static_range_loop_is_not_flagged():
    code = "def k(n, x):\n    for i in range(4):\n        x = x + i\n    return x\n"
    assert data_dependent_loops(code) == []


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
