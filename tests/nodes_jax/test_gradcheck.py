"""Tests for the gradient check (issue #73).

The Phase 2 validation that shipped before this node proved the emitted
code *traced*. These tests pin the difference between that and proving it
is differentiable — and, just as importantly, pin the one class of defect
finite differences provably cannot see, so nobody reads a pass here as
more than it is.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

from fortranspire.agent.nodes_jax.gradcheck import (  # noqa: E402
    _extent,
    check_kernel,
    gradcheck_agent,
)


def kernel(name: str = "k", inputs=("x",), dims=None) -> dict:
    return {
        "routine_name": name,
        "inputs": list(inputs),
        "outputs": ["y"],
        "intent_map": {},
        "dimensions": dims if dims is not None else {"x": ["8"]},
    }


def compile_fn(src: str, name: str = "k"):
    namespace: dict = {}
    exec(src, namespace)  # noqa: S102 - test fixture
    return namespace[name]


class TestCatchesRealDefects:
    def test_correct_kernel_passes(self):
        fn = compile_fn("""
import jax.numpy as jnp
def k(x, a):
    return jnp.sin(x) * a + x ** 2
""")
        report = check_kernel(fn, kernel(inputs=("x", "a")))
        assert report["status"] == "pass"
        assert report["max_abs_err"] < 1e-6

    def test_nan_gradient_from_an_unguarded_where(self):
        """The classic translated-Fortran defect.

        `jnp.where` evaluates *both* branches, so a `sqrt` guarded by an
        `IF` in the Fortran produces NaN gradients while the forward values
        look perfectly fine. A tracing check sees nothing wrong.
        """
        fn = compile_fn("""
import jax.numpy as jnp
def k(x):
    return jnp.sum(jnp.where(x > 0.0, jnp.sqrt(x), 0.0))
""")
        report = check_kernel(fn, kernel())
        assert report["status"] == "fail"
        assert any(f["kind"] == "non-finite" for f in report["failures"])

    def test_guarded_where_is_the_fix_and_passes(self):
        """Making the unsafe branch safe first — what the prompt instructs."""
        fn = compile_fn("""
import jax.numpy as jnp
def k(x):
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.sum(jnp.where(x > 0.0, jnp.sqrt(safe), 0.0))
""")
        assert check_kernel(fn, kernel())["status"] == "pass"

    def test_detached_gradient_is_caught(self):
        """`stop_gradient` — or a mutation that silently detaches."""
        fn = compile_fn("""
import jax, jax.numpy as jnp
def k(x):
    return jnp.sum(jax.lax.stop_gradient(x) ** 2)
""")
        report = check_kernel(fn, kernel())
        assert report["status"] == "fail"
        assert any(f["kind"] == "mismatch" for f in report["failures"])

    def test_python_branch_survives_grad_but_fails_jit(self):
        """Why the node checks `jit` and not only `grad`.

        Outside `jit`, reverse-mode traces with a tracer that carries a
        concrete primal, so `if x.sum() > 0` resolves against it and the
        gradient comes out correct. The kernel still raises the first time
        anyone jits it — which is the whole reason to be in JAX. Checking
        `grad` alone would pass this.
        """
        fn = compile_fn("""
import jax.numpy as jnp
def k(x):
    if x.sum() > 0:      # Python branch on a traced value
        return jnp.sum(x)
    return jnp.sum(-x)
""")
        report = check_kernel(fn, kernel())
        assert report["status"] == "fail"
        assert any(f["kind"] == "jit-failed" for f in report["failures"])

    def test_where_is_the_jit_safe_form(self):
        fn = compile_fn("""
import jax.numpy as jnp
def k(x):
    return jnp.where(x.sum() > 0, jnp.sum(x), jnp.sum(-x))
""")
        report = check_kernel(fn, kernel())
        assert report["status"] == "pass"
        assert report["jit"] is True


class TestDocumentedLimitation:
    def test_locally_flat_function_passes_despite_being_useless(self):
        """Pinned on purpose: this is what the check *cannot* see.

        `floor` severs the gradient chain, but finite differences of a
        locally flat function are also zero, so both sides agree. The
        module docstring says so; this test makes sure that claim stays
        true and is not quietly assumed away.
        """
        fn = compile_fn("""
import jax.numpy as jnp
def k(x):
    return jnp.sum(jnp.floor(x * 1000.0))
""")
        report = check_kernel(fn, kernel())
        assert report["status"] == "pass"
        assert report["max_abs_err"] == 0.0


class TestInputSynthesis:
    @pytest.mark.parametrize(
        "dim,expected",
        [("8", 8), ("1:16", 16), ("n", 8), ("0:nx-1", 8), ("999", 64)],
    )
    def test_extent_resolution(self, dim, expected):
        """Symbolic bounds fall back; literals are honoured but capped."""
        assert _extent(dim) == expected

    def test_integer_extent_arguments_are_not_differentiated(self):
        """`n` is a loop bound — differentiating it is meaningless."""
        fn = compile_fn("""
import jax.numpy as jnp
def k(x, n):
    return jnp.sum(x[:n] ** 2)
""")
        report = check_kernel(fn, kernel(inputs=("x", "n"), dims={"x": ["8"]}))
        assert "n" in report["skipped_args"]
        assert "x" in report["checked_args"]


class TestNodeContract:
    def test_failure_is_blocking(self):
        state = {
            "kernel_results": [{
                **kernel(),
                "jax_code": "import jax, jax.numpy as jnp\n"
                            "def k(x):\n    return jnp.sum(jax.lax.stop_gradient(x) ** 2)\n",
                "status": "pending",
            }],
            "executed_agents": [],
        }
        out = gradcheck_agent(state)
        assert out["gradcheck_passed"] is False
        assert out["kernel_results"][0]["status"] == "error"
        assert out["gradcheck_log"]

    def test_success_marks_the_kernel(self):
        state = {
            "kernel_results": [{
                **kernel(),
                "jax_code": "import jax.numpy as jnp\ndef k(x):\n    return jnp.sum(x ** 2)\n",
                "status": "pending",
            }],
            "executed_agents": [],
        }
        out = gradcheck_agent(state)
        assert out["gradcheck_passed"] is True
        assert out["kernel_results"][0]["status"] == "success"

    def test_kernel_blocked_upstream_is_not_checked(self):
        state = {
            "kernel_results": [{**kernel(), "status": "skipped", "jax_code": ""}],
            "executed_agents": [],
        }
        out = gradcheck_agent(state)
        assert out["gradcheck_passed"] is True
        assert out["kernel_results"][0]["status"] == "skipped"

    def test_uncompilable_emission_fails_rather_than_crashing(self):
        state = {
            "kernel_results": [{**kernel(), "jax_code": "def k(x:\n", "status": "pending"}],
            "executed_agents": [],
        }
        out = gradcheck_agent(state)
        assert out["gradcheck_passed"] is False


class TestTypeAwareInputs:
    """Issue #4 — integer args typed from the AST, not a name whitelist."""

    def test_integer_bound_is_not_differentiated_and_kernel_passes(self):
        # `n` is an integer bound; a float would break `jnp.arange(n)`.
        fn = compile_fn("""
import jax.numpy as jnp
def k(n, x):
    return x * jnp.arange(n, dtype=jnp.float64)
""")
        k = {**kernel(inputs=("n", "x")),
             "arg_dtypes": {"n": "integer", "x": "real"},
             "dimensions": {"x": ["8"]}}
        report = check_kernel(fn, k)
        assert report["status"] == "pass"
        assert "n" in report["skipped_args"]     # integer → not differentiated
        assert "x" in report["checked_args"]

    def test_unresolved_index_table_reports_needs_fixture_not_a_crash(self):
        # `idx` is a PROMOTED integer table of unknown shape (no dims, no
        # resolution) — gradcheck cannot know its rank, so it must report a
        # fixture, not fabricate or crash.
        k = {**kernel(inputs=("idx", "x")),
             "arg_dtypes": {"idx": "integer", "x": "real"},
             "index_args": ["idx"],
             "free_reads": ["idx"],           # module state, not a declared arg
             "dimensions": {"x": ["8"]}}       # idx shape unknown
        report = check_kernel(lambda **kw: 0.0, k)
        assert report["status"] == "needs_fixture"
        assert "idx" in report["fixture_args"]

    def test_needs_fixture_is_unverified_but_not_blocking(self):
        state = {
            "kernel_results": [{
                **kernel(inputs=("idx", "x")),
                "arg_dtypes": {"idx": "integer", "x": "real"},
                "index_args": ["idx"],
                "free_reads": ["idx"],
                "dimensions": {"x": ["8"]},
                "jax_code": "def k(idx, x):\n    return x[idx]\n",
                "status": "pending",
            }],
            "executed_agents": [],
        }
        out = gradcheck_agent(state)
        assert out["gradcheck_passed"] is True            # not a refutation
        assert out["gradcheck_unverified"]                # but flagged unverified
        assert out["kernel_results"][0]["status"] == "needs_fixture"

    def test_resolved_index_table_is_synthesised_and_verified(self):
        # With the table's shape resolved cross-module, gradcheck builds a valid
        # degenerate fixture and actually checks the gradient (issue #99).
        fn = compile_fn("""
import jax.numpy as jnp
def k(idx, coef, x):
    return coef * x[idx]
""")
        k = {**kernel(inputs=("idx", "coef", "x")),
             "arg_dtypes": {"idx": "integer", "coef": "real", "x": "real"},
             "index_args": ["idx"],
             "free_reads": ["idx", "coef"],
             "resolved": {"idx": {"dtype": "integer", "rank": 1},
                          "coef": {"dtype": "real", "rank": 1}},
             "dimensions": {"x": ["8"]}}
        report = check_kernel(fn, k)
        assert report["status"] == "pass"
        assert "coef" in report["checked_args"] and "x" in report["checked_args"]
        assert "idx" in report["skipped_args"]
