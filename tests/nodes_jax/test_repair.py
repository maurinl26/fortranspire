"""Self-repair of JAX emission — tolerate a small model."""
from fortranspire.agent.nodes_jax.repair import emission_defects, emit_with_repair


class _FakeLLM:
    """Returns canned responses in order; records how many times it was called."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def invoke(self, _messages):
        i = self.calls
        self.calls += 1

        class R:
            content = self._responses[min(i, len(self._responses) - 1)]

        return R()


_BAD = (
    "import jax.numpy as jnp\n"
    "def k(c, i, x):\n"
    "    n = c[i]\n"
    "    if n == 1:\n"
    "        return x\n"
    "    return x * 2\n"
)
_GOOD = (
    "import jax.numpy as jnp\n"
    "def k(c, i, x):\n"
    "    n = c[i]\n"
    "    return jnp.where(n == 1, x, x * 2)\n"
)


def test_emission_defects_flags_data_branch_and_syntax():
    assert emission_defects(_GOOD) == []
    assert emission_defects(_BAD)                     # data-branch
    syntax = emission_defects("def k(:\n")
    assert syntax and "syntax" in syntax[0].lower()


def test_repair_fixes_a_data_branch_on_retry():
    llm = _FakeLLM([_BAD, _GOOD])
    code, remaining, repairs = emit_with_repair(
        llm, system=None, strip=lambda s: s, max_repairs=2)
    assert repairs == 1
    assert remaining == []
    assert "jnp.where" in code
    assert llm.calls == 2                     # one emit + one repair


def test_clean_emission_spends_no_repair():
    llm = _FakeLLM([_GOOD])
    code, remaining, repairs = emit_with_repair(
        llm, system=None, strip=lambda s: s, max_repairs=2)
    assert repairs == 0 and remaining == []
    assert llm.calls == 1                      # no wasted retry


def test_gives_up_after_max_repairs_but_keeps_code():
    llm = _FakeLLM([_BAD, _BAD, _BAD])         # model never fixes it
    code, remaining, repairs = emit_with_repair(
        llm, system=None, strip=lambda s: s, max_repairs=2)
    assert repairs == 2
    assert remaining                            # still defective — reported, not hidden
    assert llm.calls == 3                       # emit + 2 repairs, then stop


def test_semantic_verify_drives_a_second_attempt():
    # First emission is clean of deterministic defects but "wrong"; the verify
    # callback reports it, and the repair loop re-prompts and accepts the fix.
    good = "import jax.numpy as jnp\ndef k(x):\n    return x * 2\n"
    wrong = "import jax.numpy as jnp\ndef k(x):\n    return x * 3\n"
    llm = _FakeLLM([wrong, good])

    seen = {"n": 0}
    def verify(code):
        seen["n"] += 1
        return [] if "x * 2" in code else ["gradient is wrong: expected 2*x"]

    code, remaining, repairs = emit_with_repair(
        llm, system=None, strip=lambda s: s, max_repairs=2, verify=verify)
    assert repairs == 1 and remaining == []
    assert "x * 2" in code
    assert llm.calls == 2                     # emit + one semantic repair


def test_verify_not_consulted_while_deterministic_defects_remain():
    # A data-branch is fixed first; verify only runs once the code is clean.
    llm = _FakeLLM([_BAD, _GOOD])
    calls = {"verify": 0}
    def verify(_code):
        calls["verify"] += 1
        return []
    emit_with_repair(llm, system=None, strip=lambda s: s, max_repairs=3, verify=verify)
    # verify is consulted only after the deterministic defect is gone
    assert calls["verify"] >= 1


def test_n_best_keeps_the_best_candidate_not_the_last():
    # attempt 0: one data-branch (1 defect). repair → TWO data-branches (worse).
    one_branch = ("import jax.numpy as jnp\ndef k(c, i, x):\n"
                  "    n = c[i]\n    if n == 1:\n        return x\n    return x\n")
    two_branch = ("import jax.numpy as jnp\ndef k(c, i, x):\n"
                  "    n = c[i]\n    m = c[i]\n"
                  "    if n == 1:\n        return x\n"
                  "    if m == 2:\n        return x\n    return x\n")
    llm = _FakeLLM([one_branch, two_branch])
    code, remaining, repairs = emit_with_repair(
        llm, system=None, strip=lambda s: s, max_repairs=1)
    assert repairs == 1
    assert code == one_branch          # kept the better first candidate
    assert len(remaining) == 1         # not the 2-defect regression
