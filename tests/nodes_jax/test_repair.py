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
