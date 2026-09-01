"""Guards and smooth relaxations for differentiable Fortran ports (issue #73).

Generated JAX kernels import from here rather than re-deriving these
expressions each run. Three reasons, in order of how much they hurt when
ignored: the numerically stable form is easy to get wrong (a literal
``log(exp(b*a) + exp(b*b))`` overflows for ``b*a > 709`` in float64, while
``logaddexp`` does not); the limit as the parameter tightens is worth
testing once rather than per generation; and a named import makes it
visible to a reviewer *where* the model was relaxed and with which
parameter.

Two kinds of function live here and they must not be confused.

**Guards** (:func:`safe_sqrt`, :func:`safe_divide`, :func:`safe_log`,
:func:`where_guarded`) repair a translation. They leave the forward values
alone wherever the original was defined, and only stop a NaN or an
infinite derivative appearing in the *untaken* branch of a ``where``.
Applying one is not a modelling decision.

**Relaxations** (:func:`smooth_max`, :func:`smooth_abs`, :func:`smooth_step`
and friends) change what the code computes. ``MAX(a, b)`` replaced by a
softmax no longer returns ``max(a, b)``; it returns something that tends to
it as ``beta`` grows. That is exactly right for an adjoint or an
optimisation loop, and wrong for a flux limiter that has to stay TVD.
Every one of them takes an explicit parameter, converges to the hard form
in a documented limit, and carries its bias in the docstring.

The convergence claims are pinned by tests, not asserted here.
"""
from __future__ import annotations

from typing import Any, NamedTuple

# Defaults are deliberately loose. A caller who has not thought about the
# parameter should get a visibly smoothed answer rather than one that
# silently pretends to be the hard form.
DEFAULT_BETA = 50.0   # sharpness for max/min/step families: larger = harder
DEFAULT_EPS = 1e-6    # width for abs/sign families, and guard floors


def _jnp() -> Any:
    """Import ``jax.numpy`` lazily so this module is importable without JAX."""
    import jax.numpy as jnp

    return jnp


# ── Guards — forward values unchanged ───────────────────────────────────────

def where_guarded(condition, when_true, when_false):
    """``jnp.where`` that does not poison the gradient with the untaken branch.

    ``jnp.where`` evaluates both branches. If the untaken one produces NaN
    or infinity — the usual case being a ``sqrt`` or a division that the
    Fortran ``IF`` was there to guard — the *gradient* is NaN even though
    the forward value is perfectly finite. This is the single most common
    defect in translated Fortran, and a tracing check sees nothing wrong.

    The fix is to make both branches finite before selecting, which callers
    do by passing already-guarded expressions; this helper exists to name
    the pattern at the call site so it is greppable in generated output.
    """
    jnp = _jnp()
    return jnp.where(condition, when_true, when_false)


def safe_sqrt(x, eps: float = DEFAULT_EPS):
    """``sqrt`` with a finite derivative at zero.

    ``d/dx sqrt(x) = 1/(2 sqrt(x))`` diverges at 0, so a kernel that takes
    the square root of a quantity reaching zero produces an infinite
    gradient. Clamping the argument keeps the derivative bounded by
    ``1/(2 sqrt(eps))``.

    Forward values are unchanged for ``x >= eps``; below that the result is
    ``sqrt(eps)`` rather than ``sqrt(x)``.
    """
    jnp = _jnp()
    return jnp.sqrt(jnp.maximum(x, eps))


def safe_divide(numerator, denominator, eps: float = DEFAULT_EPS):
    """Division whose derivative stays finite as the denominator vanishes.

    The denominator is pushed away from zero while keeping its sign, so a
    quantity approaching zero from below does not flip the result's sign.
    Forward values are unchanged wherever ``|denominator| >= eps``.
    """
    jnp = _jnp()
    sign = jnp.where(denominator >= 0, 1.0, -1.0)
    floored = sign * jnp.maximum(jnp.abs(denominator), eps)
    return numerator / floored


def safe_log(x, eps: float = DEFAULT_EPS):
    """``log`` with a finite value and derivative at zero."""
    jnp = _jnp()
    return jnp.log(jnp.maximum(x, eps))


# ── Relaxations — forward values change, by design ──────────────────────────

def smooth_max(a, b, beta: float = DEFAULT_BETA):
    """Differentiable ``MAX(a, b)``.

    ``logaddexp(beta*a, beta*b) / beta``, which is stable for any input
    magnitude — unlike the textbook ``log(exp(.) + exp(.))``.

    Converges to ``max(a, b)`` as ``beta -> inf``. The bias is largest when
    the arguments are equal, where it is exactly ``log(2)/beta``, and decays
    exponentially in ``beta * |a - b|``. With the default ``beta = 50`` that
    worst case is about ``0.014``.

    A hard ``max`` is *already* differentiable in JAX — it just has a
    discontinuous, one-sided gradient. Reach for this when that
    discontinuity is the problem, not to make the code run.
    """
    jnp = _jnp()
    return jnp.logaddexp(beta * a, beta * b) / beta


def smooth_min(a, b, beta: float = DEFAULT_BETA):
    """Differentiable ``MIN(a, b)`` — ``-smooth_max(-a, -b)``.

    Same convergence and the same ``log(2)/beta`` bias, mirrored.
    """
    return -smooth_max(-a, -b, beta)


def smooth_abs(x, eps: float = DEFAULT_EPS):
    """Differentiable ``ABS(x)`` — the pseudo-Huber form ``sqrt(x^2 + eps^2)``.

    Converges to ``|x|`` as ``eps -> 0``; the bias is exactly ``eps`` at
    ``x = 0`` and falls off as ``eps^2 / (2|x|)`` away from it. The
    derivative is ``x / sqrt(x^2 + eps^2)``, which is continuous through
    zero where ``sign(x)`` is not.
    """
    jnp = _jnp()
    return jnp.sqrt(x * x + eps * eps)


def smooth_sign(x, eps: float = DEFAULT_EPS):
    """Differentiable ``SIGN(x)`` — ``tanh(x / eps)``.

    Converges to ``sign(x)`` as ``eps -> 0`` for every ``x != 0``, and is
    exactly 0 at the origin where ``sign`` is conventionally ``0`` or ``1``
    depending on the language. Fortran's ``SIGN(a, b)`` transfers a sign
    rather than extracting one — use ``a * smooth_sign(b)`` for that.
    """
    jnp = _jnp()
    return jnp.tanh(x / eps)


def smooth_step(x, threshold=0.0, eps: float = DEFAULT_EPS):
    """Differentiable ``x > threshold`` — a logistic of width ``eps``.

    Returns a weight in ``(0, 1)`` rather than a boolean. Converges to the
    Heaviside step as ``eps -> 0``, and equals exactly ``0.5`` at the
    threshold.
    """
    jnp = _jnp()
    import jax

    return jax.nn.sigmoid((x - threshold) / eps)


def smooth_where(x, threshold, when_above, when_below, eps: float = DEFAULT_EPS):
    """Differentiable ``IF (x > threshold) THEN ... ELSE ...``.

    Blends the two branches with :func:`smooth_step` instead of selecting
    one. Both branches are evaluated — as they are under ``jnp.where`` —
    so an unsafe branch still has to be guarded first.
    """
    weight = smooth_step(x, threshold, eps)
    return weight * when_above + (1.0 - weight) * when_below


def smooth_clamp(x, lo, hi, beta: float = DEFAULT_BETA):
    """Differentiable ``MIN(MAX(x, lo), hi)`` — the limiter pattern.

    Composes :func:`smooth_max` then :func:`smooth_min`, so the bias is
    bounded by ``2 log(2) / beta`` in the worst case and the two kinks at
    ``lo`` and ``hi`` become smooth shoulders.

    Worth stating plainly: applying this to a flux limiter changes its
    mathematical character. Limiters are piecewise *on purpose* — a smooth
    one is no longer TVD. Use it for sensitivity analysis, not to produce
    a solver you then trust to be monotone.
    """
    return smooth_min(smooth_max(x, lo, beta), hi, beta)


def smooth_argmax(values, beta: float = DEFAULT_BETA, axis: int = -1):
    """Differentiable ``MAXLOC`` — a temperature-weighted index.

    Returns a fractional index: the softmax-weighted average of the
    positions, which converges to the true ``argmax`` as ``beta -> inf``
    when the maximum is unique. Ties give the mean of the tied positions,
    which is the sensible continuous answer.
    """
    jnp = _jnp()
    import jax

    weights = jax.nn.softmax(beta * jnp.asarray(values), axis=axis)
    positions = jnp.arange(jnp.asarray(values).shape[axis], dtype=weights.dtype)
    return jnp.sum(weights * positions, axis=axis)


def interp_table(x, nodes, values):
    """Differentiable table lookup — linear interpolation.

    A nearest-neighbour lookup has zero gradient everywhere, so a kernel
    that reads a physical property from a table is differentiable in name
    only. Linear interpolation is exact at the nodes and has a well-defined
    derivative between them, which is almost always what the table was
    approximating anyway.
    """
    jnp = _jnp()
    return jnp.interp(x, jnp.asarray(nodes), jnp.asarray(values))


# ── Catalogue of non-smooth Fortran constructs ─────────────────────────────
# Single source of truth, read by three consumers: the `FORT031` detection
# rule in `analyze`, the emission prompt, and the docs. Keeping it here
# rather than restating it in each means a construct cannot be detected
# without a documented replacement, or offered to the model without being
# detectable.

class NonSmooth(NamedTuple):
    """One non-smooth construct and what to do about it."""

    key: str
    pattern: str          # regex, matched case-insensitively against source
    why: str              # what breaks, in one line
    replacement: str | None   # function in this module, or None if there is none
    limit: str            # what the replacement converges to, and how


NON_SMOOTH: tuple[NonSmooth, ...] = (
    NonSmooth(
        "max", r"\bMAX\s*\(",
        "gradient is a hard switch: all of it goes to one argument, none to the other",
        "smooth_max", "max(a, b) as beta grows; bias log(2)/beta when equal",
    ),
    NonSmooth(
        "min", r"\bMIN\s*\(",
        "same hard switch as MAX, mirrored",
        "smooth_min", "min(a, b) as beta grows; bias log(2)/beta when equal",
    ),
    NonSmooth(
        "abs", r"\bABS\s*\(",
        "kink at zero: the derivative jumps from -1 to +1",
        "smooth_abs", "|x| as eps -> 0; bias exactly eps at the origin",
    ),
    NonSmooth(
        "sign", r"\bSIGN\s*\(",
        "step function: derivative is zero everywhere and undefined at zero",
        "smooth_sign", "sign(x) as eps -> 0 for every x != 0",
    ),
    NonSmooth(
        "maxloc", r"\b(MAXLOC|MINLOC)\s*\(",
        "returns a discrete index — no gradient at all",
        "smooth_argmax", "the true argmax as beta grows, when the maximum is unique",
    ),
    NonSmooth(
        "truncation", r"\b(FLOOR|CEILING|NINT|AINT|ANINT)\s*\(",
        "locally flat: the gradient is zero everywhere, which is also what a "
        "finite-difference check reports — this class is invisible to gradcheck",
        None,
        "no general relaxation. If it computes an index the quantity is genuinely "
        "discrete; if it reads a table, interpolate instead (interp_table).",
    ),
    NonSmooth(
        "modulo", r"\b(MOD|MODULO)\s*\(",
        "sawtooth: the derivative jumps at every wrap",
        None,
        "no general relaxation. If the quantity is an angle, work in sin/cos instead.",
    ),
    NonSmooth(
        "while", r"^\s*DO\s+WHILE\b",
        "a convergence loop becomes lax.while_loop, which is NOT reverse-mode "
        "differentiable in JAX — the kernel traces and then fails under grad",
        None,
        "use a fixed iteration count (scan / fori_loop), or differentiate the "
        "converged solution implicitly.",
    ),
    NonSmooth(
        "sqrt", r"\bSQRT\s*\(",
        "derivative diverges at zero",
        "safe_sqrt", "guard, not a relaxation: values are unchanged for x >= eps",
    ),
)


def catalogue_for_prompt() -> str:
    """Render the catalogue as the block the emission prompt embeds."""
    lines = []
    for entry in NON_SMOOTH:
        target = f"`{entry.replacement}`" if entry.replacement else "**no direct replacement**"
        lines.append(f"- `{entry.key.upper()}` — {entry.why}. Use {target}. {entry.limit}")
    return "\n".join(lines)


# Names the emitted code may import, and the report lists when used.
GUARDS = ("where_guarded", "safe_sqrt", "safe_divide", "safe_log")
RELAXATIONS = (
    "smooth_max", "smooth_min", "smooth_abs", "smooth_sign",
    "smooth_step", "smooth_where", "smooth_clamp", "smooth_argmax",
    "interp_table",
)

__all__ = [
    *GUARDS, *RELAXATIONS,
    "DEFAULT_BETA", "DEFAULT_EPS", "GUARDS", "RELAXATIONS",
    "NON_SMOOTH", "NonSmooth", "catalogue_for_prompt",
]
