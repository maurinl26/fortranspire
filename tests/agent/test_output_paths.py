"""Regression tests for the translate/gt4py output-path derivation.

A destructive bug: `filepath.replace(".f90", "_jax.py")` returned the *input*
path unchanged for any other suffix (`.F`, `.f`, `.for`), so an empty/blocked
run wrote its output straight over the source file and destroyed it (observed
on CMAQ `rbfeval.F`, a fixed-form `.F`). The output must never equal the input,
and an empty module must never clobber an existing file.
"""
from pathlib import Path

import pytest

from fortranspire.agent.cli import _sibling_output, _write_output


@pytest.mark.parametrize(
    "name, tail, expected",
    [
        ("kernel.f90", "_jax.py", "kernel_jax.py"),
        ("kernel.F90", "_jax.py", "kernel_jax.py"),
        ("rbfeval.F", "_jax.py", "rbfeval_jax.py"),      # the CMAQ case
        ("legacy.f", "_jax.py", "legacy_jax.py"),
        ("old.for", "_gt4py.py", "old_gt4py.py"),
        ("MODEL.FOR", "_gt4py.py", "MODEL_gt4py.py"),
    ],
)
def test_output_is_beside_input_and_never_equal(tmp_path, name, tail, expected):
    src = tmp_path / name
    out = _sibling_output(str(src), tail)
    assert Path(out).name == expected
    assert Path(out).resolve() != src.resolve()  # the whole point


def test_empty_module_does_not_clobber_existing_file(tmp_path, capsys):
    target = tmp_path / "rbfeval_jax.py"
    target.write_text("PRECIOUS")
    _write_output(str(target), "   \n", "JAX module")  # blocked run → empty
    assert target.read_text() == "PRECIOUS"            # untouched
    assert "left untouched" in capsys.readouterr().out


def test_nonempty_module_is_written(tmp_path):
    target = tmp_path / "kernel_jax.py"
    _write_output(str(target), "import jax\n", "JAX module")
    assert target.read_text() == "import jax\n"
