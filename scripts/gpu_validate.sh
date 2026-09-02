#!/usr/bin/env bash
# gpu_validate.sh — the missing GPU validation step: compile the port with
# nvfortran AND run the numerical equivalence harness (GPU vs CPU).
#
# fortranspire's pipeline generates output/tests/test_<mod>_equivalence.py but
# never runs it — a GPU build and a real GPU are needed, which the dev machine
# does not have. Run this on a GPU node (NVIDIA HPC SDK + an NVIDIA GPU):
#
#     bash scripts/gpu_validate.sh output/
#
# It fails loudly if the GPU output does not match the original Fortran within
# tolerance — a missing reduction clause, a wrong pragma, a race — turning
# "compiles" into "verified".
set -euo pipefail

OUT="${1:-output}"
FC="${FC:-nvfortran}"

echo "=== fortranspire GPU validation ==="
echo "  output dir : ${OUT}"
echo "  compiler   : ${FC}"

# ── 0. Environment ───────────────────────────────────────────────────────────
command -v "${FC}" >/dev/null 2>&1 || {
    echo "❌ ${FC} not found — need the NVIDIA HPC SDK. On this node install it,"
    echo "   or use containers/Dockerfile.nvhpc."; exit 2; }
command -v gfortran >/dev/null 2>&1 || { echo "❌ gfortran not found (CPU oracle)"; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || {
    echo "⚠️  no NVIDIA GPU visible — the harness will build but cannot run on GPU"; }

# ── 1. CPU oracle: f2py-wrap the ORIGINAL Fortran (double precision) ──────────
ORIG=$(ls "${OUT}"/fortran_gpu/module_kernels*.f90 2>/dev/null | head -1 || true)
[ -n "${ORIG}" ] || { echo "❌ no generated Fortran under ${OUT}/fortran_gpu/"; exit 2; }

echo "  [1/3] Building the CPU oracle (f2py, gfortran) …"
( cd "${OUT}" && python -m numpy.f2py -c "$(basename "${ORIG}")" -m cpu_mod \
      --backend meson >/dev/null )

# ── 2. GPU build: the Cython extension via nvfortran -acc ─────────────────────
echo "  [2/3] Building the GPU extension (${FC} -acc) …"
( cd "${OUT}" && FC="${FC}" python setup.py build_ext --inplace >/dev/null )

# ── 3. Run the generated numerical equivalence harness (GPU vs CPU) ───────────
echo "  [3/3] Running the equivalence harness (GPU output vs original) …"
if [ -d "${OUT}/tests" ]; then
    ( cd "${OUT}" && python -m pytest tests/ -q )
    echo "✅ GPU output matches the original Fortran within tolerance — VALIDATED."
else
    echo "❌ no equivalence harness at ${OUT}/tests/ — re-run 'fortranspire gpu'"; exit 2
fi
