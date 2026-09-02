# fortranspire containers

| Image | Base | Purpose |
| ----- | ---- | ------- |
| `Dockerfile`        | slim python | core analysis + JAX (Phase 2) |
| `Dockerfile.hpc`    | `nvcr.io/nvidia/jax` | JAX on a GPU (Phase 2 execution) |
| `Dockerfile.nvhpc`  | `nvcr.io/nvidia/nvhpc` | **nvfortran** — GPU **validation** of the Phase 1 port (#121) |

## GPU validation (Dockerfile.nvhpc)

The dev machine and default CI have no NVIDIA GPU or `nvfortran`, so the Phase 1
GPU port is only *generated* there — never compiled, run, or numerically
validated. This image has `nvfortran` (OpenACC) + `gfortran` (CPU oracle) +
`meson`/`ninja` (f2py backend), so it can do the real check:

```bash
# On a GPU node with Docker + the NVIDIA container toolkit:
docker run --rm --gpus all -v "$PWD:/work" \
    ghcr.io/maurinl26/fortranspire-nvhpc:latest \
    bash scripts/gpu_validate.sh output/
```

`gpu_validate.sh` builds the CPU oracle (f2py of the original), builds the GPU
extension (`nvfortran -acc`), and **runs** the equivalence harness — a mismatch
(a missing reduction clause, a wrong data clause) fails it. Compilation alone is
not correctness.

- **Build/publish**: `.github/workflows/build-nvhpc.yml` (needs `NGC_API_KEY` to
  pull `nvcr.io/nvidia/nvhpc`) → `ghcr.io/<owner>/fortranspire-nvhpc`.
- **CI**: `.github/workflows/gpu-validate.yml` runs the validation inside this
  image on a self-hosted `gpu` runner.
- **Cloud on-demand**: #122 (runpodctl). **Sovereign**: #43 (EWC).
