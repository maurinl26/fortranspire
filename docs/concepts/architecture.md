# Architecture

The pipeline is a six-stage LangGraph state machine. Two stages call the
LLM; four are deterministic.

```text
📂 kernel.f90  (Fortran monolithique)
     │
     ▼ 🔍 parser          Loki AST — detects INTENT, SAVE, COMMON, loops, I/O
     │                    Deterministic — no LLM
     │
     ▼ 🔧 extractor       LLM (1 call) — extracts 2-D loops as MODULE
     │                    subroutines, removes COMMON, exposes SAVE
     │                    as INTENT(INOUT)
     │                    → module_kernels.f90  +  driver.f90
     │
     ▼ ✨ pure_elemental   AST rules — annotate PURE/ELEMENTAL
     │                    Validates: no I/O, no SAVE, INTENT explicit
     │
     ▼ 🚀 openacc         LLM (1 driver call) — !$acc parallel loop
     │                    collapse(2), !$acc data copyin/copy around
     │                    the time loop
     │
     ▼ 🐍 cython_wrapper  LLM (2 calls) — .pyx + kernel_c.h (iso_c_binding),
     │                    NumPy typed memoryviews, np.asfortranarray()
     │
     ▼ ✅ validation       gfortran × 2 flavors → nvfortran -acc (GPU)
     │                    Deterministic — compilation
     │
     📦 output/fortran_gpu/module_kernels_gpu.f90
        output/cython/module.pyx
```

## LLM budget per run

Four LLM calls maximum:

| Stage             | Calls | Role                                                                   |
| ----------------- | ----- | ---------------------------------------------------------------------- |
| `extractor`       |   1   | Refactor a monolithic `PROGRAM` into a `MODULE` of subroutines         |
| `openacc`         |   1   | Insert OpenACC parallel-loop and data-region pragmas around the driver |
| `cython_wrapper`  |   2   | Generate the `.pyx` body and the `iso_c_binding` C header              |

At Mistral-Large tariffs this is roughly 0.06 USD per kernel and about two
minutes wall-clock. Loki carries the deterministic AST work; the LLM only
intervenes where semantic understanding is required.

## State shape

The pipeline state is a typed dict carried through the LangGraph nodes:

- `fortran_filepath`, `fortran_code` — inputs.
- `ast_info` — Loki AST summary.
- `module_fortran`, `driver_fortran`, `kernel_names` — after extraction.
- `pure_elemental_fortran`, `openacc_fortran` — after purity and pragma
  passes.
- `cython_pyx`, `cython_header`, `cython_setup` — wrapper artifacts.
- `validation_passed`, `validation_log` — final compiler outcome.

Each node reads what it needs and writes only its own keys, so individual
stages can be replayed without rerunning the full pipeline.

## Human-in-the-loop

Every intermediate artifact is written to disk before the next stage runs,
so a reviewer can inspect (or hand-edit) the extracted module, the OpenACC
driver, or the Cython wrapper between stages. Re-running the pipeline from
an existing intermediate file skips the LLM call for that stage.
