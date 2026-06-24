"""fortranspire — LLM + Model Context Protocol pipeline porting legacy
Fortran 90 HPC kernels to GPU (OpenACC / OpenMP target) and Cython, with
optional Fortran → JAX (Phase 2) translation.

Built on the ECMWF Loki AST framework for deterministic Fortran analysis;
LLM stages call any OpenAI-compatible endpoint (Mistral La Plateforme by
default, self-hosted vLLM / TGI / Ollama for full sovereignty).

Entry points: ``fortranspire <verb>`` console script (analyze, doc,
explain, format, graph, diff, report, bench, gpu, port-batch, translate,
profile, mcp). See ``fortranspire --help`` for usage.
"""

__version__ = "0.2.0"
