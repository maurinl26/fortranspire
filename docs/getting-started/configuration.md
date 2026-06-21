# Configuration

`fortranspire` reads its configuration exclusively from environment
variables (or from a local `.env` file, loaded automatically). A complete
template is shipped at the root of the repository as `.env.example`.

## LLM endpoint

| Variable             | Default                            | Description                                         |
| -------------------- | ---------------------------------- | --------------------------------------------------- |
| `MISTRAL_ENDPOINT`   | `https://api.mistral.ai/v1`        | Base URL of any OpenAI-compatible chat endpoint     |
| `MISTRAL_API_KEY`    | *(unset)*                          | Bearer token sent to the endpoint                   |
| `MISTRAL_MODEL`      | `mistral-large-latest`             | Model name passed in every request                  |
| `LLM_TEMPERATURE`    | `0.0`                              | Sampling temperature                                |
| `LLM_TOP_P`          | `0.9`                              | Nucleus sampling cutoff                             |
| `LLM_NUM_PREDICT`    | `2048`                             | Maximum tokens per completion                       |

The same variables work for Mistral La Plateforme, a self-hosted vLLM
server, an Ollama instance, or any other OpenAI-compatible backend.

## MCP server

| Variable    | Default   | Description                                            |
| ----------- | --------- | ------------------------------------------------------ |
| `MCP_HOST`  | `0.0.0.0` | Bind address                                           |
| `MCP_PORT`  | `8000`    | Listen port                                            |
| `API_KEY`   | *(unset)* | If set, require `Authorization: Bearer <key>` on every |
|             |           | request (skipped on `/health` and `/ping`)             |

## Pipeline behavior

| Variable          | Default              | Description                                |
| ----------------- | -------------------- | ------------------------------------------ |
| `WORKSPACE_DIR`   | repo root            | Where intermediate files are written       |
| `MAX_ITERATIONS`  | `10`                 | Maximum LangGraph re-entry count           |
| `OUTPUT_DIR`      | `output/`            | Root for generated Fortran / Cython / JAX  |

## Compiler toolchain

| Variable        | Default      | Description                                   |
| --------------- | ------------ | --------------------------------------------- |
| `FC_CPU`        | `gfortran`   | Compiler used for the CPU syntax checks       |
| `FC_GPU`        | `nvfortran`  | Compiler used for the OpenACC validation step |
| `CUDA_ARCH`     | `cc80`       | GPU architecture passed to `nvfortran -gpu=`  |

## Loading the `.env`

The CLI and the MCP server both call `python-dotenv` at startup, so simply
placing a `.env` file at the repository root (or in your working directory)
is enough — no flag required. CI workflows should set the same variables
through repository secrets.
