"""Deterministic CUDA Graph capture/replay scaffold for an OpenACC driver.

An iterative HPC driver launches the *same* sequence of GPU kernels every time
step. Re-issuing each launch from the CPU has per-launch overhead that dominates
for many small kernels. A CUDA Graph captures the launch sequence once and
replays it as a single operation each step.

Design — decoupled from the kernel signatures. The generated C helper does
**only** capture / instantiate / replay on the CUDA stream that backs an
OpenACC async queue (via ``acc_get_cuda_stream``). The *driver* keeps launching
its existing kernels (with their arguments) between ``capture_begin`` and
``capture_end`` on that queue, so the helper never needs to know a kernel's
arguments. This makes the helper **generic** (identical for every project) and,
crucially, correct without compile-testing against a specific kernel set.

The helper is nvfortran + CUDA specific (OpenACC ``async`` maps to a CUDA
stream). It compiles and runs only on an NVIDIA GPU toolchain — it is emitted as
a build artifact, not exercised in CI here.
"""
from __future__ import annotations

_HELPER_C = r'''/* fortranspire CUDA Graph capture/replay — deterministic, do not edit by hand.
 *
 * Capture the OpenACC async kernel sequence once, then replay it each step.
 * Build with nvfortran/nvc + CUDA. Requires an NVIDIA GPU at run time.
 */
#include <cuda_runtime.h>
#include <openacc.h>
#include <stdio.h>

static cudaGraph_t      fs_graph = NULL;
static cudaGraphExec_t  fs_exec  = NULL;
static int              fs_ready = 0;

static cudaStream_t fs_stream(int queue) {
    /* The CUDA stream backing OpenACC async queue `queue`. */
    return (cudaStream_t) acc_get_cuda_stream(queue);
}

/* Begin capturing every kernel the driver launches on `queue` after this call. */
void fortranspire_capture_begin(int queue) {
    if (fs_ready) return;                 /* already captured — nothing to do */
    cudaStreamBeginCapture(fs_stream(queue), cudaStreamCaptureModeThreadLocal);
}

/* End capture and instantiate the executable graph. */
void fortranspire_capture_end(int queue) {
    if (fs_ready) return;
    cudaStreamEndCapture(fs_stream(queue), &fs_graph);
    cudaGraphInstantiate(&fs_exec, fs_graph, NULL, NULL, 0);
    fs_ready = 1;
}

/* Replay the captured graph on `queue`. Falls back to a no-op before capture. */
void fortranspire_graph_launch(int queue) {
    if (!fs_ready) return;
    cudaGraphLaunch(fs_exec, fs_stream(queue));
}

/* Free the graph (call once at shutdown). */
void fortranspire_graph_destroy(void) {
    if (fs_exec)  { cudaGraphExecDestroy(fs_exec);  fs_exec  = NULL; }
    if (fs_graph) { cudaGraphDestroy(fs_graph);     fs_graph = NULL; }
    fs_ready = 0;
}
'''

_HELPER_H = r'''/* fortranspire CUDA Graph capture/replay — C interface (bind(c)-compatible). */
#ifndef FORTRANSPIRE_CUDA_GRAPH_H
#define FORTRANSPIRE_CUDA_GRAPH_H
#ifdef __cplusplus
extern "C" {
#endif
void fortranspire_capture_begin(int queue);
void fortranspire_capture_end(int queue);
void fortranspire_graph_launch(int queue);
void fortranspire_graph_destroy(void);
#ifdef __cplusplus
}
#endif
#endif /* FORTRANSPIRE_CUDA_GRAPH_H */
'''


def _driver_usage(queue: int, kernel_calls: list[str]) -> str:
    """A Fortran snippet showing how the time loop uses the graph."""
    seq = "\n".join(f"           call {c}   ! async({queue}) kernel" for c in kernel_calls) \
        or f"           ! ... the async({queue}) kernel launches, unchanged ..."
    return f'''! ── Driver integration (add to the time loop) ─────────────────────────
!   interface
!     subroutine fortranspire_capture_begin(q) bind(c, name="fortranspire_capture_begin")
!       use iso_c_binding; integer(c_int), value :: q; end subroutine
!     subroutine fortranspire_capture_end(q)   bind(c, name="fortranspire_capture_end")
!       use iso_c_binding; integer(c_int), value :: q; end subroutine
!     subroutine fortranspire_graph_launch(q)  bind(c, name="fortranspire_graph_launch")
!       use iso_c_binding; integer(c_int), value :: q; end subroutine
!   end interface
!
!   do it = 1, nstep
!     if (it == 1) then
!        call fortranspire_capture_begin({queue})
{seq}
!        !$acc wait({queue})
!        call fortranspire_capture_end({queue})
!     end if
!     call fortranspire_graph_launch({queue})
!     !$acc wait({queue})
!   end do
! ──────────────────────────────────────────────────────────────────────'''


def render_cuda_graph(kernel_calls: list[str] | None = None,
                      queue: int = 1) -> dict:
    """Return {'c', 'h', 'usage'} — the generic capture helper + driver snippet.

    ``kernel_calls`` (ordered kernel launch statements from the driver) only
    documents the usage snippet; the C helper is generic and independent of it.
    """
    return {
        "c": _HELPER_C,
        "h": _HELPER_H,
        "usage": _driver_usage(queue, kernel_calls or []),
    }
