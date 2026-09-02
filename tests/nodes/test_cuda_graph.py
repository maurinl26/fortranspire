"""CUDA Graph capture/replay scaffold generation.

The helper is nvfortran+CUDA and cannot be compiled in CI; these assert the
generated scaffold has the correct capture/replay structure and interface.
"""
from fortranspire.agent.nodes.cuda_graph import render_cuda_graph


def test_c_helper_has_the_capture_replay_lifecycle():
    c = render_cuda_graph()["c"]
    for call in ("cudaStreamBeginCapture", "cudaStreamEndCapture",
                 "cudaGraphInstantiate", "cudaGraphLaunch", "acc_get_cuda_stream"):
        assert call in c, f"missing {call} in the CUDA graph helper"


def test_capture_is_guarded_to_run_once():
    c = render_cuda_graph()["c"]
    # capture_begin/end short-circuit once the graph is ready
    assert "if (fs_ready) return;" in c
    assert "fs_ready = 1;" in c


def test_header_declares_the_four_bindc_entrypoints():
    h = render_cuda_graph()["h"]
    for fn in ("fortranspire_capture_begin", "fortranspire_capture_end",
               "fortranspire_graph_launch", "fortranspire_graph_destroy"):
        assert fn in h
    assert 'extern "C"' in h


def test_usage_snippet_lists_the_driver_kernel_sequence():
    usage = render_cuda_graph(["update_flux(a, b)", "advance_temp(t)"], queue=2)["usage"]
    assert "update_flux(a, b)" in usage
    assert "advance_temp(t)" in usage
    assert "fortranspire_capture_begin(2)" in usage      # honours the queue
    assert "fortranspire_graph_launch(2)" in usage


def test_usage_is_valid_without_a_known_sequence():
    usage = render_cuda_graph([])["usage"]
    assert "capture_begin(1)" in usage and "graph_launch(1)" in usage
