# /// script
# requires-python = ">=3.10"
# ///
"""iperf3 network test runner for Kubernetes inference pods.

Extracted from run-tests.py — runs iperf3 TCP bandwidth and UDP
jitter/loss tests between all pod pairs.

Public API:
    run_iperf3(pods, display_names) -> dict
"""

import json
import subprocess
import threading
import time

from importlib import import_module

# Import shared utilities from the common module.
_common = None


def _get_common():
    global _common
    if _common is None:
        _common = import_module("run-tests-common")
    return _common


# Delegated to run-tests-common
def exec_in_pod(*args, **kwargs):
    return _get_common().exec_in_pod(*args, **kwargs)


def _kubectl_ns_args():
    return _get_common()._kubectl_ns_args()


# ---------------------------------------------------------------------------
# iperf3 functions
# ---------------------------------------------------------------------------
def start_iperf3_servers(pods):
    for name, _ip in pods:
        cmd = _get_common()._build_remote_cmd(name, ["iperf3", "-s"])
        if _get_common().VERBOSE:
            print(f"  $ {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _get_common()._server_procs_append(proc)
    print(f"Started iperf3 servers on {len(pods)} pods, waiting 2s ...")
    time.sleep(2)


def run_iperf3_client(src_name, dst_ip, extra_args=None):
    cmd_args = ["iperf3", "-c", dst_ip] + (extra_args or []) + ["-J"]
    result = exec_in_pod(src_name, cmd_args, timeout=60)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def parse_iperf3_bw_gbps(data):
    """Extract bandwidth in Gbps from iperf3 TCP JSON output."""
    if data is None:
        return None
    try:
        return data["end"]["sum_sent"]["bits_per_second"] / 1e9
    except (KeyError, TypeError):
        return None


def parse_iperf3_udp_jitter_ms(data):
    """Extract jitter in ms from iperf3 UDP JSON output."""
    if data is None:
        return None
    try:
        return data["end"]["sum"]["jitter_ms"]
    except (KeyError, TypeError):
        return None


def parse_iperf3_udp_loss_pct(data):
    """Extract packet loss percentage from iperf3 UDP JSON output."""
    if data is None:
        return None
    try:
        return data["end"]["sum"]["lost_percent"]
    except (KeyError, TypeError):
        return None


def run_iperf3(pods, display_names):
    """Run iperf3 TCP bandwidth + UDP jitter/latency for all pairs.

    Non-interfering pairs (no shared source or destination pod) run in
    parallel within the same wave, using one thread per pair.

    iperf3 uses its own default buffer sizes (128K TCP, 8K UDP) — the
    --block-sizes / --latency-size options only apply to RDMA perftest.
    """
    _c = _get_common()
    print(f"\n{'=' * 60}")
    print("  IPERF3 BANDWIDTH & LATENCY")
    print(f"{'=' * 60}")

    n = len(pods)
    bw_matrix = [[None] * n for _ in range(n)]
    jitter_matrix = [[None] * n for _ in range(n)]
    loss_matrix = [[None] * n for _ in range(n)]

    print("\nStarting iperf3 servers on all pods ...")
    start_iperf3_servers(pods)

    test_pairs = _c._cross_role_pairs(pods)
    waves = _c._schedule_parallel_pairs(n, pairs=test_pairs)
    total_pairs = len(test_pairs)
    done = 0

    for wave_idx, wave in enumerate(waves):
        pair_labels = ", ".join(
            f"{display_names[pods[i][0]]}->{display_names[pods[j][0]]}"
            for i, j in wave
        )
        print(f"\n  [wave {wave_idx + 1}/{len(waves)}] "
              f"{len(wave)} pair(s) in parallel: {pair_labels}")

        # Per-pair thread state
        buffers = []
        results_slot = [None] * len(wave)
        done_events = [threading.Event() for _ in wave]

        def _worker(slot, i, j, buf, done_evt):
            try:
                src_name = pods[i][0]
                dst_ip = pods[j][1]
                src_short = display_names[src_name]
                dst_short = display_names[pods[j][0]]

                def _out(msg):
                    buf.write(msg + "\n")

                _out(f"\n    {src_short} -> {dst_short}")

                # TCP bandwidth test (5s, 1s interval)
                try:
                    tcp_data = run_iperf3_client(src_name, dst_ip, ["-t", "5", "-i", "1"])
                    bw = parse_iperf3_bw_gbps(tcp_data)
                except subprocess.TimeoutExpired:
                    bw = None
                if bw is not None:
                    _out(f"    TCP bandwidth: {bw:.2f} Gbps")
                else:
                    _out(f"    TCP bandwidth: FAIL")

                # UDP jitter/latency test (5s, 1s interval)
                try:
                    udp_data = run_iperf3_client(src_name, dst_ip, ["-u", "-t", "5", "-i", "1"])
                    jitter = parse_iperf3_udp_jitter_ms(udp_data)
                    loss = parse_iperf3_udp_loss_pct(udp_data)
                except subprocess.TimeoutExpired:
                    jitter = None
                    loss = None

                parts = []
                if jitter is not None:
                    parts.append(f"{jitter:.3f} ms jitter")
                if loss is not None:
                    parts.append(f"{loss:.2f}% loss")
                if parts:
                    _out(f"    UDP: {', '.join(parts)}")
                else:
                    _out(f"    UDP: FAIL")

                results_slot[slot] = (i, j, bw, jitter, loss)
            except Exception as exc:
                buf.write(f"\n    ERROR: {exc}\n")
                results_slot[slot] = (i, j, None, None, None)
            finally:
                done_evt.set()

        # Launch threads for this wave
        threads = []
        for slot, (i, j) in enumerate(wave):
            buf = _c._StreamingBuffer()
            buffers.append(buf)
            t = threading.Thread(
                target=_worker,
                args=(slot, i, j, buf, done_events[slot]),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait and flush output in pair order
        for slot in range(len(wave)):
            done_events[slot].wait()
            buffers[slot].flush_all()

        for t in threads:
            t.join(timeout=5)

        # Store results in matrices
        for slot in range(len(wave)):
            if results_slot[slot] is not None:
                i, j, bw, jitter, loss = results_slot[slot]
                bw_matrix[i][j] = bw
                jitter_matrix[i][j] = jitter
                loss_matrix[i][j] = loss
                done += 1

    print(f"\n  Completed {done}/{total_pairs} pair(s)")
    print("\nStopping iperf3 servers ...")
    _c._cleanup_servers()

    return {
        "iperf3 Bandwidth (Gbps)": bw_matrix,
        "iperf3 UDP Jitter (ms)": jitter_matrix,
        "iperf3 UDP Loss (%)": loss_matrix,
    }


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-iperf3.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys

    _USAGE = """\
Usage: uv run run-tests-iperf3.py [options]

Run iperf3 TCP bandwidth and UDP jitter/loss tests between Kubernetes
inference pods.

Equivalent to: run-tests.sh -t iperf3

Options:
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -h, --help            Show this help message.
  -i, --install-deps    Install iperf3 if missing.
  -l, --label SELECTOR  Label selector to discover pods
                        (default: "llm-d.ai/inferenceServing=true").
  -n, --namespace NS    Kubernetes namespace for all kubectl commands.
  -p, --percentage-margin PCT
                        Flag results deviating more than PCT% from average
                        (default: 10).
  -v, --verbose         Print kubectl commands as they run.
  -x, --explain-verify  Run each explain command and verify output.
                        Implies --explain.
"""
    if "-h" in _sys.argv or "--help" in _sys.argv:
        print(_USAGE)
        _sys.exit(0)

    _c = _get_common()
    _cfg = _c._parse_common_args()
    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    _results = run_iperf3(_pods, _display_names)
    for _title, _matrix in _results.items():
        print(f"\n{_title}:")
        _header = [""] + [_display_names[p[0]] for p in _pods]
        print("  " + "\t".join(_header))
        for _i, (_name, _ip) in enumerate(_pods):
            _row = [_display_names[_name]]
            for _j in range(len(_pods)):
                _val = _matrix[_i][_j]
                _row.append(f"{_val:.2f}" if _val is not None else "-")
            print("  " + "\t".join(_row))
    _c.print_results_summary(_pods, _results, _display_names)

