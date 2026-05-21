#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.7",
# ]
# ///
"""Unified network test runner: perftest and iperf3 between pods.

Discovers pods with label llm-d.ai/inferenceServing=true (override with
--label SELECTOR) and runs selected tests between every pair.

By default only runs perftest.  Use --tests to select:
  --tests perftest          (default) RDMA latency & bandwidth
                            (ib_{read,write,send}_{lat,bw})
  --tests iperf3            iperf3 TCP bandwidth + UDP jitter/latency
  --tests all               run both tests
  --tests perftest,iperf3   same as all

--rdma-block-sizes and --rdma-latency-size control RDMA perftest message sizes only.
iperf3 uses its own defaults (128K TCP buffer, 8K UDP buffer).

All tests run directly in the pod by default.  Pass --debug-image to use
ephemeral debug containers instead.

Pass --install-deps to automatically install iperf3 and build perftest
from source when missing.
"""

import os
import sys
import threading

from importlib import import_module as _import_module

_common = _import_module("run-tests-common")

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
USAGE = """\
Usage: run-tests.sh [options]

Run network tests (perftest, iperf3) between Kubernetes pods.

By default only runs perftest (RDMA latency & bandwidth:
ib_{read,write,send}_{lat,bw}).
Use --tests to select which tests to run.

Options:
  -b, --rdma-block-sizes SIZES
                        Comma-separated list of message sizes for RDMA
                        perftest bandwidth (ib_{read,write,send}_bw).
                        Accepts bytes or suffixes K/M/G (e.g. 64K,1M,1G).
                        Does not affect iperf3. (default: 1G).
  -d, --discover-topology
                        Only run GPU topology discovery (nvidia-smi topo,
                        nvlink status, lscpu, lspci) and exit without
                        running any network tests.  Note: topology discovery
                        also runs by default before tests; use -t to skip it
                        and run only the specified tests.
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
                        Optional — by default all tests run directly in the
                        pod's main container.
  -e, --explain         Show the kubectl/shell commands used to determine
                        each finding, so results can be reproduced manually.
  -g, --gid-index INDEX GID index for RoCE.  Auto-detected if not specified.
  -h, --help            Show this help message.
  -i, --install-deps    Install all test dependencies on every pod:
                        iperf3, perftest build tools, etcd, and nixlbench.
                        Builds perftest and nixlbench from source if missing.
  -l, --label SELECTOR  Label selector to discover pods (default:
                        "llm-d.ai/model").
  -n, --namespace NS    Kubernetes namespace for all kubectl commands.
                        Uses current context namespace if not specified.
  -p, --percentage-margin PCT
                        Percentage margin for flagging outlier results.
                        Any result deviating more than PCT% from the average
                        is flagged (default: 10).
  -r, --rdma-device DEVICE
                        RDMA device to use (e.g. mlx5_0).  Auto-detected if
                        not specified.
  -s, --rdma-latency-size SIZE
                        Message size for RDMA perftest latency tests
                        (ib_{read,write,send}_lat). Does not affect iperf3.
                        (default: 2 bytes).
  -t, --tests TESTS     Comma-separated list of tests to run.
                        Choices: perftest, iperf3, nccl-rccl, nixlbench, all
                        (default: perftest).  When -t is given explicitly,
                        topology discovery is skipped (use -d to force it).
  -v, --verbose         Print kubectl/SSH commands as they run.
  -x, --explain-verify  Run each explain command and verify it produces
                        expected output.  Implies --explain.  Use with -d
                        for a full self-test of the discovery logic.
  --nixlbench-backend BACKEND
                        NIXLBench backend (default: UCX).
  --nixlbench-buffer-size SIZE
                        NIXLBench total buffer size (default: 8G).
  --nixlbench-seg-type TYPE
                        NIXLBench segment type for initiator and target
                        (default: VRAM).  Choices: DRAM, VRAM.
  --preflight-info      Call GET /info on all pods in preflight pause mode,
                        display results, and summarize what is shared vs.
                        different between pods.  Exits without running tests.
  --preflight-status    Check pod logs for preflight pause-mode indicators.
                        Reports per-pod whether preflight checks are running
                        and in what state (paused, completed, or not detected).
                        Exits without running network tests.
  --unpause             Call /exit on all pods in preflight pause mode to
                        resume vLLM startup.  Detects port from logs (tries
                        8000, then 8200 for decode pods).
                        Exits without running network tests.
  --ssh-command CMD     SSH command to use (default: "ssh").  Example:
                        "ssh -i /path/key -o StrictHostKeyChecking=no".
                        Requires --ssh-hosts.
  --ssh-hosts HOSTS     Comma-separated list of SSH hosts to use instead of
                        kubectl pod discovery.  Format: host1,host2 or
                        user@host1,user@host2 or host1:ip1,host2:ip2.

Environment variables:
  PERFTEST_LABEL        Label selector (same as -l/--label).
  PERFTEST_NAMESPACE    Kubernetes namespace (same as -n/--namespace).
  PERFTEST_INSTALL_DEPS Set to 1/true/yes to enable -i/--install-deps via env.
  PERFTEST_DEVICE       Override RDMA device (same as -r/--rdma-device).
  PERFTEST_GID_INDEX    Override GID index (same as -g/--gid-index).
  PERFTEST_DURATION     Test duration in seconds for BW tests (default: 5).
  PERFTEST_ITERATIONS   Number of iterations for latency tests (default: 1000).
  RDMA_BLOCK_SIZES      RDMA perftest BW message sizes (same as
                        -b/--rdma-block-sizes, default: 1073741824).
  RDMA_LATENCY_SIZE     RDMA perftest latency message size (same as
                        -s/--rdma-latency-size, default: 2).
  PERFTEST_PORT         Base port for perftest (default: 18515).
  DEBUG_IMAGE           Debug container image (same as -D/--debug-image).
"""

if "-h" in sys.argv or "--help" in sys.argv:
    print(USAGE)
    sys.exit(0)

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
DISCOVER_ONLY = "-d" in sys.argv or "--discover-topology" in sys.argv
EXPLAIN_VERIFY = "-x" in sys.argv or "--explain-verify" in sys.argv
EXPLAIN = "-e" in sys.argv or "--explain" in sys.argv or EXPLAIN_VERIFY
OPT_PREFLIGHT_STATUS = "--preflight-status" in sys.argv
OPT_UNPAUSE = "--unpause" in sys.argv
OPT_PREFLIGHT_INFO = "--preflight-info" in sys.argv

DEBUG_IMAGE = os.environ.get("DEBUG_IMAGE", "").strip()
INSTALL_DEPS = os.environ.get("PERFTEST_INSTALL_DEPS", "").strip() in ("1", "true", "yes")
OPT_DEVICE = os.environ.get("PERFTEST_DEVICE", "").strip() or None
OPT_GID_INDEX = os.environ.get("PERFTEST_GID_INDEX", "").strip() or None
_LABEL_FROM_ENV = os.environ.get("PERFTEST_LABEL", "").strip()
if _LABEL_FROM_ENV.endswith("=") and "=" not in _LABEL_FROM_ENV[:-1]:
    _LABEL_FROM_ENV = _LABEL_FROM_ENV[:-1]
OPT_LABEL = _LABEL_FROM_ENV or "llm-d.ai/model"
OPT_NAMESPACE = os.environ.get("PERFTEST_NAMESPACE", "").strip() or None
OPT_TESTS = "perftest"
_TESTS_EXPLICIT = False  # True when -t/--tests was passed on CLI
OPT_RDMA_BLOCK_SIZES = os.environ.get("RDMA_BLOCK_SIZES", "1073741824").strip()  # default 1 GB
OPT_RDMA_LATENCY_SIZE = os.environ.get("RDMA_LATENCY_SIZE", "2").strip()  # default 2 bytes
OPT_PERCENTAGE_MARGIN = 10.0  # default 10%
OPT_SSH_COMMAND = None
OPT_SSH_HOSTS = None
OPT_NIXLBENCH_BACKEND = "UCX"
OPT_NIXLBENCH_SEG_TYPE = "VRAM"
OPT_NIXLBENCH_BUFFER_SIZE = "8G"

_skip_next = False
for _i, _arg in enumerate(sys.argv[1:], 1):
    if _skip_next:
        _skip_next = False
        continue
    if _arg in ("-D", "--debug-image") and _i < len(sys.argv) - 1:
        DEBUG_IMAGE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--debug-image="):
        DEBUG_IMAGE = _arg.split("=", 1)[1]
    elif _arg in ("-r", "--rdma-device") and _i < len(sys.argv) - 1:
        OPT_DEVICE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--rdma-device="):
        OPT_DEVICE = _arg.split("=", 1)[1]
    elif _arg in ("-g", "--gid-index") and _i < len(sys.argv) - 1:
        OPT_GID_INDEX = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--gid-index="):
        OPT_GID_INDEX = _arg.split("=", 1)[1]
    elif _arg in ("-l", "--label") and _i < len(sys.argv) - 1:
        OPT_LABEL = sys.argv[_i + 1]
        if OPT_LABEL.endswith("=") and "=" not in OPT_LABEL[:-1]:
            OPT_LABEL = OPT_LABEL[:-1]
        _skip_next = True
    elif _arg.startswith("--label="):
        OPT_LABEL = _arg.split("=", 1)[1]
        if OPT_LABEL.endswith("=") and "=" not in OPT_LABEL[:-1]:
            OPT_LABEL = OPT_LABEL[:-1]
    elif _arg in ("-n", "--namespace") and _i < len(sys.argv) - 1:
        OPT_NAMESPACE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--namespace="):
        OPT_NAMESPACE = _arg.split("=", 1)[1]
    elif _arg in ("-t", "--tests") and _i < len(sys.argv) - 1:
        OPT_TESTS = sys.argv[_i + 1]
        _TESTS_EXPLICIT = True
        _skip_next = True
    elif _arg.startswith("--tests="):
        OPT_TESTS = _arg.split("=", 1)[1]
        _TESTS_EXPLICIT = True
    elif _arg in ("-b", "--rdma-block-sizes") and _i < len(sys.argv) - 1:
        OPT_RDMA_BLOCK_SIZES = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--rdma-block-sizes="):
        OPT_RDMA_BLOCK_SIZES = _arg.split("=", 1)[1]
    elif _arg in ("-s", "--rdma-latency-size") and _i < len(sys.argv) - 1:
        OPT_RDMA_LATENCY_SIZE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--rdma-latency-size="):
        OPT_RDMA_LATENCY_SIZE = _arg.split("=", 1)[1]
    elif _arg in ("-p", "--percentage-margin") and _i < len(sys.argv) - 1:
        OPT_PERCENTAGE_MARGIN = float(sys.argv[_i + 1])
        _skip_next = True
    elif _arg.startswith("--percentage-margin="):
        OPT_PERCENTAGE_MARGIN = float(_arg.split("=", 1)[1])
    elif _arg in ("-i", "--install-deps"):
        INSTALL_DEPS = True
    elif _arg == "--ssh-command" and _i < len(sys.argv) - 1:
        OPT_SSH_COMMAND = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--ssh-command="):
        OPT_SSH_COMMAND = _arg.split("=", 1)[1]
    elif _arg == "--ssh-hosts" and _i < len(sys.argv) - 1:
        OPT_SSH_HOSTS = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--ssh-hosts="):
        OPT_SSH_HOSTS = _arg.split("=", 1)[1]
    elif _arg == "--nixlbench-backend" and _i < len(sys.argv) - 1:
        OPT_NIXLBENCH_BACKEND = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--nixlbench-backend="):
        OPT_NIXLBENCH_BACKEND = _arg.split("=", 1)[1]
    elif _arg == "--nixlbench-seg-type" and _i < len(sys.argv) - 1:
        OPT_NIXLBENCH_SEG_TYPE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--nixlbench-seg-type="):
        OPT_NIXLBENCH_SEG_TYPE = _arg.split("=", 1)[1]
    elif _arg == "--nixlbench-buffer-size" and _i < len(sys.argv) - 1:
        OPT_NIXLBENCH_BUFFER_SIZE = sys.argv[_i + 1]
        _skip_next = True
    elif _arg.startswith("--nixlbench-buffer-size="):
        OPT_NIXLBENCH_BUFFER_SIZE = _arg.split("=", 1)[1]

USE_DEBUG_CONTAINER = bool(DEBUG_IMAGE)
DEBUG_SUFFIX = "-debug"

# Resolve test list
VALID_TESTS = {"perftest", "iperf3", "nccl-rccl", "nixlbench"}
ALL_TESTS_ORDER = ["iperf3", "perftest", "nccl-rccl", "nixlbench"]  # default order for --tests all
if OPT_TESTS.strip().lower() == "all":
    SELECTED_TESTS = list(ALL_TESTS_ORDER)
else:
    SELECTED_TESTS = [t.strip().lower() for t in OPT_TESTS.split(",") if t.strip()]
    unknown = set(SELECTED_TESTS) - VALID_TESTS
    if unknown:
        print(f"Error: unknown test(s): {', '.join(unknown)}.  Valid: {', '.join(sorted(VALID_TESTS))}, all", file=sys.stderr)
        sys.exit(1)

# Parse RDMA block sizes list (for bandwidth tests)
RDMA_BLOCK_SIZES = [_common._parse_size(s) for s in OPT_RDMA_BLOCK_SIZES.split(",") if s.strip()]
if not RDMA_BLOCK_SIZES:
    print("Error: --rdma-block-sizes must contain at least one size.", file=sys.stderr)
    sys.exit(1)

# Parse RDMA latency message size
RDMA_LATENCY_SIZE = _common._parse_size(OPT_RDMA_LATENCY_SIZE)

# Perftest settings
PERFTEST_DURATION = int(os.environ.get("PERFTEST_DURATION", "5"))
PERFTEST_ITERATIONS = int(os.environ.get("PERFTEST_ITERATIONS", "1000"))
PERFTEST_PORT = int(os.environ.get("PERFTEST_PORT", "18515"))

# Parse SSH options
import shlex as _shlex
_SSH_COMMAND = None
_SSH_HOSTS = []
_ssh_target_map = {}
if OPT_SSH_HOSTS:
    _SSH_COMMAND = _shlex.split(OPT_SSH_COMMAND) if OPT_SSH_COMMAND else ["ssh"]
    _SSH_HOSTS, _ssh_target_map = _common._parse_ssh_hosts(OPT_SSH_HOSTS)
    if USE_DEBUG_CONTAINER:
        print("Warning: --debug-image is ignored in SSH mode", file=sys.stderr)
        USE_DEBUG_CONTAINER = False
elif OPT_SSH_COMMAND:
    print("Error: --ssh-command requires --ssh-hosts", file=sys.stderr)
    sys.exit(1)

# Push all parsed config into the common module so submodules can access it.
_common.configure(
    VERBOSE=VERBOSE,
    DISCOVER_ONLY=DISCOVER_ONLY,
    EXPLAIN=EXPLAIN,
    EXPLAIN_VERIFY=EXPLAIN_VERIFY,
    USE_DEBUG_CONTAINER=USE_DEBUG_CONTAINER,
    DEBUG_IMAGE=DEBUG_IMAGE,
    DEBUG_SUFFIX=DEBUG_SUFFIX,
    INSTALL_DEPS=INSTALL_DEPS,
    OPT_DEVICE=OPT_DEVICE,
    OPT_GID_INDEX=OPT_GID_INDEX,
    OPT_LABEL=OPT_LABEL,
    OPT_NAMESPACE=OPT_NAMESPACE,
    SSH_COMMAND=_SSH_COMMAND,
    SSH_HOSTS=_SSH_HOSTS,
    _ssh_target_map=_ssh_target_map,
    DISPLAY_NAME_MAX_LEN=_common.DISPLAY_NAME_MAX_LEN,
    PERFTEST_DURATION=PERFTEST_DURATION,
    PERFTEST_ITERATIONS=PERFTEST_ITERATIONS,
    PERFTEST_PORT=PERFTEST_PORT,
    RDMA_BLOCK_SIZES=RDMA_BLOCK_SIZES,
    RDMA_LATENCY_SIZE=RDMA_LATENCY_SIZE,
    OPT_PERCENTAGE_MARGIN=OPT_PERCENTAGE_MARGIN,
)


# ===========================================================================
# PERFTEST (in run-tests-perftest.py)
# ===========================================================================
_pf = _import_module("run-tests-perftest")
run_perftest = _pf.run_perftest
install_all_deps = _pf.install_all_deps
build_perftest = _pf.build_perftest
perftest_available = _pf.perftest_available
_check_binaries = _pf._check_binaries
PERFTEST_BINARIES = _pf.PERFTEST_BINARIES

# ===========================================================================
# IPERF3 (in run-tests-iperf3.py)
# ===========================================================================
_ip3 = _import_module("run-tests-iperf3")
run_iperf3 = _ip3.run_iperf3

# ===========================================================================
# NCCL/RCCL (in run-tests-nccl-rccl.py)
# ===========================================================================
_nccl = _import_module("run-tests-nccl-rccl")
run_nccl_rccl = _nccl.run_nccl_rccl

# ===========================================================================
# NIXLBENCH (in run-tests-nixlbench.py)
# ===========================================================================
_nlb = _import_module("run-tests-nixlbench")
_nlb.NIXLBENCH_BACKEND = OPT_NIXLBENCH_BACKEND
_nlb.NIXLBENCH_SEG_TYPE = OPT_NIXLBENCH_SEG_TYPE
_nlb.NIXLBENCH_BUFFER_SIZE = OPT_NIXLBENCH_BUFFER_SIZE
run_nixlbench = _nlb.run_nixlbench

# ===========================================================================
# GPU Topology Validation (in run-tests-discovery.py)
# ===========================================================================
_disc = _import_module("run-tests-discovery")
run_topology_validation = _disc.run_topology_validation


# ===========================================================================
# Output: tables and heatmaps
# ===========================================================================
def _fmt_val(val, is_latency):
    """Format a value with higher precision for latency metrics."""
    if is_latency:
        return f"{val:.4f}"
    return f"{val:.2f}"


def _is_latency_title(title):
    t = title.lower()
    return "latency" in t or "jitter" in t or "loss" in t


def print_combined_table(pods, all_results, display_names):
    """Print one combined table: rows = pod pairs, columns = test metrics."""
    n = len(pods)
    metrics = list(all_results.keys())
    if not metrics:
        return

    # Build rows: one per directed pair (src -> dst)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src_short = display_names[pods[i][0]]
            dst_short = display_names[pods[j][0]]
            pairs.append((i, j, f"{src_short} -> {dst_short}"))

    # Column widths
    pair_width = max(len(p[2]) for p in pairs)
    metric_widths = {}
    for m in metrics:
        is_lat = _is_latency_title(m)
        # Width = max of header or formatted values
        w = len(m)
        for i, j, _ in pairs:
            val = all_results[m][i][j]
            if val is not None:
                w = max(w, len(_fmt_val(val, is_lat)))
            else:
                w = max(w, 4)  # "FAIL"
        metric_widths[m] = w

    # Print header
    print(f"\n{'=' * 60}")
    print("  COMBINED RESULTS — All Tests")
    print(f"{'=' * 60}")
    header = " " * (pair_width + 2) + "  ".join(m.rjust(metric_widths[m]) for m in metrics)
    print(header)
    print("-" * len(header))

    # Print rows
    for i, j, pair_label in pairs:
        cells = []
        for m in metrics:
            val = all_results[m][i][j]
            w = metric_widths[m]
            if val is not None:
                cells.append(_fmt_val(val, _is_latency_title(m)).rjust(w))
            else:
                cells.append("FAIL".rjust(w))
        print(f"{pair_label.rjust(pair_width)}  " + "  ".join(cells))

    # Pod name legend
    print(f"\nPod name legend:")
    for name, _ip in pods:
        print(f"  {display_names[name]:{_common.DISPLAY_NAME_MAX_LEN}s}  = {name}")
    print()


print_results_summary = _common.print_results_summary


def generate_combined_png(pods, all_results, display_names, output="all.png"):
    """Generate a single combined table as PNG: rows = pod pairs, columns = metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping PNG generation", file=sys.stderr)
        return

    n = len(pods)
    metrics = list(all_results.keys())
    if not metrics:
        return

    # Build pair labels and data grid
    pair_labels = []
    data_rows = []  # list of lists, one per pair
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src_short = display_names[pods[i][0]]
            dst_short = display_names[pods[j][0]]
            pair_labels.append(f"{src_short} -> {dst_short}")
            row = []
            for m in metrics:
                row.append(all_results[m][i][j])
            data_rows.append(row)

    num_pairs = len(pair_labels)
    num_metrics = len(metrics)

    # Build cell text and colors
    cell_text = []
    cell_colors = []

    for row in data_rows:
        text_row = []
        color_row = []
        for col_idx, val in enumerate(row):
            m = metrics[col_idx]
            is_lat = _is_latency_title(m)
            if val is not None:
                text_row.append(_fmt_val(val, is_lat))
                color_row.append("white")
            else:
                text_row.append("FAIL")
                color_row.append("#ffcccc")
        cell_text.append(text_row)
        cell_colors.append(color_row)

    # Color cells by relative value within each column
    for col_idx, m in enumerate(metrics):
        is_lat = _is_latency_title(m)
        col_vals = [data_rows[r][col_idx] for r in range(num_pairs)
                    if data_rows[r][col_idx] is not None]
        if not col_vals:
            continue
        avg = np.mean(col_vals)
        for r in range(num_pairs):
            val = data_rows[r][col_idx]
            if val is None:
                continue
            if is_lat:
                # Higher than avg = worse for latency
                if val > avg * 1.05:
                    cell_colors[r][col_idx] = "#ffdddd"  # light red
                elif val < avg * 0.95:
                    cell_colors[r][col_idx] = "#ddffdd"  # light green
            else:
                # Lower than avg = worse for bandwidth
                if val < avg * 0.9:
                    cell_colors[r][col_idx] = "#ffdddd"
                elif val > avg * 1.1:
                    cell_colors[r][col_idx] = "#ddffdd"

    # Figure sizing
    col_w = max(1.8, max(len(m) for m in metrics) * 0.12)
    row_h = 0.35
    fig_w = max(10, 3 + num_metrics * col_w)
    fig_h = max(6, 2 + num_pairs * row_h + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=pair_labels,
        colLabels=metrics,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    # Style header row
    for col_idx in range(num_metrics):
        cell = table[0, col_idx]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6)

    # Style row labels
    for row_idx in range(num_pairs):
        cell = table[row_idx + 1, -1]
        cell.set_facecolor("#D9E2F3")
        cell.set_text_props(fontsize=6)

    # Pod name legend at bottom
    legend_lines = [f"{display_names[p[0]]}  =  {p[0]}" for p in pods]
    legend_text = "\n".join(legend_lines)
    fig.text(
        0.5, 0.01, legend_text,
        ha="center", va="bottom", fontsize=6, family="monospace",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    plt.suptitle("Pod-to-Pod Network Tests — Combined Results", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Combined results saved to {output}")


# ===========================================================================
# Preflight check helpers
# ===========================================================================
import re as _re


def _get_pod_logs(pod_name, tail=50):
    """Fetch the last `tail` lines of logs from a pod."""
    ns = _common._kubectl_ns_args()
    cmd = ["kubectl", "logs"] + ns + [pod_name, "--tail", str(tail)]
    result = _common.run_cmd(cmd, timeout=30)
    if result.returncode != 0:
        return ""
    return result.stdout


def _detect_preflight_port(log_text):
    """Parse preflight server port from log text. Returns int or None."""
    m = _re.search(r"Preflight server listening on :(\d+)", log_text)
    if m:
        return int(m.group(1))
    return None


def _detect_port_for_pod(pod_name):
    """Detect preflight port: try logs first, then probe 8000/8200.

    Distinguishes the preflight server from other services (vLLM, sidecars)
    by checking that /health returns the preflight-specific JSON body.
    """
    logs = _get_pod_logs(pod_name, tail=500)
    port = _detect_preflight_port(logs)
    if port is not None:
        return port, logs

    for try_port in (8000, 8200):
        result = _common.exec_in_pod(
            pod_name, ["curl", "-s", f"http://localhost:{try_port}/health"], timeout=5
        )
        if result.returncode == 0 and '"status"' in result.stdout and "ok" in result.stdout:
            return try_port, logs
    return None, logs


def _preflight_status(pods, display_names):
    """Check pod logs and HTTP probes for preflight pause-mode state."""
    print("\n=== Preflight Status Check ===\n")

    paused_markers = [
        "=== Preflight checks PAUSED: waiting for /exit before allowing regular pod startup ===",
        "=== PAUSED: vLLM startup is on hold ===",
    ]
    starting_marker = "=== llm-d-preflight-checks.py starting ==="
    pause_mode_marker = "(mode='pause')"
    shutdown_marker = "Shutting down preflight server..."
    continuing_marker = "Continuing..."

    for name, _ip in pods:
        dname = display_names[name]
        logs = _get_pod_logs(name, tail=500)

        if not logs:
            print(f"  {dname}: (no logs available)")
            continue

        has_starting = starting_marker in logs
        has_paused = any(m in logs for m in paused_markers)
        has_pause_mode = pause_mode_marker in logs
        has_shutdown = shutdown_marker in logs
        has_continuing = continuing_marker in logs
        port = _detect_preflight_port(logs)

        # If log markers indicate pause mode but we didn't find the port,
        # probe the HTTP server to confirm it's still active
        if (has_paused or has_pause_mode) and not has_shutdown:
            if port is None:
                port, _ = _detect_port_for_pod(name)
            if port is not None:
                result = _common.exec_in_pod(
                    name, ["curl", "-s", f"http://localhost:{port}/health"], timeout=5
                )
                if result.returncode == 0 and "ok" in result.stdout:
                    print(f"  {dname}: PAUSED — HTTP server active on port {port}")
                else:
                    print(f"  {dname}: COMPLETED — preflight ran (pause mode) but server no longer responding")
            else:
                print(f"  {dname}: PAUSED (port unknown) — logs show pause mode but could not detect port")
        elif has_shutdown and has_continuing:
            print(f"  {dname}: COMPLETED — preflight exited, vLLM continuing")
        elif has_starting and not has_pause_mode:
            print(f"  {dname}: RUNNING — preflight started (not in pause mode)")
        elif not has_starting:
            # No log markers at all — try probing HTTP health as last resort
            probe_port, _ = _detect_port_for_pod(name)
            if probe_port is not None:
                result = _common.exec_in_pod(
                    name, ["curl", "-s", f"http://localhost:{probe_port}/health"], timeout=5
                )
                if result.returncode == 0 and "ok" in result.stdout:
                    print(f"  {dname}: PAUSED — HTTP server active on port {probe_port} (no log markers in last 500 lines)")
                else:
                    print(f"  {dname}: NOT DETECTED — no preflight indicators found")
            else:
                print(f"  {dname}: NOT DETECTED — no preflight indicators found")
        else:
            print(f"  {dname}: UNKNOWN — preflight started but state unclear")
    print()


def _preflight_unpause(pods, display_names):
    """Call /exit on all discovered pods to resume vLLM startup."""
    print("\n=== Unpausing Preflight (calling /exit) ===\n")

    for name, _ip in pods:
        dname = display_names[name]
        port, _logs = _detect_port_for_pod(name)

        if port is None:
            print(f"  {dname}: SKIP — could not detect preflight port")
            continue

        result = _common.exec_in_pod(
            name, ["curl", "-s", f"http://localhost:{port}/exit"], timeout=10
        )
        if result.returncode == 0 and "shutting down" in result.stdout:
            print(f"  {dname}: OK — /exit called on port {port}")
        else:
            stderr_hint = f" ({result.stderr.strip()})" if result.stderr and result.stderr.strip() else ""
            print(f"  {dname}: FAILED — port {port}, rc={result.returncode}{stderr_hint}")
    print()


def _parse_info_sections(text):
    """Parse ===== section_name ===== delimited text into dict."""
    sections = {}
    current_section = None
    lines = []
    for line in text.split("\n"):
        m = _re.match(r"^=====\s+(.+?)\s+=====\s*$", line)
        if m:
            if current_section is not None:
                sections[current_section] = "\n".join(lines)
            current_section = m.group(1)
            lines = []
        else:
            lines.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(lines)
    return sections


def _preflight_info(pods, display_names):
    """Call GET /info on all pods, display results, summarize shared vs different."""
    print("\n=== Preflight Info Collection ===\n")

    pod_infos = {}

    for name, _ip in pods:
        dname = display_names[name]
        port, _logs = _detect_port_for_pod(name)

        if port is None:
            print(f"  {dname}: SKIP — could not detect preflight port")
            continue

        result = _common.exec_in_pod(
            name, ["curl", "-s", f"http://localhost:{port}/info"], timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            pod_infos[dname] = result.stdout
            print(f"  {dname}: OK — collected info from port {port}")
        else:
            print(f"  {dname}: FAILED — no info returned from port {port}")

    if not pod_infos:
        print("\nNo info collected from any pod.")
        return

    all_sections = {dname: _parse_info_sections(text) for dname, text in pod_infos.items()}

    # Collect all section names preserving first-seen order
    all_section_names = []
    seen = set()
    for sections in all_sections.values():
        for sname in sections:
            if sname not in seen:
                all_section_names.append(sname)
                seen.add(sname)

    pod_names_list = list(pod_infos.keys())

    print(f"\n{'=' * 60}")
    print(f"  INFO COMPARISON ({len(pod_infos)} pods)")
    print(f"{'=' * 60}")

    shared_sections = []
    different_sections = []

    for section_name in all_section_names:
        contents = [all_sections.get(d, {}).get(section_name, "(missing)") for d in pod_names_list]
        if len(set(contents)) == 1:
            shared_sections.append(section_name)
        else:
            different_sections.append(section_name)

    if shared_sections:
        print(f"\n  SHARED (identical across all {len(pod_infos)} pods):")
        for s in shared_sections:
            print(f"    - {s}")
        # Show shared content once
        first_pod = pod_names_list[0]
        for s in shared_sections:
            content = all_sections[first_pod].get(s, "")
            print(f"\n  ===== {s} =====")
            for line in content.strip().split("\n"):
                print(f"    {line}")

    if different_sections:
        print(f"\n  DIFFERENT (varies between pods):")
        for s in different_sections:
            print(f"    - {s}")

        for s in different_sections:
            print(f"\n  ===== {s} =====")
            for dname in pod_names_list:
                content = all_sections.get(dname, {}).get(s, "(missing)")
                print(f"    [{dname}]:")
                for line in content.strip().split("\n"):
                    print(f"      {line}")
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    _is_preflight = OPT_PREFLIGHT_STATUS or OPT_UNPAUSE or OPT_PREFLIGHT_INFO

    if _is_preflight:
        if OPT_PREFLIGHT_STATUS:
            print("Preflight mode: --preflight-status (no tests will be run)")
        elif OPT_PREFLIGHT_INFO:
            print("Preflight mode: --preflight-info (no tests will be run)")
        elif OPT_UNPAUSE:
            print("Preflight mode: --unpause (no tests will be run)")
    else:
        bw_sizes_str = ", ".join(_common._human_size(s) for s in RDMA_BLOCK_SIZES)
        print(f"Selected tests: {', '.join(SELECTED_TESTS)}")
        print(f"RDMA perftest BW message sizes: [{bw_sizes_str}]")
        print(f"RDMA perftest latency message size: {_common._human_size(RDMA_LATENCY_SIZE)}")
        print(f"iperf3: uses default buffer sizes (128K TCP, 8K UDP)")
        if USE_DEBUG_CONTAINER:
            print(f"Debug image: {DEBUG_IMAGE}")
        else:
            print("Running directly in pod containers (no debug image).")
        print(f"Install deps if missing: {'yes' if INSTALL_DEPS else 'no (use --install-deps to enable)'}")

    # Discover pods
    print(f"\nDiscovering pods with label: {OPT_LABEL} ...")
    pods = _common.discover_pods(OPT_LABEL)
    display_names = _common.make_display_names([p[0] for p in pods])

    # Detect pod roles (prefill / decode) for cross-role test filtering
    _common.discover_pod_roles(OPT_LABEL)
    role_info = []
    for name, ip in pods:
        role = _common.get_pod_role(name)
        role_tag = f"  [{role}]" if role else ""
        role_info.append((name, ip, role_tag))

    print(f"Found {len(pods)} pod(s):")
    for name, ip, role_tag in role_info:
        print(f"  {display_names[name]:{_common.DISPLAY_NAME_MAX_LEN}s}  {name}  ({ip}){role_tag}")

    # Show cross-role filtering status
    test_pairs = _common._cross_role_pairs(pods)
    all_pairs_count = len(pods) * (len(pods) - 1)
    if len(test_pairs) < all_pairs_count:
        prefill_count = sum(1 for n, _ in pods if _common.get_pod_role(n) == "prefill")
        decode_count = sum(1 for n, _ in pods if _common.get_pod_role(n) == "decode")
        print(f"  Cross-role testing: {prefill_count} prefill + {decode_count} decode pods "
              f"→ {len(test_pairs)} cross-role pairs (skipping {all_pairs_count - len(test_pairs)} same-role pairs)")

    # Preflight action options — run and exit without network tests
    if OPT_PREFLIGHT_STATUS:
        _preflight_status(pods, display_names)
        return
    if OPT_UNPAUSE:
        _preflight_unpause(pods, display_names)
        return
    if OPT_PREFLIGHT_INFO:
        _preflight_info(pods, display_names)
        return

    # Create debug containers if any test needs them
    if USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _common.create_debug_containers(pods)

    # GPU topology validation — runs by default and with -d, skipped when -t is explicit
    if DISCOVER_ONLY or not _TESTS_EXPLICIT:
        run_topology_validation(pods, display_names)

    if DISCOVER_ONLY:
        print("Topology discovery complete (--discover-topology). Skipping network tests.")
        return

    # Map test -> (commands to check, human-readable label)
    dep_map = {
        "iperf3":    (["iperf3"],        "iperf3"),
        "perftest":  (PERFTEST_BINARIES, "perftest"),
        # nccl-rccl manages its own deps (mpirun, openssh, nccl/rccl-tests)
        # inside run_nccl_rccl() — no upfront dep check needed.
        # nixlbench manages its own deps (nixlbench binary + etcd)
        # inside run_nixlbench() — no upfront dep check needed.
    }

    def check_deps():
        """Check binaries for all selected tests. Returns set of tests with missing deps.

        Uses a single kubectl exec per pod to check all required binaries at once.
        """
        # Collect all binaries needed across selected tests
        all_binaries = []
        for test in SELECTED_TESTS:
            if test in dep_map:
                all_binaries.extend(dep_map[test][0])
        if not all_binaries:
            return set()

        # One exec per pod to check all binaries
        pod_missing = {}  # pod_display_name -> set of missing binaries
        for name, _ip in pods:
            missing_bins = _check_binaries(name, all_binaries)
            if missing_bins:
                pod_missing[display_names[name]] = missing_bins

        # Map results back to tests
        tests_missing = set()
        for test in SELECTED_TESTS:
            if test not in dep_map:
                continue
            cmd_names, label = dep_map[test]
            bad_pods = sorted(
                pname for pname, mbins in pod_missing.items()
                if mbins & set(cmd_names)
            )
            if bad_pods:
                tests_missing.add(test)
                print(f"  Warning: {label} not found on: {', '.join(bad_pods)}", file=sys.stderr)
            else:
                print(f"  {label}: OK on all pods")
        return tests_missing

    # First pass: check if all binaries are available
    print("\nChecking dependencies on all pods ...")
    tests_to_skip = check_deps()

    # If any tests have missing deps and --install-deps is enabled, install and re-check
    if tests_to_skip and INSTALL_DEPS:
        print("\nMissing dependencies detected — installing on all pods in parallel ...")
        need_perftest_build = "perftest" in tests_to_skip

        n = len(pods)
        buffers = [_common._StreamingBuffer() for _ in range(n)]
        done_events = [threading.Event() for _ in range(n)]

        def _install_worker(idx, pod_name, buf, done_evt):
            try:
                def _out(msg):
                    buf.write(msg + "\n")

                install_all_deps(pod_name, out=_out)
                if need_perftest_build and not perftest_available(pod_name):
                    build_perftest(pod_name, out=_out)
            except Exception as exc:
                buf.write(f"  ERROR installing on {pod_name}: {exc}\n")
            finally:
                done_evt.set()

        # Launch all install threads
        install_threads = []
        for i, (name, _ip) in enumerate(pods):
            t = threading.Thread(
                target=_install_worker,
                args=(i, name, buffers[i], done_events[i]),
                daemon=True,
            )
            install_threads.append(t)
            t.start()

        # Stream output in pod order (first pod live, then second, etc.)
        for i in range(n):
            while not done_events[i].is_set():
                buffers[i].flush_new()
                done_events[i].wait(timeout=0.1)
            buffers[i].flush_all()

        for t in install_threads:
            t.join(timeout=5)

        # Re-check after installation
        print("\nRe-checking dependencies after installation ...")
        tests_to_skip = check_deps()

    if tests_to_skip:
        if not INSTALL_DEPS:
            print(
                f"\nHint: re-run with -i/--install-deps to automatically install "
                f"missing dependencies.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nInstallation was attempted but some dependencies are still missing.\n"
                f"The pod filesystem may be read-only or the package repo unavailable.",
                file=sys.stderr,
            )
        remaining = [t for t in SELECTED_TESTS if t not in tests_to_skip]
        if not remaining:
            print(f"\nError: no tests can run — all selected tests have missing dependencies.", file=sys.stderr)
            sys.exit(1)
        print(f"\nSkipping tests with missing deps: {', '.join(sorted(tests_to_skip))}")
        print(f"Will run: {', '.join(remaining)}")
    else:
        print("All dependencies OK.")

    # Run tests in the order specified by --tests
    test_runners = {
        "iperf3":    run_iperf3,
        "perftest":  run_perftest,
        "nccl-rccl": run_nccl_rccl,
        "nixlbench": run_nixlbench,
    }
    all_results = {}
    for test in SELECTED_TESTS:
        if test in tests_to_skip:
            continue
        results = test_runners[test](pods, display_names)
        all_results.update(results)

    # Separate matrix results (perftest, iperf3) from non-matrix results
    # (nccl-rccl returns rows, not pod-pair matrices).
    matrix_results = {k: v for k, v in all_results.items()
                      if isinstance(v, list) and v and isinstance(v[0], list)}

    # Print combined summary table, outlier analysis, and generate PNG
    if matrix_results:
        print_combined_table(pods, matrix_results, display_names)
        print_results_summary(pods, matrix_results, display_names, OPT_PERCENTAGE_MARGIN)

        run_all = set(SELECTED_TESTS) == VALID_TESTS or OPT_TESTS.strip().lower() == "all"
        output_file = "all.png" if run_all else "tests-matrix.png"
        generate_combined_png(pods, matrix_results, display_names, output=output_file)


if __name__ == "__main__":
    main()
