# /// script
# requires-python = ">=3.10"
# ///
"""RDMA perftest runner for Kubernetes inference pods.

Extracted from run-tests.py — runs ib_{read,write,send}_{lat,bw} tests
between pods, auto-detects RDMA devices and GID indexes, and optionally
installs/builds perftest from source.

Public API:
    run_perftest(pods, display_names) -> dict
    install_all_deps(pod_name) -> None
    build_perftest(pod_name) -> None
    perftest_available(pod_name) -> bool
    detect_rdma_config(pod_name) -> (device, gid_index)
"""

import os
import re
import subprocess
import sys
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
def run_cmd(*args, **kwargs):
    return _get_common().run_cmd(*args, **kwargs)


def exec_in_pod(*args, **kwargs):
    return _get_common().exec_in_pod(*args, **kwargs)


def _kubectl_ns_args():
    return _get_common()._kubectl_ns_args()


def _human_size(n):
    return _get_common()._human_size(n)


# ---------------------------------------------------------------------------
# Perftest constants
# ---------------------------------------------------------------------------
PERFTEST_BINARIES = [
    "ib_read_lat", "ib_write_lat", "ib_send_lat",
    "ib_read_bw", "ib_write_bw", "ib_send_bw",
]
PERFTEST_REPO = "https://github.com/linux-rdma/perftest.git"
PERFTEST_DEPS = [
    "libibverbs-devel", "libibverbs-utils", "librdmacm-devel",
    "pciutils-devel", "automake", "autoconf", "libtool",
]

# Non-root .deb extraction paths and URLs
PERFTEST_NONROOT_DIR = "/tmp/perftest-bin"
PERFTEST_DEB_URL = "http://archive.ubuntu.com/ubuntu/pool/universe/p/perftest/perftest_4.4+0.37-1_amd64.deb"
RDMACM_DEB_URL = "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/librdmacm1_39.0-1_amd64.deb"
LIBIBUMAD_DEB_URL = "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibumad3_39.0-1_amd64.deb"


# ---------------------------------------------------------------------------
# Non-root environment detection and helpers
# ---------------------------------------------------------------------------
_nonroot_cache = {}


def _detect_nonroot(pod_name):
    """Detect if a pod runs as non-root (cannot write to /usr/local/bin)."""
    if pod_name in _nonroot_cache:
        return _nonroot_cache[pod_name]
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         "touch /usr/local/bin/.test_write 2>/dev/null "
         "&& rm -f /usr/local/bin/.test_write && echo ROOT || echo NONROOT"],
        use_debug=False,
    )
    is_nonroot = "NONROOT" in (result.stdout or "")
    _nonroot_cache[pod_name] = is_nonroot
    return is_nonroot


def _perftest_env_prefix(pod_name):
    """Build shell env prefix for non-root pods to find perftest binaries."""
    if not _detect_nonroot(pod_name):
        return ""
    return (
        f"export PATH={PERFTEST_NONROOT_DIR}:$PATH && "
        f"export LD_LIBRARY_PATH={PERFTEST_NONROOT_DIR}/lib:${{LD_LIBRARY_PATH:-}} && "
    )


def _install_perftest_nonroot(pod_name, out=None):
    """Install perftest + librdmacm via .deb extraction for non-root pods."""
    _out = out or print
    _out(f"  Installing perftest via .deb extraction on {pod_name} (non-root) ...")
    script = (
        f"mkdir -p {PERFTEST_NONROOT_DIR}/lib && "
        f"cd /tmp && "
        f"curl -sL -o perftest.deb '{PERFTEST_DEB_URL}' && "
        f"dpkg-deb -x perftest.deb perftest-extract && "
        f"cp perftest-extract/usr/bin/* {PERFTEST_NONROOT_DIR}/ && "
        f"curl -sL -o rdmacm.deb '{RDMACM_DEB_URL}' && "
        f"dpkg-deb -x rdmacm.deb rdmacm-extract && "
        f"find rdmacm-extract -name '*.so*' -exec cp {{}} {PERFTEST_NONROOT_DIR}/lib/ \\; && "
        f"curl -sL -o libibumad.deb '{LIBIBUMAD_DEB_URL}' && "
        f"dpkg-deb -x libibumad.deb libibumad-extract && "
        f"find libibumad-extract -name '*.so*' -exec cp {{}} {PERFTEST_NONROOT_DIR}/lib/ \\; && "
        f"rm -rf perftest.deb rdmacm.deb libibumad.deb perftest-extract rdmacm-extract libibumad-extract && "
        f"chmod +x {PERFTEST_NONROOT_DIR}/ib_* && "
        f"test -x {PERFTEST_NONROOT_DIR}/ib_write_bw && echo PERFTEST_INSTALLED"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", script], timeout=120, use_debug=False)
    if "PERFTEST_INSTALLED" in (result.stdout or ""):
        _out(f"  perftest installed successfully in {PERFTEST_NONROOT_DIR} on {pod_name}.")
        return True
    _out(f"  Warning: perftest .deb install may have failed on {pod_name}.")
    if result.stderr:
        _out(f"    stderr: {result.stderr.strip()[:300]}")
    return False


# ---------------------------------------------------------------------------
# Perftest functions
# ---------------------------------------------------------------------------
def _check_binaries(pod_name, binaries):
    """Check which binaries are available on a pod in a single exec call.

    Returns a set of binaries that are missing.
    """
    # Build a one-liner: for each binary, print FOUND:<name> or MISSING:<name>
    checks = " && ".join(
        f'(command -v {b} >/dev/null 2>&1 && echo "FOUND:{b}" || echo "MISSING:{b}")'
        for b in binaries
    )
    env_prefix = _perftest_env_prefix(pod_name)
    result = exec_in_pod(pod_name, ["bash", "-c", env_prefix + checks], use_debug=False)
    missing = set()
    if result.returncode != 0:
        return set(binaries)  # assume all missing on failure
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("MISSING:"):
            missing.add(line[8:])
    return missing


def perftest_available(pod_name):
    """Check if all perftest binaries are available on the pod."""
    return len(_check_binaries(pod_name, PERFTEST_BINARIES)) == 0


def install_all_deps(pod_name, out=None):
    """Install all dependencies for every test on a pod.

    Installs packages for ping (iputils-ping), iperf3, and perftest build
    dependencies regardless of which tests are selected, so that any test
    can be re-run later without repeating the install step.
    """
    _out = out or print
    _out(f"  Installing all test dependencies on {pod_name} ...")

    if _detect_nonroot(pod_name):
        _out(f"  Non-root pod detected, using .deb extraction ...")
        _install_perftest_nonroot(pod_name, out=_out)
        return

    # Package groups — installed one group at a time so a failure in one
    # (e.g. iputils on a read-only filesystem) doesn't block the rest.
    rpm_groups = [
        (["iperf3"], "iperf3"),
        (["libibverbs-utils", "infiniband-diags"], "RDMA diagnostic tools (ibv_devices, ibstat)"),
        (PERFTEST_DEPS + ["git", "make", "gcc"], "perftest build tools"),
    ]
    apt_groups = [
        (["iperf3"], "iperf3"),
        (["ibverbs-utils", "infiniband-diags"], "RDMA diagnostic tools (ibv_devices, ibstat)"),
        ([
            "libibverbs-dev", "ibverbs-utils", "librdmacm-dev", "libpci-dev",
            "automake", "autoconf", "libtool", "git", "make", "gcc",
        ], "perftest build tools"),
    ]

    def _install_status(result, label, already_patterns):
        """Print install/skip/fail status based on command output."""
        if result.returncode != 0:
            _out(f"  Warning: failed to install {label}: {result.stderr.strip()[:200]}")
        else:
            stdout = (result.stdout or "").lower()
            if any(p in stdout for p in already_patterns):
                _out(f"    Skipped {label} (already installed).")
            else:
                _out(f"    Installed {label}.")

    rpm_already = ["already installed", "nothing to do"]
    apt_already = ["already the newest version", "is already the newest"]

    for pkg_mgr, install_cmd in [
        ("dnf", ["dnf", "install", "-y"]),
        ("yum", ["yum", "install", "-y"]),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            for pkgs, label in rpm_groups:
                result = exec_in_pod(pod_name, install_cmd + pkgs, timeout=300, use_debug=False)
                _install_status(result, label, rpm_already)
            break
    else:
        exec_in_pod(pod_name, ["apt-get", "update"], timeout=120, use_debug=False)
        for pkgs, label in apt_groups:
            result = exec_in_pod(pod_name, ["apt-get", "install", "-y"] + pkgs, timeout=300, use_debug=False)
            _install_status(result, label, apt_already)

    _out(f"  Package installation complete on {pod_name}.")


def build_perftest(pod_name, out=None):
    """Clone and build perftest from source on a pod's main container."""
    _out = out or print
    if _detect_nonroot(pod_name):
        return _install_perftest_nonroot(pod_name, out=_out)
    _out(f"  Cloning perftest on {pod_name} ...")
    exec_in_pod(pod_name, ["rm", "-rf", "/tmp/perftest"], use_debug=False)
    result = exec_in_pod(pod_name, ["git", "clone", PERFTEST_REPO, "/tmp/perftest"], timeout=120, use_debug=False)
    if result.returncode != 0:
        _out(f"  Error cloning perftest: {result.stderr}")
        return False

    _out(f"  Building perftest on {pod_name} ...")
    for step_cmd in [
        ["bash", "-c", "cd /tmp/perftest && ./autogen.sh"],
        ["bash", "-c", "cd /tmp/perftest && ./configure"],
        ["bash", "-c", "cd /tmp/perftest && make -j$(nproc)"],
        ["bash", "-c", "cd /tmp/perftest && make install"],
    ]:
        result = exec_in_pod(pod_name, step_cmd, timeout=300, use_debug=False)
        if _get_common().VERBOSE and result.stdout.strip():
            _out(result.stdout[-500:])
        if result.returncode != 0:
            _out(f"  Build step failed: {' '.join(step_cmd)}")
            _out(f"  stderr: {result.stderr[-500:]}")
            return False
    _out(f"  perftest built successfully on {pod_name}.")
    return True


def ensure_perftest(pod_name, out=None):
    _out = out or print
    if perftest_available(pod_name):
        _out(f"  perftest binaries already available on {pod_name}.")
        return
    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: perftest binaries not found on {pod_name}.")
        _out(f"  Re-run with --install-deps to automatically build them.")
        sys.exit(1)
    _out(f"  perftest binaries not found on {pod_name}, building from source ...")
    build_perftest(pod_name, out=_out)
    if not perftest_available(pod_name):
        _out(f"  Error: perftest binaries still not available after build on {pod_name}.")
        sys.exit(1)


def detect_rdma_device(pod_name):
    result = exec_in_pod(pod_name, ["bash", "-c", "ls /sys/class/infiniband/ 2>/dev/null | head -1"], use_debug=False)
    dev = result.stdout.strip()
    if result.returncode != 0 or not dev:
        print(f"  Warning: no RDMA devices found on {pod_name}", file=sys.stderr)
        return None
    return dev


def detect_gid_index(pod_name, device):
    script = (
        f'for f in /sys/class/infiniband/{device}/ports/1/gid_attrs/types/*; do '
        f'  idx=$(basename "$f"); '
        f'  gtype=$(cat "$f" 2>/dev/null); '
        f'  gid=$(cat /sys/class/infiniband/{device}/ports/1/gids/"$idx" 2>/dev/null); '
        f'  [ -n "$gid" ] && [ "$gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ] && '
        f'  echo "$idx $gtype $gid"; '
        f'done; true'
    )
    result = exec_in_pod(pod_name, ["bash", "-c", script], use_debug=False)
    if result.stdout.strip():
        rocev2_ipv4 = rocev2_any = rocev1_ipv4 = any_gid = None
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) < 3:
                continue
            idx = parts[0]
            gid = parts[-1]
            gtype = " ".join(parts[1:-1])
            is_ipv4_mapped = ":ffff:" in gid and not gid.startswith("fe80:")
            is_rocev2 = "v2" in gtype.lower()
            if _get_common().VERBOSE:
                print(f"    GID[{idx}] type={gtype} gid={gid} ipv4={is_ipv4_mapped} rocev2={is_rocev2}")
            if any_gid is None:
                any_gid = idx
            if is_rocev2 and is_ipv4_mapped:
                rocev2_ipv4 = idx
            elif is_rocev2 and rocev2_any is None:
                rocev2_any = idx
            elif not is_rocev2 and is_ipv4_mapped and rocev1_ipv4 is None:
                rocev1_ipv4 = idx
        chosen = rocev2_ipv4 or rocev2_any or rocev1_ipv4 or any_gid
        if chosen is not None:
            return chosen

    # Fallback: ibv_devinfo
    result = exec_in_pod(pod_name, ["ibv_devinfo", "-d", device, "-v"], use_debug=False)
    if result.returncode == 0 and result.stdout.strip():
        for match in re.finditer(r"GID\[\s*(\d+)\]:\s+(\S+)", result.stdout):
            idx, gid = match.group(1), match.group(2)
            if gid != "0000:0000:0000:0000:0000:0000:0000:0000":
                return idx
    return None


def detect_rdma_config(pod_name):
    device = _get_common().OPT_DEVICE or detect_rdma_device(pod_name)
    gid_index = _get_common().OPT_GID_INDEX
    if device and gid_index is None:
        gid_index = detect_gid_index(pod_name, device)
    print(f"  {pod_name}: device={device}, gid_index={gid_index}")
    return device, gid_index


def _is_latency_test(binary):
    return binary.endswith("_lat")


def _perftest_kind(binary):
    """Extract operation name from binary, e.g. 'ib_read_bw' -> 'Read'."""
    # binary format: ib_<op>_<lat|bw>
    parts = binary.split("_")
    if len(parts) >= 2:
        return parts[1].capitalize()  # read->Read, write->Write, send->Send
    return binary


def build_perftest_args(binary, device, gid_index, port, size, server_ip=None):
    """Build command-line args for any perftest binary (latency or bandwidth)."""
    args = []
    if server_ip:
        args.append(server_ip)
    if device:
        args += ["-d", device]
    if gid_index is not None:
        args += ["-x", str(gid_index)]
    args += ["-F", "--port", str(port), "--size", str(size)]
    if _is_latency_test(binary):
        args += ["-n", str(_get_common().PERFTEST_ITERATIONS)]
    else:
        args += ["-q", "1", "--report_gbits", "--duration", str(_get_common().PERFTEST_DURATION)]
    return args


def start_perftest_server(server_pod, binary, device, gid_index, port, size, out=None):
    _out = out or print
    ib_args = build_perftest_args(binary, device, gid_index, port, size)
    env_prefix = _perftest_env_prefix(server_pod)
    if env_prefix:
        cmd_parts = ["bash", "-c", env_prefix + binary + " " + " ".join(ib_args)]
    else:
        cmd_parts = [binary] + ib_args
    cmd = _get_common()._build_remote_cmd(server_pod, cmd_parts)

    ib_cmd_str = f"{binary} {' '.join(ib_args)}"
    _out(f"  Server cmd: {ib_cmd_str}")
    if _get_common().VERBOSE:
        _out(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _get_common()._server_procs_append(proc)
    _out(f"  Started {binary} server on {server_pod}, waiting 2s ...")
    time.sleep(2)
    return proc


def run_perftest_client(client_pod, binary, server_ip, device, gid_index, port, size, out=None):
    _out = out or print
    ib_args = build_perftest_args(binary, device, gid_index, port, size, server_ip=server_ip)
    env_prefix = _perftest_env_prefix(client_pod)
    if env_prefix:
        cmd_parts = ["bash", "-c", env_prefix + binary + " " + " ".join(ib_args)]
    else:
        cmd_parts = [binary] + ib_args
    cmd = _get_common()._build_remote_cmd(client_pod, cmd_parts)

    ib_cmd_str = f"{binary} {' '.join(ib_args)}"
    _out(f"  Client cmd: {ib_cmd_str}")
    if _get_common().VERBOSE:
        _out(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def parse_bw_output(output):
    """Parse bandwidth test output (ib_read_bw / ib_write_bw).
    Columns: #bytes, iterations, bw_peak, bw_avg, msg_rate
    Returns avg bandwidth in Gb/s (when --report_gbits is used)."""
    in_data = False
    last_bw = None
    for line in output.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("#bytes"):
            in_data = True
            continue
        if in_data and stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) >= 5:
                last_bw = float(parts[3])  # bw_avg
    return last_bw


def parse_lat_output(output):
    """Parse latency test output (ib_read_lat / ib_write_lat).
    Columns: #bytes, iterations, t_min[usec], t_max[usec], t_typical[usec], t_avg[usec]
    Returns avg latency in usec."""
    in_data = False
    last_lat = None
    for line in output.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("#bytes"):
            in_data = True
            continue
        if in_data and stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) >= 6:
                last_lat = float(parts[5])  # t_avg
    return last_lat


def run_perftest_pair(src_name, dst_ip, src_dev, src_gid, dst_name, dst_dev, dst_gid,
                      binary="ib_write_bw", port=None, size=65536, out=None):
    """Run a single perftest binary between two pods. Returns parsed value or None."""
    _out = out or print
    if port is None:
        port = _get_common().PERFTEST_PORT
    proc = start_perftest_server(dst_name, binary, dst_dev, dst_gid, port, size, out=_out)
    result = run_perftest_client(src_name, binary, dst_ip, src_dev, src_gid, port, size, out=_out)
    # Cleanup this server
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass
    _get_common()._server_procs_remove(proc)

    if result.returncode != 0:
        _out(f"    FAILED: {result.stderr.strip()}")
        return None

    if _is_latency_test(binary):
        return parse_lat_output(result.stdout)
    else:
        return parse_bw_output(result.stdout)


def _run_perftest_matrix(pods, display_names, rdma_configs, binary, port, size):
    """Run a single perftest binary for all pairs at a given size. Returns matrix.

    Non-interfering pairs (no shared source or destination pod) run in
    parallel within the same wave, using one thread per pair.
    """
    _c = _get_common()
    n = len(pods)
    test_pairs = _c._cross_role_pairs(pods)
    total_pairs = len(test_pairs)
    is_lat = _is_latency_test(binary)
    unit = "usec" if is_lat else "Gb/s"
    size_label = _human_size(size)

    print(f"\n{'─' * 50}")
    print(f"  Running {binary} size={size_label} ({unit}) — port {port} — {total_pairs} pair(s)")
    print(f"{'─' * 50}")

    waves = _c._schedule_parallel_pairs(n, pairs=test_pairs)
    matrix = [[None] * n for _ in range(n)]
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
        results_slot = [None] * len(wave)  # (i, j, value)
        done_events = [threading.Event() for _ in wave]

        def _worker(slot, i, j, buf, done_evt):
            try:
                src_name, _ = pods[i]
                dst_name, dst_ip = pods[j]
                src_short = display_names[src_name]
                dst_short = display_names[dst_name]
                src_dev, src_gid = rdma_configs[src_name]
                dst_dev, dst_gid = rdma_configs[dst_name]

                def _out(msg):
                    buf.write(msg + "\n")

                _out(f"\n    {src_short} -> {dst_short}")
                val = run_perftest_pair(
                    src_name, dst_ip, src_dev, src_gid,
                    dst_name, dst_dev, dst_gid,
                    binary=binary, port=port, size=size, out=_out,
                )
                if val is not None:
                    fmt = f"{val:.4f}" if is_lat else f"{val:.2f}"
                    _out(f"    {fmt} {unit}")
                else:
                    _out(f"    FAIL")
                results_slot[slot] = (i, j, val)
            except Exception as exc:
                buf.write(f"\n    ERROR: {exc}\n")
                results_slot[slot] = (i, j, None)
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

        # Store results in matrix
        for slot in range(len(wave)):
            if results_slot[slot] is not None:
                i, j, val = results_slot[slot]
                matrix[i][j] = val
                done += 1

    print(f"\n  Completed {done}/{total_pairs} pair(s)")
    return matrix


def run_perftest(pods, display_names):
    """Run latency tests (with --latency-size) and bandwidth tests (per --block-sizes)."""
    bw_sizes_str = ", ".join(_human_size(s) for s in _get_common().RDMA_BLOCK_SIZES)
    lat_size_str = _human_size(_get_common().RDMA_LATENCY_SIZE)
    print(f"\n{'=' * 60}")
    print("  PERFTEST (RDMA Latency & Bandwidth)")
    print(f"{'=' * 60}")
    print(f"Settings: duration={_get_common().PERFTEST_DURATION}s, iterations={_get_common().PERFTEST_ITERATIONS}, "
          f"latency_size={lat_size_str}, bw_sizes=[{bw_sizes_str}], base_port={_get_common().PERFTEST_PORT}")

    print("\nEnsuring perftest is available on all pods (parallel) ...")
    _c = _get_common()
    n = len(pods)
    _ensure_bufs = [_c._StreamingBuffer() for _ in range(n)]
    _ensure_done = [threading.Event() for _ in range(n)]

    def _ensure_worker(idx, pod_name, buf, done_evt):
        try:
            ensure_perftest(pod_name, out=lambda msg: buf.write(msg + "\n"))
        except SystemExit:
            buf.write(f"  FATAL: perftest setup failed on {pod_name}\n")
        except Exception as exc:
            buf.write(f"  ERROR on {pod_name}: {exc}\n")
        finally:
            done_evt.set()

    _ensure_threads = []
    for _idx, (_name, _ip) in enumerate(pods):
        _t = threading.Thread(
            target=_ensure_worker,
            args=(_idx, _name, _ensure_bufs[_idx], _ensure_done[_idx]),
            daemon=True,
        )
        _ensure_threads.append(_t)
        _t.start()

    for _idx in range(n):
        while not _ensure_done[_idx].is_set():
            _ensure_bufs[_idx].flush_new()
            _ensure_done[_idx].wait(timeout=0.1)
        _ensure_bufs[_idx].flush_all()

    for _t in _ensure_threads:
        _t.join(timeout=5)

    print("\nDetecting RDMA configuration on all pods ...")
    rdma_configs = {}
    for name, _ip in pods:
        rdma_configs[name] = detect_rdma_config(name)

    results = {}

    # Latency tests — run once with RDMA_LATENCY_SIZE
    for bin_idx, binary in enumerate(PERFTEST_BINARIES):
        if not _is_latency_test(binary):
            continue
        port = _get_common().PERFTEST_PORT + bin_idx
        matrix = _run_perftest_matrix(pods, display_names, rdma_configs,
                                      binary, port, _get_common().RDMA_LATENCY_SIZE)
        kind = _perftest_kind(binary)
        results[f"RDMA {kind} Latency (usec)"] = matrix

    # Bandwidth tests — run for each block size
    for size in _get_common().RDMA_BLOCK_SIZES:
        size_label = _human_size(size)
        for bin_idx, binary in enumerate(PERFTEST_BINARIES):
            if _is_latency_test(binary):
                continue
            port = _get_common().PERFTEST_PORT + bin_idx
            matrix = _run_perftest_matrix(pods, display_names, rdma_configs,
                                          binary, port, size)
            kind = _perftest_kind(binary)
            if len(_get_common().RDMA_BLOCK_SIZES) == 1:
                key = f"RDMA {kind} BW (Gb/s)"
            else:
                key = f"RDMA {kind} BW {size_label} (Gb/s)"
            results[key] = matrix

    return results


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-perftest.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _USAGE = """\
Usage: uv run run-tests-perftest.py [options]

Run RDMA perftest (latency & bandwidth) between Kubernetes inference pods.

Equivalent to: run-tests.sh -t perftest

Options:
  -b, --rdma-block-sizes SIZES
                        Comma-separated message sizes for RDMA bandwidth
                        (e.g. 64K,1M,1G).  Default: 1G.
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -g, --gid-index INDEX GID index for RoCE.  Auto-detected if not specified.
  -h, --help            Show this help message.
  -i, --install-deps    Install perftest build tools and build from source.
  -l, --label SELECTOR  Label selector to discover pods
                        (default: "llm-d.ai/inferenceServing=true").
  -n, --namespace NS    Kubernetes namespace for all kubectl commands.
  -p, --percentage-margin PCT
                        Flag results deviating more than PCT% from average
                        (default: 10).
  -r, --rdma-device DEVICE
                        RDMA device to use (e.g. mlx5_0).  Auto-detected if
                        not specified.
  -s, --rdma-latency-size SIZE
                        Message size for latency tests (default: 2 bytes).
  -v, --verbose         Print kubectl commands as they run.
  -x, --explain-verify  Run each explain command and verify output.
                        Implies --explain.
"""
    if "-h" in sys.argv or "--help" in sys.argv:
        print(_USAGE)
        sys.exit(0)

    _c = _get_common()
    _cfg = _c._parse_common_args(extra_flags={
        ("-r", "--rdma-device"):      ("OPT_DEVICE", True),
        ("-g", "--gid-index"):        ("OPT_GID_INDEX", True),
        ("-b", "--rdma-block-sizes"): ("_RDMA_BLOCK_SIZES_RAW", True),
        ("-s", "--rdma-latency-size"): ("_RDMA_LATENCY_SIZE_RAW", True),
    })

    # Parse size strings into integers
    _raw_bs = _cfg.pop("_RDMA_BLOCK_SIZES_RAW", None)
    if _raw_bs:
        _cfg["RDMA_BLOCK_SIZES"] = [_c._parse_size(s) for s in _raw_bs.split(",") if s.strip()]
    _raw_ls = _cfg.pop("_RDMA_LATENCY_SIZE_RAW", None)
    if _raw_ls:
        _cfg["RDMA_LATENCY_SIZE"] = _c._parse_size(_raw_ls)

    # Also pick up env-var overrides (same as run-tests.py)
    _env_dev = os.environ.get("PERFTEST_DEVICE", "").strip()
    if _env_dev and "OPT_DEVICE" not in _cfg:
        _cfg["OPT_DEVICE"] = _env_dev
    _env_gid = os.environ.get("PERFTEST_GID_INDEX", "").strip()
    if _env_gid and "OPT_GID_INDEX" not in _cfg:
        _cfg["OPT_GID_INDEX"] = _env_gid
    _cfg.setdefault("PERFTEST_DURATION",
                     int(os.environ.get("PERFTEST_DURATION", "5")))
    _cfg.setdefault("PERFTEST_ITERATIONS",
                     int(os.environ.get("PERFTEST_ITERATIONS", "1000")))
    _cfg.setdefault("PERFTEST_PORT",
                     int(os.environ.get("PERFTEST_PORT", "18515")))

    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    _results = run_perftest(_pods, _display_names)
    # Print a simple summary
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

