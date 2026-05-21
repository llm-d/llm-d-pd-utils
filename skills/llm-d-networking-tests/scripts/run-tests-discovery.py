# /// script
# requires-python = ">=3.10"
# ///
"""GPU topology discovery for Kubernetes inference pods.

Extracted from run-tests.py — validates GPU topology, NVLink/NVSwitch,
NUMA affinity, PCIe layout, RDMA NIC status, and environment variables
(NCCL, UCX, vLLM, CUDA, PyTorch distributed, NIXL) across pods.

Public API:
    run_topology_validation(pods, display_names) -> dict
"""

import re
import sys
import threading
from importlib import import_module

# Import shared utilities from the common module.
# Python module names with hyphens require importlib.
# Use lazy binding: _common is resolved on first access,
# avoiding circular import issues during module loading.
_common = None


def _get_common():
    global _common
    if _common is None:
        _common = import_module("run-tests-common")
    return _common


# Delegated to run-tests-common
def exec_in_pod(*args, **kwargs):
    return _get_common().exec_in_pod(*args, **kwargs)


def explain_cmd(*args, **kwargs):
    return _get_common().explain_cmd(*args, **kwargs)


def _kubectl_ns_args():
    return _get_common()._kubectl_ns_args()


def _kubectl_exec_prefix(pod_name="<POD>"):
    return _get_common()._kubectl_exec_prefix(pod_name)


def _print_verify_summary():
    return _get_common()._print_verify_summary()


def _strip_ansi(text):
    return _get_common()._strip_ansi(text)


# ---------------------------------------------------------------------------
# GPU Topology Validation
# ---------------------------------------------------------------------------

# Cache which AMD GPU tool is available per pod ("rocm-smi" or "amd-smi")
_amd_tool_cache = {}  # pod_name -> str


def _get_amd_tool(pod_name):
    """Return the AMD GPU tool name for a pod (default: 'rocm-smi')."""
    return _amd_tool_cache.get(pod_name, "rocm-smi")


def _detect_gpu_vendor(pod_name):
    """Detect GPU vendor on a pod: 'nvidia', 'amd', or 'unknown'.

    Tries nvidia-smi first, then rocm-smi, then amd-smi.
    """
    result = exec_in_pod(pod_name, ["bash", "-c", "command -v nvidia-smi >/dev/null 2>&1 && echo nvidia; "
                                     "command -v rocm-smi >/dev/null 2>&1 && echo rocm-smi; "
                                     "command -v amd-smi >/dev/null 2>&1 && echo amd-smi; true"],
                         timeout=10, use_debug=False)
    if result.returncode == 0:
        out = result.stdout.strip().lower()
        if "nvidia" in out:
            return "nvidia"
        if "rocm-smi" in out:
            _amd_tool_cache[pod_name] = "rocm-smi"
            return "amd"
        if "amd-smi" in out:
            _amd_tool_cache[pod_name] = "amd-smi"
            return "amd"
    return "unknown"


def _get_gpu_topo(pod_name, vendor):
    """Get GPU topology matrix. Returns (output_str, success).

    nvidia: nvidia-smi topo -m
    amd:    rocm-smi --showtopo / amd-smi topology
    """
    if vendor == "nvidia":
        result = exec_in_pod(pod_name, ["nvidia-smi", "topo", "-m"], timeout=30, use_debug=False)
    elif vendor == "amd":
        if _get_amd_tool(pod_name) == "amd-smi":
            result = exec_in_pod(pod_name, ["amd-smi", "topology"], timeout=30, use_debug=False)
        else:
            result = exec_in_pod(pod_name, ["rocm-smi", "--showtopo"], timeout=30, use_debug=False)
    else:
        return None, False
    if result.returncode != 0:
        return None, False
    return _strip_ansi(result.stdout.strip()), True


def _get_gpu_link_status(pod_name, vendor):
    """Get GPU interconnect link status. Returns (output_str, success).

    nvidia: nvidia-smi nvlink --status
    amd:    rocm-smi --showtopoweight / amd-smi topology --weight (xGMI/Infinity Fabric link weights)
    """
    if vendor == "nvidia":
        result = exec_in_pod(pod_name, ["nvidia-smi", "nvlink", "--status"], timeout=30, use_debug=False)
    elif vendor == "amd":
        if _get_amd_tool(pod_name) == "amd-smi":
            result = exec_in_pod(pod_name, ["amd-smi", "topology", "--weight"], timeout=30, use_debug=False)
        else:
            result = exec_in_pod(pod_name, ["rocm-smi", "--showtopoweight"], timeout=30, use_debug=False)
    else:
        return None, False
    if result.returncode != 0:
        return None, False
    return _strip_ansi(result.stdout.strip()), True


def _get_gpu_bus_ids(pod_name, vendor):
    """Get GPU index and PCI bus ID pairs. Returns list of (index_int, bus_id_str).

    nvidia: nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader
    amd:    rocm-smi --showbus / amd-smi static --bus (parse GPU ID and bus ID)
    """
    pairs = []
    if vendor == "nvidia":
        result = exec_in_pod(
            pod_name,
            ["bash", "-c", "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null"],
            timeout=15, use_debug=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if "," in line:
                    idx_s, bus = line.split(",", 1)
                    idx_s = idx_s.strip()
                    bus = bus.strip().lower()
                    # Normalize 8-digit domain to 4-digit
                    bus = re.sub(r'^0{4,}', '0000', bus)
                    try:
                        pairs.append((int(idx_s), bus))
                    except ValueError:
                        pass
    elif vendor == "amd":
        if _get_amd_tool(pod_name) == "amd-smi":
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", "amd-smi static --bus 2>/dev/null"],
                timeout=15, use_debug=False,
            )
        else:
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", "rocm-smi --showbus 2>/dev/null"],
                timeout=15, use_debug=False,
            )
        if result.returncode == 0 and result.stdout.strip():
            # rocm-smi --showbus format:  GPU[0] : PCI Bus: 0000:03:00.0
            # amd-smi static --bus format: GPU: 0  BUS: 0000:03:00.0
            for line in _strip_ansi(result.stdout).strip().split("\n"):
                m = re.match(r'.*GPU\[(\d+)\].*:\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)', line)
                if not m:
                    m = re.match(r'.*GPU:\s*(\d+).*BUS:\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)', line, re.IGNORECASE)
                if m:
                    pairs.append((int(m.group(1)), m.group(2).lower()))
    return pairs


def _parse_rocm_topo(output):
    """Parse rocm-smi --showtopo output into connections dict and gpu_labels.

    rocm-smi topology output format varies but typically shows a weight matrix:
        GPU0  GPU1  GPU2 ...
    GPU0   0    15    15
    GPU1  15     0    15
    ...
    Or shows link type: XGMI, PCIE, etc.

    Returns (connections, gpu_labels) in the same format as _parse_topo_matrix.
    """
    if not output:
        return {}, []
    lines = output.strip().split("\n")
    connections = {}
    gpu_labels = []

    # Try to find a matrix with GPU labels
    # Look for lines containing "XGMI" or "PCIE" or weight numbers
    # Also handle the "Link Type" section of --showtopo
    in_type_section = False
    in_weight_section = False
    header_gpus = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect section headers
        if "type" in stripped.lower() and "link" in stripped.lower():
            in_type_section = True
            in_weight_section = False
            header_gpus = []
            continue
        if "weight" in stripped.lower() or "hop" in stripped.lower():
            in_type_section = False
            in_weight_section = True
            header_gpus = []
            continue

        # Header row with GPU labels
        if in_type_section or in_weight_section:
            parts = stripped.split()
            if parts and all(p.startswith("GPU") and p[3:].isdigit() for p in parts):
                header_gpus = parts
                continue
            # Data row: GPU0  XGMI  PCIE  XGMI ...
            if parts and parts[0].startswith("GPU") and parts[0][3:].isdigit() and header_gpus:
                row_gpu = parts[0]
                if row_gpu not in gpu_labels:
                    gpu_labels.append(row_gpu)
                for col_idx, col_gpu in enumerate(header_gpus):
                    val_idx = col_idx + 1
                    if val_idx < len(parts):
                        val = parts[val_idx].upper()
                        if in_type_section:
                            # Map XGMI -> NV-equivalent for consistent analysis
                            if val == "XGMI":
                                connections[(row_gpu, col_gpu)] = "XGMI"
                            elif val == "PCIE":
                                connections[(row_gpu, col_gpu)] = "SYS"
                            elif val == "0" or val == "X":
                                connections[(row_gpu, col_gpu)] = "X"
                            else:
                                connections[(row_gpu, col_gpu)] = val

    # If no matrix found, try simpler parsing
    if not connections:
        for line in lines:
            stripped = line.strip()
            # Look for: GPU[0] -> GPU[1]: XGMI
            m = re.match(r'GPU\[?(\d+)\]?\s*->\s*GPU\[?(\d+)\]?\s*:\s*(\w+)', stripped)
            if m:
                g1 = f"GPU{m.group(1)}"
                g2 = f"GPU{m.group(2)}"
                link = m.group(3).upper()
                connections[(g1, g2)] = link
                if g1 not in gpu_labels:
                    gpu_labels.append(g1)
                if g2 not in gpu_labels:
                    gpu_labels.append(g2)

    return connections, gpu_labels


def _parse_rocm_link_status(output):
    """Parse rocm-smi --showtopoweight output for xGMI link info.

    Returns (active_count, inactive_count) analogous to NVLink status.
    """
    if not output:
        return 0, 0
    active = 0
    inactive = 0
    for line in output.split("\n"):
        lower = line.lower()
        # Count non-zero weights as active xGMI links
        if "xgmi" in lower:
            active += 1
        # Look for weight values — non-zero between GPUs = active link
        parts = line.strip().split()
        if parts and parts[0].startswith("GPU"):
            for val in parts[1:]:
                try:
                    w = int(val)
                    if w > 0 and w < 1000:  # reasonable weight range
                        active += 1
                except ValueError:
                    pass
    return active, inactive


def _run_topo_cmd(pod_name, cmd_args, label):
    """Run a topology command on a pod and return (stdout, success)."""
    result = exec_in_pod(pod_name, cmd_args, timeout=30, use_debug=False)
    if result.returncode != 0:
        return None, False
    return _strip_ansi(result.stdout.strip()), True


def _parse_nvlink_status(output):
    """Parse nvidia-smi nvlink --status output.

    Returns (active_count, inactive_count) for NVLink connections.
    """
    if not output:
        return 0, 0
    active = 0
    inactive = 0
    for line in output.split("\n"):
        lower = line.lower()
        if "active" in lower and "link" in lower:
            active += 1
        elif "inactive" in lower and "link" in lower:
            inactive += 1
    return active, inactive


def _parse_topo_matrix(output):
    """Parse nvidia-smi topo -m output.

    Returns a dict mapping (gpu_i, gpu_j) -> connection_type (e.g. NV18, SYS, PIX, PHB, NV#, etc.)
    and a list of GPU labels.
    """
    if not output:
        return {}, []
    lines = output.strip().split("\n")
    # Find the header line (starts with GPU or contains GPU0)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("GPU"):
            # Check if it looks like a header row with GPU0, GPU1, etc.
            parts = line.split()
            if len(parts) > 1 and any(p.startswith("GPU") and p[3:].isdigit() for p in parts[1:]):
                header_idx = i
                break
            # Could also be a data row starting with GPU0, GPU1 etc.
            if len(parts) > 1 and parts[0].startswith("GPU") and parts[0][3:].isdigit():
                if header_idx is None:
                    # Look back for column headers
                    if i > 0:
                        prev = lines[i - 1].split()
                        if any(p.startswith("GPU") for p in prev):
                            header_idx = i - 1
                            break
                    header_idx = i
                    break

    if header_idx is None:
        return {}, []

    header_parts = lines[header_idx].split()
    # Column headers: first column is label, rest are GPU0..GPUN (possibly with CPU/NIC after)
    gpu_cols = []
    for p in header_parts:
        if p.startswith("GPU") and len(p) > 3 and p[3:].isdigit():
            gpu_cols.append(p)

    connections = {}
    gpu_labels = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        # Legend section starts after the matrix
        if line.strip().startswith("Legend:") or line.strip().startswith("X ="):
            break
        parts = line.split()
        if not parts or not (parts[0].startswith("GPU") and len(parts[0]) > 3 and parts[0][3:].isdigit()):
            continue
        row_gpu = parts[0]
        gpu_labels.append(row_gpu)
        # The connection values follow the GPU label
        for col_idx, col_gpu in enumerate(gpu_cols):
            val_idx = col_idx + 1  # offset by 1 for the row label
            if val_idx < len(parts):
                connections[(row_gpu, col_gpu)] = parts[val_idx]

    return connections, gpu_labels


def _check_nvlink_topology(connections, gpu_labels):
    """Check if GPUs are connected via NVLink/NVSwitch/xGMI vs PCIe only.

    Recognizes both NVIDIA (NV##, NVL) and AMD (XGMI) high-bandwidth links.

    Returns (has_nvlink, summary_str).
    """
    if not connections or not gpu_labels:
        return False, "no topology data"

    nvlink_pairs = []
    pcie_only_pairs = []
    pcie_types = {"PIX", "PHB", "PXB", "SYS", "NODE"}

    for (g1, g2), conn in connections.items():
        if g1 == g2 or conn == "X":
            continue
        if g1 > g2:
            continue  # avoid counting pairs twice
        # NVLink (NV#, NVL) or AMD xGMI — high-bandwidth GPU interconnect
        if conn.startswith("NV") or conn == "NVL" or conn == "XGMI":
            nvlink_pairs.append((g1, g2, conn))
        elif conn in pcie_types:
            pcie_only_pairs.append((g1, g2, conn))

    if nvlink_pairs:
        return True, f"{len(nvlink_pairs)} GPU pair(s) connected via NVLink/NVSwitch"
    elif pcie_only_pairs:
        return False, f"all {len(pcie_only_pairs)} GPU pair(s) connected via PCIe only"
    else:
        return False, "could not determine GPU interconnect type"


def _check_hop_distance(connections, gpu_labels):
    """Check for excessive hop distances between GPUs.

    nvidia-smi topo -m connection types ranked from best to worst:
      NV## / NVL  — NVLink (direct or via NVSwitch), best
      PIX         — same PCIe switch
      PXB         — cross PCIe switch but same PCIe root complex
      PHB         — same NUMA node but different PCIe root complex
      NODE        — same NUMA node (legacy alias, treated like PHB)
      SYS         — cross NUMA / cross-socket (QPI/UPI hop)

    Returns list of warning strings for sub-optimal connections.
    """
    if not connections or len(gpu_labels) < 2:
        return []

    # Severity ranking: lower is better.  Anything above the threshold warns.
    SEVERITY = {"NV": 0, "NVL": 0, "XGMI": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 3, "SYS": 4}
    LABEL = {"NV": "NVLink", "NVL": "NVLink", "XGMI": "xGMI/Infinity Fabric",
             "PIX": "same PCIe switch",
             "PXB": "cross PCIe switch", "PHB": "cross PCIe root (same NUMA)",
             "NODE": "same NUMA node", "SYS": "cross-NUMA/cross-socket"}
    # Warn when severity >= this value (PHB and above)
    WARN_THRESHOLD = 3

    def _conn_severity(conn):
        if conn.startswith("NV"):
            return 0, "NVLink"
        if conn == "XGMI":
            return 0, "xGMI/Infinity Fabric"
        return SEVERITY.get(conn, 5), LABEL.get(conn, conn)

    warns = []
    seen = set()
    excessive_hops = {}   # conn_type -> list of (g1, g2)
    for (g1, g2), conn in connections.items():
        if g1 == g2 or conn == "X":
            continue
        pair = tuple(sorted([g1, g2]))
        if pair in seen:
            continue
        seen.add(pair)
        sev, desc = _conn_severity(conn)
        if sev >= WARN_THRESHOLD:
            excessive_hops.setdefault(conn, []).append(pair)

    for conn_type in sorted(excessive_hops, key=lambda c: _conn_severity(c)[0], reverse=True):
        pairs = excessive_hops[conn_type]
        _, desc = _conn_severity(conn_type)
        pair_strs = [f"{a}<->{b}" for a, b in pairs]
        if len(pair_strs) <= 4:
            detail = ", ".join(pair_strs)
        else:
            detail = f"{len(pair_strs)} pairs"
        warns.append(f"{conn_type} ({desc}): {detail}")

    return warns


def _check_nvswitch_domains(connections, gpu_labels):
    """Detect whether GPUs belong to different NVSwitch domains.

    When all GPUs are in the same NVSwitch domain, every GPU pair shows a
    uniform NV## link count (e.g. NV18 everywhere).  If the NVLink width
    varies significantly between pairs, GPUs are likely split across
    separate NVSwitch domains — suboptimal for collective operations.

    Also detects a mixed topology where some GPU pairs use NVLink and
    others fall back to PCIe, which is a clear domain split.

    Returns (is_uniform, warnings_list).
    """
    if not connections or len(gpu_labels) < 2:
        return True, []

    # Collect NVLink widths and PCIe-fallback pairs
    nvlink_widths = {}   # (g1,g2) -> int  (the ## in NV##)
    pcie_pairs = []      # pairs that use PCIe despite NVLink existing elsewhere
    seen = set()

    has_any_nvlink = False
    for (g1, g2), conn in connections.items():
        if g1 == g2 or conn == "X":
            continue
        pair = tuple(sorted([g1, g2]))
        if pair in seen:
            continue
        seen.add(pair)
        if conn.startswith("NV") or conn == "XGMI":
            has_any_nvlink = True
            # Extract numeric width: NV18 -> 18, NVL/XGMI -> -1 (present but unnumbered)
            if conn.startswith("NV"):
                suffix = conn[2:]
                width = int(suffix) if suffix.isdigit() else -1
            else:
                width = -1  # XGMI — no lane count
            nvlink_widths[pair] = width
        else:
            pcie_pairs.append((pair, conn))

    if not has_any_nvlink:
        return True, []  # no NVLink at all, nothing to check

    warns = []

    # Check 1: Some pairs use NVLink, others fall back to PCIe
    if pcie_pairs:
        fallback_strs = [f"{a}<->{b} ({conn})" for (a, b), conn in pcie_pairs]
        if len(fallback_strs) <= 4:
            detail = ", ".join(fallback_strs)
        else:
            detail = f"{len(fallback_strs)} pairs"
        warns.append(
            f"NVSwitch domain split: {len(nvlink_widths)} pair(s) use NVLink but "
            f"{len(pcie_pairs)} pair(s) fall back to PCIe: {detail}"
        )

    # Check 2: Non-uniform NVLink widths (e.g. some NV18, some NV12)
    numeric_widths = [w for w in nvlink_widths.values() if w > 0]
    if numeric_widths:
        unique_widths = set(numeric_widths)
        if len(unique_widths) > 1:
            width_counts = {}
            for w in numeric_widths:
                width_counts[w] = width_counts.get(w, 0) + 1
            breakdown = ", ".join(f"NV{w}: {c} pair(s)" for w, c in sorted(width_counts.items()))
            warns.append(
                f"Non-uniform NVLink widths across GPU pairs ({breakdown}) — "
                f"GPUs may span different NVSwitch domains"
            )

    return len(warns) == 0, warns


def _fetch_nccl_env(pod_name):
    """Fetch NCCL/RDMA/vLLM/CUDA/PyTorch/NIXL environment variables from a pod.

    Returns a dict of var_name -> value for all matching env vars.
    """
    # Grep for all inference-relevant env vars in one exec call
    script = (
        "env | grep -E "
        "'^(NCCL_|RDMA_|UCX_|VLLM_|CUDA_|TORCH_|MASTER_|NIXL_|RANK=|WORLD_SIZE=|LOCAL_RANK=)' "
        "| sort; true"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", script], timeout=15, use_debug=False)
    env_vars = {}
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip()
    return env_vars


def _fetch_rdma_devices(pod_name):
    """List available RDMA devices, using ibv_devices if available, else sysfs.

    Returns a list of device names (e.g. ['mlx5_0', 'mlx5_1']).
    """
    # Try ibv_devices first — provides authoritative list from libibverbs
    # Filter out header lines (device, ------) keeping only actual device names
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         "ibv_devices 2>/dev/null | awk '/^\\s+/{print $1}' | grep -vE '^(-|device$)'; true"],
        timeout=15, use_debug=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        devs = sorted(d for d in result.stdout.strip().split()
                       if not d.startswith("-") and d != "device")
        if devs:
            return devs

    # Fallback to sysfs
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", "ls /sys/class/infiniband/ 2>/dev/null; true"],
        timeout=15, use_debug=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return sorted(result.stdout.strip().split())
    return []


def _fetch_rdma_nic_status(pod_name, rdma_devices):
    """Fetch detailed RDMA NIC status for each device.

    Uses ibstat if available (richer output including link layer, rate, state),
    otherwise falls back to sysfs reads.

    Returns a dict: {device_name: {state, phys_state, rate, link_layer, fw_ver, ...}}
    """
    if not rdma_devices:
        return {}

    # Try ibstat first — one call for all devices
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", "command -v ibstat >/dev/null 2>&1 && echo HAS_IBSTAT || echo NO_IBSTAT"],
        timeout=10, use_debug=False,
    )
    has_ibstat = result.returncode == 0 and "HAS_IBSTAT" in result.stdout

    nic_status = {}

    if has_ibstat:
        # ibstat gives detailed per-device output; query all devices at once
        result = exec_in_pod(
            pod_name,
            ["bash", "-c", f"ibstat {' '.join(rdma_devices)} 2>/dev/null; true"],
            timeout=30, use_debug=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            current_dev = None
            current_port = None
            info = {}
            for line in result.stdout.split("\n"):
                stripped = line.strip()
                # Device header: "CA 'mlx5_0'"
                if stripped.startswith("CA '") or stripped.startswith("CA '"):
                    if current_dev and info:
                        nic_status[current_dev] = info
                    dev_match = stripped.split("'")
                    current_dev = dev_match[1] if len(dev_match) >= 2 else None
                    info = {"source": "ibstat"}
                    current_port = None
                elif stripped.startswith("Port ") and current_dev:
                    current_port = stripped.rstrip(":")
                elif ":" in stripped and current_dev:
                    key, _, val = stripped.partition(":")
                    key = key.strip().lower()
                    val = val.strip()
                    if key == "state":
                        info["state"] = val
                    elif key == "physical state":
                        info["phys_state"] = val
                    elif key == "rate":
                        info["rate"] = val
                    elif key == "base lid":
                        info["lid"] = val
                    elif key == "sm lid":
                        info["sm_lid"] = val
                    elif key == "link layer":
                        info["link_layer"] = val
                    elif key == "fw ver":
                        info["fw_ver"] = val
                    elif key == "node guid":
                        info["node_guid"] = val
                    elif key == "ca type":
                        info["ca_type"] = val
                    elif key == "number of ports":
                        info["num_ports"] = val
            if current_dev and info:
                nic_status[current_dev] = info

    # Fallback or fill gaps using sysfs — batched in a single exec
    missing = [d for d in rdma_devices if d not in nic_status]
    if missing or not has_ibstat:
        devices_to_query = missing if has_ibstat else rdma_devices
        if devices_to_query:
            # Build a batched sysfs read script
            reads = []
            for dev in devices_to_query:
                reads.append(
                    f'echo "DEV:{dev}";'
                    f'cat /sys/class/infiniband/{dev}/ports/1/state 2>/dev/null || echo "?";'
                    f'cat /sys/class/infiniband/{dev}/ports/1/phys_state 2>/dev/null || echo "?";'
                    f'cat /sys/class/infiniband/{dev}/ports/1/rate 2>/dev/null || echo "?";'
                    f'cat /sys/class/infiniband/{dev}/ports/1/link_layer 2>/dev/null || echo "?";'
                    f'cat /sys/class/infiniband/{dev}/fw_ver 2>/dev/null || echo "?";'
                )
            script = " ".join(reads) + " true"
            result = exec_in_pod(
                pod_name, ["bash", "-c", script], timeout=30, use_debug=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("DEV:"):
                        dev = line[4:]
                        info = nic_status.get(dev, {"source": "sysfs"})
                        # Read next 5 values: state, phys_state, rate, link_layer, fw_ver
                        if i + 1 < len(lines):
                            raw_state = lines[i + 1].strip()
                            # sysfs state format: "4: ACTIVE" or just "ACTIVE"
                            info.setdefault("state", raw_state.split(":", 1)[-1].strip() if ":" in raw_state else raw_state)
                        if i + 2 < len(lines):
                            raw_phys = lines[i + 2].strip()
                            info.setdefault("phys_state", raw_phys.split(":", 1)[-1].strip() if ":" in raw_phys else raw_phys)
                        if i + 3 < len(lines):
                            info.setdefault("rate", lines[i + 3].strip())
                        if i + 4 < len(lines):
                            info.setdefault("link_layer", lines[i + 4].strip())
                        if i + 5 < len(lines):
                            info.setdefault("fw_ver", lines[i + 5].strip())
                        nic_status[dev] = info
                        i += 6
                    else:
                        i += 1

    return nic_status


def _fetch_net_interfaces(pod_name):
    """List network interfaces from the pod.

    Returns a list of interface names.
    """
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", "ls /sys/class/net/ 2>/dev/null; true"],
        timeout=15, use_debug=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return sorted(result.stdout.strip().split())
    return []


def _validate_nccl_env(env_vars, rdma_devices, net_interfaces):
    """Validate NCCL environment variables against actual pod hardware.

    Returns (findings, warnings) where findings is a list of info strings
    and warnings is a list of warning strings.
    """
    findings = []
    warns = []

    # --- NCCL_IB_DISABLE ---
    ib_disable = env_vars.get("NCCL_IB_DISABLE")
    if ib_disable is not None:
        if ib_disable == "1":
            findings.append("NCCL_IB_DISABLE=1 — InfiniBand/RoCE disabled")
            if rdma_devices:
                warns.append(
                    f"NCCL_IB_DISABLE=1 but RDMA devices present: "
                    f"{', '.join(rdma_devices)} — IB/RoCE will not be used"
                )
        else:
            findings.append(f"NCCL_IB_DISABLE={ib_disable} — InfiniBand/RoCE enabled")
    else:
        findings.append("NCCL_IB_DISABLE not set (IB/RoCE enabled by default)")

    # --- NCCL_IB_HCA ---
    ib_hca = env_vars.get("NCCL_IB_HCA")
    if ib_hca is not None:
        findings.append(f"NCCL_IB_HCA={ib_hca}")
        # Parse HCA list: entries can be =exact, ^exclude, prefix, or device:port
        hca_entries = [e.strip() for e in ib_hca.split(",") if e.strip()]
        for entry in hca_entries:
            # Strip leading = or ^ and trailing :port
            clean = entry.lstrip("=^")
            if ":" in clean:
                clean = clean.split(":")[0]
            # Check if this device exists (prefix match for entries without =)
            if entry.startswith("^"):
                # Exclusion — just note it
                continue
            if entry.startswith("="):
                # Exact match
                if clean not in rdma_devices:
                    warns.append(
                        f"NCCL_IB_HCA specifies ={clean} but device not found "
                        f"(available: {', '.join(rdma_devices) or 'none'})"
                    )
            else:
                # Prefix match
                matched = [d for d in rdma_devices if d.startswith(clean)]
                if not matched:
                    warns.append(
                        f"NCCL_IB_HCA specifies '{entry}' but no RDMA device "
                        f"matches prefix '{clean}' "
                        f"(available: {', '.join(rdma_devices) or 'none'})"
                    )
    else:
        if rdma_devices:
            findings.append(
                f"NCCL_IB_HCA not set — NCCL will use all RDMA devices: "
                f"{', '.join(rdma_devices)}"
            )
        else:
            findings.append("NCCL_IB_HCA not set (no RDMA devices found)")

    # --- NCCL_EXCLUDE_IB_HCA ---
    exclude_hca = env_vars.get("NCCL_EXCLUDE_IB_HCA")
    if exclude_hca is not None:
        findings.append(f"NCCL_EXCLUDE_IB_HCA={exclude_hca}")
        if ib_hca is not None:
            warns.append(
                "Both NCCL_IB_HCA and NCCL_EXCLUDE_IB_HCA are set — "
                "NCCL_IB_HCA takes precedence, NCCL_EXCLUDE_IB_HCA is ignored"
            )

    # --- NCCL_SOCKET_IFNAME ---
    socket_ifname = env_vars.get("NCCL_SOCKET_IFNAME")
    if socket_ifname is not None:
        findings.append(f"NCCL_SOCKET_IFNAME={socket_ifname}")
        # Can be a prefix or comma-separated list of prefixes; ^ means exclude
        ifname_entries = [e.strip() for e in socket_ifname.split(",") if e.strip()]
        for entry in ifname_entries:
            if entry.startswith("^"):
                continue  # exclusion
            # Check prefix match against interfaces
            matched = [i for i in net_interfaces if i.startswith(entry)]
            if not matched:
                warns.append(
                    f"NCCL_SOCKET_IFNAME specifies '{entry}' but no network "
                    f"interface matches (available: {', '.join(net_interfaces) or 'none'})"
                )
    else:
        findings.append("NCCL_SOCKET_IFNAME not set (NCCL will auto-select)")

    # --- NCCL_NET ---
    nccl_net = env_vars.get("NCCL_NET")
    if nccl_net is not None:
        findings.append(f"NCCL_NET={nccl_net}")

    # --- NCCL_IB_GID_INDEX ---
    gid_index = env_vars.get("NCCL_IB_GID_INDEX")
    if gid_index is not None:
        findings.append(f"NCCL_IB_GID_INDEX={gid_index}")

    # --- NCCL_IB_TIMEOUT ---
    ib_timeout = env_vars.get("NCCL_IB_TIMEOUT")
    if ib_timeout is not None:
        findings.append(f"NCCL_IB_TIMEOUT={ib_timeout}")

    # --- NCCL_IB_TC ---
    ib_tc = env_vars.get("NCCL_IB_TC")
    if ib_tc is not None:
        findings.append(f"NCCL_IB_TC={ib_tc}")

    # --- NCCL_NET_GDR_LEVEL ---
    gdr = env_vars.get("NCCL_NET_GDR_LEVEL")
    if gdr is not None:
        findings.append(f"NCCL_NET_GDR_LEVEL={gdr}")

    # --- NCCL_VERSION ---
    nccl_ver = env_vars.get("NCCL_VERSION")
    if nccl_ver is not None:
        findings.append(f"NCCL_VERSION={nccl_ver}")

    # =====================================================================
    # UCX environment variables — used by frameworks like PyTorch, DeepSpeed,
    # and Horovod when UCX is the transport layer (instead of or alongside NCCL).
    # UCX controls which transports, devices, and protocols are used for
    # GPU-GPU and node-to-node communication.
    # =====================================================================
    ucx_vars = {k: v for k, v in env_vars.items() if k.startswith("UCX_")}
    if ucx_vars:
        findings.append(f"UCX: {len(ucx_vars)} variable(s) set")

    # --- UCX_TLS (transport layers) ---
    ucx_tls = env_vars.get("UCX_TLS")
    if ucx_tls is not None:
        findings.append(f"UCX_TLS={ucx_tls}")
        tls_list = [t.strip().lower() for t in ucx_tls.split(",")]
        # Topology impact analysis
        if "rc" in tls_list or "rc_v" in tls_list or "rc_x" in tls_list:
            findings.append("  -> rc/rc_v/rc_x: Reliable Connected IB verbs (RDMA, low latency)")
        if "ud" in tls_list or "ud_v" in tls_list or "ud_x" in tls_list:
            findings.append("  -> ud/ud_v/ud_x: Unreliable Datagram IB verbs (RDMA, connectionless)")
        if "dc" in tls_list or "dc_x" in tls_list:
            findings.append("  -> dc/dc_x: Dynamic Connected IB (scalable RDMA, fewer QPs)")
        if "tcp" in tls_list:
            findings.append("  -> tcp: TCP sockets (fallback, no RDMA — high latency)")
            if rdma_devices:
                warns.append(
                    "UCX_TLS includes 'tcp' but RDMA devices are present — "
                    "TCP is a slow fallback; consider rc_v or dc_x for RDMA transport"
                )
        if "cuda_copy" in tls_list:
            findings.append("  -> cuda_copy: GPU memory staging via cudaMemcpy")
        if "cuda_ipc" in tls_list:
            findings.append("  -> cuda_ipc: GPU-GPU direct transfer via CUDA IPC (same node)")
        if "gdr_copy" in tls_list:
            findings.append("  -> gdr_copy: GPUDirect RDMA copy (GPU memory <-> host via BAR)")
        if "sm" in tls_list or "shm" in tls_list:
            findings.append("  -> sm/shm: shared memory (intra-node only)")
        if "self" in tls_list:
            findings.append("  -> self: loopback (same process)")
        # Check for missing RDMA transports when RDMA devices exist
        rdma_tls = {"rc", "rc_v", "rc_x", "ud", "ud_v", "ud_x", "dc", "dc_x"}
        if rdma_devices and not (set(tls_list) & rdma_tls):
            warns.append(
                f"UCX_TLS={ucx_tls} has no RDMA transport (rc/ud/dc) but "
                f"RDMA devices are present — inter-node traffic will not use RDMA"
            )
    else:
        if ucx_vars:
            findings.append("UCX_TLS not set (UCX auto-selects transports)")

    # --- UCX_NET_DEVICES (RDMA device selection) ---
    ucx_net_devs = env_vars.get("UCX_NET_DEVICES")
    if ucx_net_devs is not None:
        findings.append(f"UCX_NET_DEVICES={ucx_net_devs}")
        # Validate devices exist
        dev_entries = [d.strip() for d in ucx_net_devs.split(",") if d.strip()]
        for entry in dev_entries:
            # UCX device format: mlx5_0:1 (device:port) or just mlx5_0
            dev_name = entry.split(":")[0]
            if dev_name not in rdma_devices and dev_name != "all":
                warns.append(
                    f"UCX_NET_DEVICES specifies '{entry}' but device '{dev_name}' "
                    f"not found (available: {', '.join(rdma_devices) or 'none'})"
                )

    # --- UCX_IB_PCI_BW (expected IB PCI bandwidth) ---
    ucx_ib_pci_bw = env_vars.get("UCX_IB_PCI_BW")
    if ucx_ib_pci_bw is not None:
        findings.append(f"UCX_IB_PCI_BW={ucx_ib_pci_bw} (PCI bandwidth hint for UCX transport selection)")

    # --- UCX_RNDV_THRESH (rendezvous threshold) ---
    ucx_rndv = env_vars.get("UCX_RNDV_THRESH")
    if ucx_rndv is not None:
        findings.append(f"UCX_RNDV_THRESH={ucx_rndv} (messages above this use rendezvous/zero-copy)")

    # --- UCX_MAX_RNDV_RAILS (multi-rail) ---
    ucx_rails = env_vars.get("UCX_MAX_RNDV_RAILS")
    if ucx_rails is not None:
        findings.append(f"UCX_MAX_RNDV_RAILS={ucx_rails} (max RDMA devices used in parallel for large messages)")

    # --- UCX_MEMTYPE_CACHE ---
    ucx_memcache = env_vars.get("UCX_MEMTYPE_CACHE")
    if ucx_memcache is not None:
        findings.append(f"UCX_MEMTYPE_CACHE={ucx_memcache}")
        if ucx_memcache.lower() in ("n", "no", "0", "false"):
            findings.append("  -> memory type cache disabled — may reduce GPU memory registration performance")

    # --- UCX_ZCOPY_THRESH (zero-copy threshold) ---
    ucx_zcopy = env_vars.get("UCX_ZCOPY_THRESH")
    if ucx_zcopy is not None:
        findings.append(f"UCX_ZCOPY_THRESH={ucx_zcopy} (threshold for zero-copy RDMA sends)")

    # --- UCX_IB_GID_INDEX ---
    ucx_gid = env_vars.get("UCX_IB_GID_INDEX")
    if ucx_gid is not None:
        findings.append(f"UCX_IB_GID_INDEX={ucx_gid} (RoCE GID index for UCX)")

    # --- UCX_IB_TRAFFIC_CLASS ---
    ucx_tc = env_vars.get("UCX_IB_TRAFFIC_CLASS")
    if ucx_tc is not None:
        findings.append(f"UCX_IB_TRAFFIC_CLASS={ucx_tc} (QoS traffic class for IB)")

    # --- UCX_IB_SL (Service Level) ---
    ucx_sl = env_vars.get("UCX_IB_SL")
    if ucx_sl is not None:
        findings.append(f"UCX_IB_SL={ucx_sl} (IB Service Level for QoS routing)")

    # --- Topology-relevant UCX interactions ---
    # Check NCCL_NET + UCX interaction
    if nccl_net is not None and nccl_net.lower() == "ucx":
        findings.append("NCCL_NET=UCX — NCCL uses UCX as its network transport plugin")
        if not ucx_tls:
            findings.append("  -> UCX_TLS not set — UCX will auto-select transports")
        if not ucx_net_devs:
            findings.append("  -> UCX_NET_DEVICES not set — UCX will auto-select RDMA devices")

    # Dump any remaining UCX vars not specifically handled above
    _handled_ucx = {"UCX_TLS", "UCX_NET_DEVICES", "UCX_IB_PCI_BW", "UCX_RNDV_THRESH",
                    "UCX_MAX_RNDV_RAILS", "UCX_MEMTYPE_CACHE", "UCX_ZCOPY_THRESH",
                    "UCX_IB_GID_INDEX", "UCX_IB_TRAFFIC_CLASS", "UCX_IB_SL"}
    other_ucx = {k: v for k, v in ucx_vars.items() if k not in _handled_ucx}
    if other_ucx:
        for k, v in sorted(other_ucx.items()):
            findings.append(f"{k}={v}")

    return findings, warns


def _validate_vllm_cuda_env(env_vars, gpu_count, rdma_devices, has_nvlink):
    """Validate vLLM, CUDA, PyTorch distributed, and NIXL env vars for inference.

    Returns (findings, warnings) where findings is a list of info strings
    and warnings is a list of warning strings.
    """
    findings = []
    warns = []

    # =====================================================================
    # CUDA environment variables
    # =====================================================================
    cuda_vars = {k: v for k, v in env_vars.items() if k.startswith("CUDA_")}
    if cuda_vars:
        findings.append(f"CUDA: {len(cuda_vars)} variable(s) set")

    # --- CUDA_VISIBLE_DEVICES ---
    cuda_visible = env_vars.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        findings.append(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
        try:
            visible_count = len([d.strip() for d in cuda_visible.split(",") if d.strip()])
            if gpu_count and visible_count != gpu_count:
                warns.append(
                    f"CUDA_VISIBLE_DEVICES lists {visible_count} device(s) but "
                    f"GPU topology shows {gpu_count} — potential mismatch"
                )
        except Exception:
            pass

    # --- CUDA_DEVICE_ORDER ---
    cuda_dev_order = env_vars.get("CUDA_DEVICE_ORDER")
    if cuda_dev_order is not None:
        findings.append(f"CUDA_DEVICE_ORDER={cuda_dev_order}")
        if cuda_dev_order != "PCI_BUS_ID":
            warns.append(
                f"CUDA_DEVICE_ORDER={cuda_dev_order} — recommend PCI_BUS_ID "
                f"for deterministic GPU ordering that matches nvidia-smi"
            )
    else:
        if gpu_count and gpu_count > 1:
            findings.append(
                "CUDA_DEVICE_ORDER not set — defaults to FASTEST_FIRST, "
                "recommend PCI_BUS_ID for stable ordering"
            )

    # --- CUDA_MODULE_LOADING ---
    cuda_module = env_vars.get("CUDA_MODULE_LOADING")
    if cuda_module is not None:
        findings.append(f"CUDA_MODULE_LOADING={cuda_module}")
        if cuda_module.upper() == "EAGER":
            findings.append(
                "  -> EAGER: all CUDA modules loaded at startup — "
                "LAZY reduces memory and startup time for inference"
            )
    else:
        findings.append("CUDA_MODULE_LOADING not set (defaults to EAGER)")

    # --- CUDA_LAUNCH_BLOCKING ---
    cuda_blocking = env_vars.get("CUDA_LAUNCH_BLOCKING")
    if cuda_blocking == "1":
        warns.append(
            "CUDA_LAUNCH_BLOCKING=1 — all CUDA operations run synchronously, "
            "severe performance penalty; should only be set for debugging"
        )

    # =====================================================================
    # vLLM environment variables
    # =====================================================================
    vllm_vars = {k: v for k, v in env_vars.items() if k.startswith("VLLM_")}
    if vllm_vars:
        findings.append(f"vLLM: {len(vllm_vars)} variable(s) set")

    # --- VLLM_HOST_IP ---
    vllm_host_ip = env_vars.get("VLLM_HOST_IP")
    if vllm_host_ip is not None:
        findings.append(f"VLLM_HOST_IP={vllm_host_ip}")

    # --- VLLM_PORT ---
    vllm_port = env_vars.get("VLLM_PORT")
    if vllm_port is not None:
        findings.append(f"VLLM_PORT={vllm_port}")

    # --- VLLM_WORKER_MULTIPROC_METHOD ---
    multiproc = env_vars.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc is not None:
        findings.append(f"VLLM_WORKER_MULTIPROC_METHOD={multiproc}")

    # --- VLLM_DISABLE_PYNCCL ---
    disable_pynccl = env_vars.get("VLLM_DISABLE_PYNCCL")
    if disable_pynccl == "1":
        warns.append(
            "VLLM_DISABLE_PYNCCL=1 — PyNCCL disabled, falling back to "
            "torch.distributed which may be slower for tensor parallel"
        )

    # --- VLLM_NCCL_SO_PATH ---
    nccl_so = env_vars.get("VLLM_NCCL_SO_PATH")
    if nccl_so is not None:
        findings.append(f"VLLM_NCCL_SO_PATH={nccl_so}")

    # --- VLLM_CPU_KVCACHE_SPACE ---
    kv_space = env_vars.get("VLLM_CPU_KVCACHE_SPACE")
    if kv_space is not None:
        findings.append(f"VLLM_CPU_KVCACHE_SPACE={kv_space} GiB (CPU memory for KV cache offload)")

    # --- VLLM_ENGINE_ITERATION_TIMEOUT_S ---
    engine_timeout = env_vars.get("VLLM_ENGINE_ITERATION_TIMEOUT_S")
    if engine_timeout is not None:
        findings.append(f"VLLM_ENGINE_ITERATION_TIMEOUT_S={engine_timeout}")
        try:
            if int(engine_timeout) < 30:
                warns.append(
                    f"VLLM_ENGINE_ITERATION_TIMEOUT_S={engine_timeout} — "
                    f"low timeout may cause spurious failures under load"
                )
        except ValueError:
            pass

    # --- VLLM_RPC_TIMEOUT ---
    rpc_timeout = env_vars.get("VLLM_RPC_TIMEOUT")
    if rpc_timeout is not None:
        findings.append(f"VLLM_RPC_TIMEOUT={rpc_timeout}")

    # --- VLLM_USE_NCCL_SYMM_MEM ---
    symm_mem = env_vars.get("VLLM_USE_NCCL_SYMM_MEM")
    if symm_mem is not None:
        findings.append(f"VLLM_USE_NCCL_SYMM_MEM={symm_mem}")

    # --- vLLM data parallelism ---
    vllm_dp_size = env_vars.get("VLLM_DP_SIZE")
    if vllm_dp_size is not None:
        findings.append(f"VLLM_DP_SIZE={vllm_dp_size}")
    vllm_dp_rank = env_vars.get("VLLM_DP_RANK")
    if vllm_dp_rank is not None:
        findings.append(f"VLLM_DP_RANK={vllm_dp_rank}")
    vllm_dp_master = env_vars.get("VLLM_DP_MASTER_IP")
    if vllm_dp_master is not None:
        findings.append(f"VLLM_DP_MASTER_IP={vllm_dp_master}")
        if vllm_dp_master in ("127.0.0.1", "localhost") and vllm_dp_size and vllm_dp_size != "1":
            warns.append(
                f"VLLM_DP_MASTER_IP={vllm_dp_master} but VLLM_DP_SIZE={vllm_dp_size} — "
                f"multi-node data parallel requires a routable IP, not localhost"
            )

    # --- VLLM_FLOAT32_MATMUL_PRECISION ---
    matmul_prec = env_vars.get("VLLM_FLOAT32_MATMUL_PRECISION")
    if matmul_prec is not None:
        findings.append(f"VLLM_FLOAT32_MATMUL_PRECISION={matmul_prec}")
        if matmul_prec == "highest":
            findings.append(
                "  -> 'highest' uses FP32; 'medium' uses TF32 which is "
                "faster and sufficient for inference"
            )

    # --- VLLM_TARGET_DEVICE ---
    target_dev = env_vars.get("VLLM_TARGET_DEVICE")
    if target_dev is not None:
        findings.append(f"VLLM_TARGET_DEVICE={target_dev}")

    # --- vLLM ROCm-specific ---
    rocm_aiter = env_vars.get("VLLM_ROCM_USE_AITER")
    if rocm_aiter is not None:
        findings.append(f"VLLM_ROCM_USE_AITER={rocm_aiter}")
    rocm_quick = env_vars.get("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION")
    if rocm_quick is not None:
        findings.append(f"VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={rocm_quick}")

    # Dump remaining vLLM vars not specifically handled above
    _handled_vllm = {
        "VLLM_HOST_IP", "VLLM_PORT", "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_DISABLE_PYNCCL", "VLLM_NCCL_SO_PATH", "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S", "VLLM_RPC_TIMEOUT",
        "VLLM_USE_NCCL_SYMM_MEM", "VLLM_DP_SIZE", "VLLM_DP_RANK",
        "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT",
        "VLLM_FLOAT32_MATMUL_PRECISION", "VLLM_TARGET_DEVICE",
        "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_PAGED_ATTN",
        "VLLM_ROCM_USE_AITER_MOE", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
    }
    other_vllm = {k: v for k, v in vllm_vars.items() if k not in _handled_vllm}
    if other_vllm:
        for k, v in sorted(other_vllm.items()):
            findings.append(f"{k}={v}")

    # =====================================================================
    # NCCL performance-tuning variables (beyond basic validation)
    # =====================================================================

    # --- NCCL_P2P_DISABLE ---
    p2p_disable = env_vars.get("NCCL_P2P_DISABLE")
    if p2p_disable == "1":
        if gpu_count and gpu_count > 1:
            warns.append(
                "NCCL_P2P_DISABLE=1 — GPU-to-GPU direct transfer disabled; "
                "severe performance hit for tensor parallel inference with multiple GPUs"
            )
        else:
            findings.append("NCCL_P2P_DISABLE=1 (no impact with single GPU)")

    # --- NCCL_SHM_DISABLE ---
    shm_disable = env_vars.get("NCCL_SHM_DISABLE")
    if shm_disable == "1":
        if gpu_count and gpu_count > 1:
            warns.append(
                "NCCL_SHM_DISABLE=1 — shared memory transport disabled; "
                "reduces intra-node communication performance"
            )

    # --- NCCL_ALGO / NCCL_PROTO ---
    nccl_algo = env_vars.get("NCCL_ALGO")
    if nccl_algo is not None:
        findings.append(f"NCCL_ALGO={nccl_algo} (forced collective algorithm)")
    nccl_proto = env_vars.get("NCCL_PROTO")
    if nccl_proto is not None:
        findings.append(f"NCCL_PROTO={nccl_proto} (forced protocol)")

    # --- NCCL_MAX_CTAS / NCCL_NTHREADS ---
    max_ctas = env_vars.get("NCCL_MAX_CTAS")
    if max_ctas is not None:
        findings.append(
            f"NCCL_MAX_CTAS={max_ctas} (limits NCCL GPU thread blocks — "
            f"lower values reduce GPU contention for inference)"
        )
    nccl_nthreads = env_vars.get("NCCL_NTHREADS")
    if nccl_nthreads is not None:
        findings.append(f"NCCL_NTHREADS={nccl_nthreads} (CUDA threads per NCCL kernel block)")

    # --- NCCL_NVLS_ENABLE ---
    nvls_enable = env_vars.get("NCCL_NVLS_ENABLE")
    if nvls_enable is not None:
        findings.append(f"NCCL_NVLS_ENABLE={nvls_enable}")
        if nvls_enable == "0" and has_nvlink:
            warns.append(
                "NCCL_NVLS_ENABLE=0 but NVLink/NVSwitch present — "
                "NVLink SHARP is disabled, may reduce allreduce performance"
            )

    # --- NCCL_CROSS_NIC ---
    cross_nic = env_vars.get("NCCL_CROSS_NIC")
    if cross_nic is not None:
        findings.append(f"NCCL_CROSS_NIC={cross_nic} (0=same NIC per ring, 1=allow cross-NIC, 2=auto)")

    # --- NCCL_COLLNET_ENABLE ---
    collnet = env_vars.get("NCCL_COLLNET_ENABLE")
    if collnet is not None:
        findings.append(f"NCCL_COLLNET_ENABLE={collnet} (InfiniBand SHARP in-network reduction)")

    # --- NCCL_TOPO_FILE ---
    topo_file = env_vars.get("NCCL_TOPO_FILE")
    if topo_file is not None:
        findings.append(f"NCCL_TOPO_FILE={topo_file} (custom topology XML)")

    # --- NCCL_DEBUG ---
    nccl_debug = env_vars.get("NCCL_DEBUG")
    if nccl_debug is not None:
        findings.append(f"NCCL_DEBUG={nccl_debug}")
        if nccl_debug.upper() in ("INFO", "TRACE"):
            warns.append(
                f"NCCL_DEBUG={nccl_debug} — verbose logging adds overhead; "
                f"use WARN for production inference"
            )

    # --- NCCL_BUFFSIZE ---
    buffsize = env_vars.get("NCCL_BUFFSIZE")
    if buffsize is not None:
        findings.append(f"NCCL_BUFFSIZE={buffsize} (internal NCCL buffer size)")

    # --- NCCL_GRAPH_REGISTER ---
    graph_reg = env_vars.get("NCCL_GRAPH_REGISTER")
    if graph_reg == "0":
        warns.append(
            "NCCL_GRAPH_REGISTER=0 — buffer registration for CUDA Graphs disabled; "
            "should be 1 for best inference performance with CUDA graphs"
        )

    # --- NCCL_LAUNCH_MODE ---
    launch_mode = env_vars.get("NCCL_LAUNCH_MODE")
    if launch_mode is not None:
        findings.append(f"NCCL_LAUNCH_MODE={launch_mode}")

    # --- NCCL_P2P_LEVEL ---
    p2p_level = env_vars.get("NCCL_P2P_LEVEL")
    if p2p_level is not None:
        findings.append(f"NCCL_P2P_LEVEL={p2p_level}")

    # --- NCCL_SOCKET_NTHREADS / NCCL_NSOCKS_PERTHREAD ---
    sock_nthreads = env_vars.get("NCCL_SOCKET_NTHREADS")
    if sock_nthreads is not None:
        findings.append(f"NCCL_SOCKET_NTHREADS={sock_nthreads}")
    nsocks = env_vars.get("NCCL_NSOCKS_PERTHREAD")
    if nsocks is not None:
        findings.append(f"NCCL_NSOCKS_PERTHREAD={nsocks}")

    # =====================================================================
    # PyTorch distributed environment variables
    # =====================================================================
    master_addr = env_vars.get("MASTER_ADDR")
    master_port = env_vars.get("MASTER_PORT")
    world_size = env_vars.get("WORLD_SIZE")
    rank = env_vars.get("RANK")
    local_rank = env_vars.get("LOCAL_RANK")

    torch_dist_set = any(v is not None for v in [master_addr, master_port, world_size, rank])
    if torch_dist_set:
        findings.append("PyTorch distributed:")
        if master_addr:
            findings.append(f"  MASTER_ADDR={master_addr}")
        if master_port:
            findings.append(f"  MASTER_PORT={master_port}")
        if world_size:
            findings.append(f"  WORLD_SIZE={world_size}")
        if rank is not None:
            findings.append(f"  RANK={rank}")
        if local_rank is not None:
            findings.append(f"  LOCAL_RANK={local_rank}")

    # --- TORCH_DISTRIBUTED_DEBUG ---
    torch_debug = env_vars.get("TORCH_DISTRIBUTED_DEBUG")
    if torch_debug is not None:
        findings.append(f"TORCH_DISTRIBUTED_DEBUG={torch_debug}")
        if torch_debug.upper() == "DETAIL":
            warns.append(
                "TORCH_DISTRIBUTED_DEBUG=DETAIL — adds significant overhead; "
                "use OFF or INFO for production inference"
            )

    # --- TORCH_NCCL_BLOCKING_WAIT ---
    nccl_blocking = env_vars.get("TORCH_NCCL_BLOCKING_WAIT")
    if nccl_blocking == "1":
        warns.append(
            "TORCH_NCCL_BLOCKING_WAIT=1 — blocks on NCCL operations, "
            "can cause hangs; prefer async error handling"
        )

    # =====================================================================
    # NIXL / llm-d KV cache transfer variables
    # =====================================================================
    nixl_vars = {k: v for k, v in env_vars.items() if k.startswith("NIXL_")}
    if nixl_vars:
        findings.append(f"NIXL (KV transfer): {len(nixl_vars)} variable(s) set")

    nixl_etcd = env_vars.get("NIXL_ETCD_ENDPOINTS")
    if nixl_etcd is not None:
        findings.append(f"NIXL_ETCD_ENDPOINTS={nixl_etcd}")

    nixl_ns = env_vars.get("NIXL_ETCD_NAMESPACE")
    if nixl_ns is not None:
        findings.append(f"NIXL_ETCD_NAMESPACE={nixl_ns}")

    # Dump remaining NIXL vars
    _handled_nixl = {"NIXL_ETCD_ENDPOINTS", "NIXL_ETCD_NAMESPACE"}
    other_nixl = {k: v for k, v in nixl_vars.items() if k not in _handled_nixl}
    for k, v in sorted(other_nixl.items()):
        findings.append(f"{k}={v}")

    # =====================================================================
    # Cross-variable consistency checks
    # =====================================================================

    # Check: NCCL_IB_DISABLE=1 with RDMA devices available + multi-GPU
    ib_disable = env_vars.get("NCCL_IB_DISABLE")
    if ib_disable == "1" and rdma_devices and gpu_count and gpu_count > 1:
        warns.append(
            f"NCCL_IB_DISABLE=1 with {len(rdma_devices)} RDMA device(s) and "
            f"{gpu_count} GPUs — inter-GPU RDMA communication disabled, "
            f"falling back to slower socket transport"
        )

    # Check: NIXL without RDMA for disaggregated KV transfer
    if nixl_vars and not rdma_devices:
        warns.append(
            "NIXL env vars set (disaggregated KV transfer) but no RDMA devices found — "
            "KV cache transfer will use TCP, which is significantly slower"
        )

    # Check: UCX_IB_GID_INDEX vs NCCL_IB_GID_INDEX mismatch
    ucx_gid = env_vars.get("UCX_IB_GID_INDEX")
    nccl_gid = env_vars.get("NCCL_IB_GID_INDEX")
    if ucx_gid is not None and nccl_gid is not None and ucx_gid != nccl_gid:
        warns.append(
            f"UCX_IB_GID_INDEX={ucx_gid} vs NCCL_IB_GID_INDEX={nccl_gid} — "
            f"GID index mismatch between UCX and NCCL may cause connection failures"
        )

    return findings, warns


def _check_gpu_numa_affinity(pod_name, gpu_labels, connections, vendor="nvidia"):
    """Check whether GPUs in a pod are on the same NUMA node / PCIe root.

    In dual-socket servers each CPU socket owns its own PCIe root complex.
    If Kubernetes schedules a pod with GPUs on opposite sockets, the only
    data path between them is SYS-level PCIe hops through the CPU
    interconnect (QPI/UPI), adding latency and contention.

    Returns (gpu_numa_map, warnings) where gpu_numa_map is
    {gpu_label: numa_node} and warnings is a list of strings.
    """
    if not gpu_labels or len(gpu_labels) < 2:
        return {}, []

    # Get GPU bus IDs using vendor-appropriate tool, then query NUMA via sysfs
    bus_pairs = _get_gpu_bus_ids(pod_name, vendor)
    gpu_numa = {}
    if bus_pairs:
        # Batch NUMA queries in one exec
        reads = []
        for idx, bus_id in bus_pairs:
            reads.append(f'echo "GPU{idx}:$(cat /sys/bus/pci/devices/{bus_id}/numa_node 2>/dev/null || echo ?)"')
        script = "; ".join(reads) + "; true"
        result = exec_in_pod(pod_name, ["bash", "-c", script], timeout=20, use_debug=False)
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    gpu_label, numa_str = line.split(":", 1)
                    gpu_label = gpu_label.strip()
                    try:
                        gpu_numa[gpu_label] = int(numa_str.strip())
                    except ValueError:
                        gpu_numa[gpu_label] = -1

    warns = []
    if not gpu_numa:
        return {}, []

    # Check if GPUs span multiple NUMA nodes
    numa_nodes_used = set(gpu_numa.values()) - {-1}
    if len(numa_nodes_used) > 1:
        # Build per-NUMA grouping
        numa_groups = {}
        for gpu, node in sorted(gpu_numa.items()):
            numa_groups.setdefault(node, []).append(gpu)
        group_strs = []
        for node in sorted(numa_groups):
            gpus = ", ".join(numa_groups[node])
            group_strs.append(f"NUMA {node}: {gpus}")

        warns.append(
            f"GPUs span {len(numa_nodes_used)} NUMA nodes ({'; '.join(group_strs)}). "
            f"Cross-socket PCIe traffic will traverse the CPU interconnect (QPI/UPI), "
            f"adding latency and contention to GPU-GPU and GPU-NIC transfers"
        )

        # Check for SYS/PHB connections between cross-NUMA GPU pairs
        cross_numa_sys = []
        cross_numa_phb = []
        seen = set()
        for (g1, g2), conn in connections.items():
            if g1 == g2 or conn == "X":
                continue
            pair = tuple(sorted([g1, g2]))
            if pair in seen:
                continue
            seen.add(pair)
            n1 = gpu_numa.get(g1)
            n2 = gpu_numa.get(g2)
            if n1 is not None and n2 is not None and n1 != n2:
                if conn == "SYS":
                    cross_numa_sys.append(pair)
                elif conn == "PHB":
                    cross_numa_phb.append(pair)

        if cross_numa_sys:
            pair_strs = [f"{a}<->{b}" for a, b in cross_numa_sys]
            warns.append(
                f"SYS connection (cross-socket): {', '.join(pair_strs)}. "
                f"This is the worst PCIe path — data crosses the CPU "
                f"inter-socket link, impacting allreduce/allgather latency"
            )
        if cross_numa_phb:
            pair_strs = [f"{a}<->{b}" for a, b in cross_numa_phb]
            warns.append(
                f"PHB connection (cross PCIe root, same NUMA): {', '.join(pair_strs)}. "
                f"Data traverses the CPU internal interconnect between PCIe root complexes"
            )
    elif len(numa_nodes_used) == 1:
        # All on same NUMA — optimal
        pass
    # else: couldn't determine NUMA nodes

    return gpu_numa, warns


def _check_pcie_root_spread(pod_name, gpu_labels, connections, vendor="nvidia"):
    """Check whether GPUs are spread across multiple PCIe root complexes.

    On larger nodes (e.g. 8 GPUs), GPUs may be wired under different PCIe
    root ports.  When GPUs span root complexes without NVLink/xGMI, cross-root
    traffic must traverse the CPU's internal fabric, reducing bandwidth
    and increasing latency for P2P transfers.

    Queries each GPU's PCI topology depth to identify its root port,
    then groups GPUs by root port.

    Returns (gpu_pcie_map, warnings) where gpu_pcie_map is
    {gpu_label: {"bus_id": ..., "root_port": ...}}
    and warnings is a list of strings.
    """
    if not gpu_labels or len(gpu_labels) < 2:
        return {}, []

    # Get bus IDs using vendor-appropriate tool, then walk sysfs for root port
    bus_pairs = _get_gpu_bus_ids(pod_name, vendor)
    gpu_pcie = {}
    if bus_pairs:
        # For each GPU, walk sysfs PCI chain to find root port
        reads = []
        for idx, bus_id in bus_pairs:
            reads.append(
                f'dev_path=$(readlink -f "/sys/bus/pci/devices/{bus_id}" 2>/dev/null); '
                f'root_port="unknown"; '
                f'if [ -n "$dev_path" ]; then '
                f'  chain=$(echo "$dev_path" | grep -oE "[0-9a-f]{{4}}:[0-9a-f]{{2}}:[0-9a-f]{{2}}\\.[0-9a-f]" | head -2); '
                f'  root_port=$(echo "$chain" | tail -1); '
                f'fi; '
                f'echo "GPU{idx}:{bus_id}:$root_port"'
            )
        script = "; ".join(reads) + "; true"
        result = exec_in_pod(pod_name, ["bash", "-c", script], timeout=20, use_debug=False)
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Parse GPU<idx>:<bus_id>:<root_port>
                first_colon = line.index(":") if ":" in line else -1
                if first_colon < 0:
                    continue
                gpu_label = line[:first_colon]
                rest = line[first_colon + 1:]
                pci_addrs = re.findall(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]', rest)
                bus_id = pci_addrs[0] if len(pci_addrs) >= 1 else "unknown"
                root_port = pci_addrs[1] if len(pci_addrs) >= 2 else pci_addrs[0] if pci_addrs else "unknown"
                gpu_pcie[gpu_label] = {"bus_id": bus_id, "root_port": root_port}

    warns = []
    if not gpu_pcie or len(gpu_pcie) < 2:
        return gpu_pcie, []

    # Group GPUs by root port
    root_groups = {}
    for gpu, info in sorted(gpu_pcie.items()):
        rp = info["root_port"]
        root_groups.setdefault(rp, []).append(gpu)

    if len(root_groups) > 1:
        group_strs = []
        for rp in sorted(root_groups):
            gpus = ", ".join(root_groups[rp])
            group_strs.append(f"root {rp}: {gpus}")

        # Check if NVLink bridges the split
        has_nvlink_across = False
        for (g1, g2), conn in connections.items():
            if g1 == g2 or conn == "X":
                continue
            r1 = gpu_pcie.get(g1, {}).get("root_port")
            r2 = gpu_pcie.get(g2, {}).get("root_port")
            if r1 and r2 and r1 != r2 and (conn.startswith("NV") or conn in ("NVL", "XGMI")):
                has_nvlink_across = True
                break

        if has_nvlink_across:
            warns.append(
                f"GPUs span {len(root_groups)} PCIe root port(s) ({'; '.join(group_strs)}) "
                f"but NVLink/xGMI bridges the gap — P2P transfers bypass PCIe"
            )
        else:
            warns.append(
                f"GPUs span {len(root_groups)} PCIe root port(s) ({'; '.join(group_strs)}). "
                f"Without NVLink/xGMI, cross-root GPU pairs must route through the CPU's "
                f"internal interconnect, reducing P2P bandwidth"
            )

            # Identify which GPU pairs are affected
            cross_root_pairs = []
            seen = set()
            for (g1, g2), conn in connections.items():
                if g1 == g2 or conn == "X":
                    continue
                pair = tuple(sorted([g1, g2]))
                if pair in seen:
                    continue
                seen.add(pair)
                r1 = gpu_pcie.get(g1, {}).get("root_port")
                r2 = gpu_pcie.get(g2, {}).get("root_port")
                if r1 and r2 and r1 != r2:
                    cross_root_pairs.append((g1, g2, conn))
            if cross_root_pairs:
                pair_strs = [f"{a}<->{b} ({c})" for a, b, c in cross_root_pairs]
                if len(pair_strs) <= 6:
                    warns.append(f"Cross-root pairs: {', '.join(pair_strs)}")
                else:
                    warns.append(f"Cross-root pairs: {len(pair_strs)} GPU pair(s) affected")

    return gpu_pcie, warns


def _check_pcie_switch_topology(pod_name, gpu_labels, connections, gpu_pcie, vendor="nvidia"):
    """Check for GPUs separated by PCIe switches (PXB) sharing upstream links.

    Even when GPUs are on the same NUMA node and under the same PCIe root
    complex, they may sit behind different PCIe switches.  The PXB connection
    type in the topo matrix means GPU traffic must cross PCIe switches,
    sharing a bottlenecked upstream link to the root complex.

    This function:
    1. Identifies PXB and PIX pairs from the topo matrix
    2. Queries the PCIe switch (bridge) each GPU sits under
    3. Groups GPUs by their immediate parent switch
    4. Warns about cross-switch pairs and shared upstream bandwidth

    Returns (switch_groups, warnings) where switch_groups is
    {switch_bus_id: [gpu_labels]} and warnings is a list of strings.
    """
    if not gpu_labels or len(gpu_labels) < 2:
        return {}, []

    # Classify connections from the topo matrix
    pxb_pairs = []
    pix_pairs = []
    phb_sys_pairs = []
    seen = set()
    for (g1, g2), conn in connections.items():
        if g1 == g2 or conn == "X":
            continue
        pair = tuple(sorted([g1, g2]))
        if pair in seen:
            continue
        seen.add(pair)
        if conn == "PXB":
            pxb_pairs.append((g1, g2))
        elif conn == "PIX":
            pix_pairs.append((g1, g2))
        elif conn in ("PHB", "SYS", "NODE"):
            phb_sys_pairs.append((g1, g2, conn))

    # Query the PCIe switch (parent bridge) for each GPU using vendor-appropriate bus IDs
    bus_pairs = _get_gpu_bus_ids(pod_name, vendor)
    result_stdout = ""
    if bus_pairs:
        reads = []
        for idx, bus_id in bus_pairs:
            reads.append(
                f'dev_path=$(readlink -f "/sys/bus/pci/devices/{bus_id}" 2>/dev/null); '
                f'switch="unknown"; '
                f'if [ -n "$dev_path" ]; then '
                f'  parent=$(dirname "$dev_path"); '
                f'  while [ "$parent" != "/sys/devices" ] && [ -n "$parent" ]; do '
                f'    bname=$(basename "$parent"); '
                f'    if echo "$bname" | grep -qE "^[0-9a-f]{{4}}:[0-9a-f]{{2}}:[0-9a-f]{{2}}\\.[0-9a-f]$"; then '
                f'      class=$(cat "$parent/class" 2>/dev/null); '
                f'      if echo "$class" | grep -qE "^0x0604"; then '
                f'        switch="$bname"; break; '
                f'      fi; '
                f'    fi; '
                f'    parent=$(dirname "$parent"); '
                f'  done; '
                f'fi; '
                f'echo "GPU{idx}:$switch"'
            )
        script = "; ".join(reads) + "; true"
        result = exec_in_pod(pod_name, ["bash", "-c", script], timeout=20, use_debug=False)
        if result.returncode == 0:
            result_stdout = result.stdout

    gpu_switch = {}  # GPU0 -> switch_bus_id
    if result_stdout.strip():
        for line in result_stdout.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                gpu_label, switch_id = line.split(":", 1)
                gpu_switch[gpu_label.strip()] = switch_id.strip()

    # Group GPUs by parent switch
    switch_groups = {}
    for gpu, sw in sorted(gpu_switch.items()):
        switch_groups.setdefault(sw, []).append(gpu)

    warns = []

    # Report PXB connections with switch detail
    if pxb_pairs:
        pair_strs = [f"{a}<->{b}" for a, b in pxb_pairs]
        if len(pair_strs) <= 6:
            detail = ", ".join(pair_strs)
        else:
            detail = f"{len(pair_strs)} pair(s)"

        # Build switch group description
        known_switches = {sw: gpus for sw, gpus in switch_groups.items() if sw != "unknown"}
        if len(known_switches) > 1:
            sw_strs = [f"switch {sw}: {', '.join(gpus)}" for sw, gpus in sorted(known_switches.items())]
            warns.append(
                f"PXB: {detail} — GPUs under different PCIe switches within "
                f"the same root complex ({'; '.join(sw_strs)}). Cross-switch "
                f"traffic shares the upstream link to the root port, which can "
                f"bottleneck P2P transfers and reduce effective GPU-GPU bandwidth"
            )
        else:
            warns.append(
                f"PXB: {detail} — GPUs cross PCIe switch boundaries within "
                f"the same root complex, sharing an upstream link that can "
                f"bottleneck P2P transfers"
            )

    # Report if some pairs are PIX (good) and others PXB (bad) — mixed switch topology
    if pix_pairs and pxb_pairs:
        pix_strs = [f"{a}<->{b}" for a, b in pix_pairs]
        warns.append(
            f"Mixed PCIe switch topology: {len(pix_pairs)} pair(s) share a switch (PIX, fast), "
            f"{len(pxb_pairs)} pair(s) cross switches (PXB, slower upstream shared link)"
        )

    return switch_groups, warns


def _discover_pod_topology(pod_name, short, out):
    """Run all topology checks for a single pod.

    Writes human-readable output to *out* (a file-like object).
    Returns (pod_topo dict, list of warning strings).
    """
    p = out.write  # shorthand

    pod_topo = {}
    pod_warnings = []

    p(f"\n--- {short} ({pod_name}) ---\n")

    # Detect GPU vendor (nvidia vs amd)
    vendor = _detect_gpu_vendor(pod_name)
    pod_topo["gpu_vendor"] = vendor
    if vendor == "amd":
        amd_tool = _get_amd_tool(pod_name)
        p(f"  GPU vendor: AMD (using {amd_tool})\n")
    elif vendor == "nvidia":
        p(f"  GPU vendor: NVIDIA (using nvidia-smi)\n")
    else:
        p(f"  GPU vendor: not detected (no nvidia-smi, rocm-smi, or amd-smi found)\n")

    # 1. GPU topology matrix
    if vendor == "nvidia":
        topo_cmd = "nvidia-smi topo -m"
    elif vendor == "amd":
        topo_cmd = "amd-smi topology" if _get_amd_tool(pod_name) == "amd-smi" else "rocm-smi --showtopo"
    else:
        topo_cmd = "GPU topology"
    p(f"  Checking GPU topology ({topo_cmd}) ... ")
    topo_out, topo_ok = _get_gpu_topo(pod_name, vendor)
    if topo_ok:
        p("OK\n")
        if vendor == "amd":
            connections, gpu_labels = _parse_rocm_topo(topo_out)
        else:
            connections, gpu_labels = _parse_topo_matrix(topo_out)
        has_nvlink, topo_summary = _check_nvlink_topology(connections, gpu_labels)
        pod_topo["gpu_count"] = len(gpu_labels)
        pod_topo["has_nvlink"] = has_nvlink
        pod_topo["topo_summary"] = topo_summary
        pod_topo["connections"] = connections
        pod_topo["gpu_labels"] = gpu_labels

        # Use vendor-appropriate naming
        has_xgmi = any(c == "XGMI" for c in connections.values())
        if has_nvlink:
            status = "xGMI/Infinity Fabric" if has_xgmi else "NVLink/NVSwitch"
        else:
            status = "PCIe only"
        p(f"    GPUs: {len(gpu_labels)}, Interconnect: {status}\n")
        p(f"    Detail: {topo_summary}\n")

        if not has_nvlink and len(gpu_labels) > 1:
            link_name = "xGMI" if vendor == "amd" else "NVLink"
            pod_warnings.append(f"{short}: GPUs connected via PCIe only (no {link_name})")

        # Check for excessive hops between GPUs
        hop_warns = _check_hop_distance(connections, gpu_labels)
        if hop_warns:
            p(f"    Hop-distance issues:\n")
            for hw in hop_warns:
                p(f"      {hw}\n")
                pod_warnings.append(f"{short}: excessive hops — {hw}")
        else:
            if len(gpu_labels) > 1:
                p(f"    Hop distance: OK (all pairs minimal)\n")

        # Check NVSwitch domain uniformity
        nvsw_uniform, nvsw_warns = _check_nvswitch_domains(connections, gpu_labels)
        pod_topo["nvswitch_uniform"] = nvsw_uniform
        if nvsw_warns:
            p(f"    NVSwitch domain issues:\n")
            for nw in nvsw_warns:
                p(f"      {nw}\n")
                pod_warnings.append(f"{short}: {nw}")
        else:
            if has_nvlink and len(gpu_labels) > 1:
                p(f"    NVSwitch domain: OK (all GPUs in same domain)\n")

        # Check GPU NUMA affinity (cross-socket placement)
        gpu_numa, numa_warns = _check_gpu_numa_affinity(pod_name, gpu_labels, connections, vendor)
        pod_topo["gpu_numa"] = gpu_numa
        if gpu_numa:
            numa_nodes_used = set(gpu_numa.values()) - {-1}
            if len(numa_nodes_used) == 1:
                p(f"    NUMA affinity: OK (all GPUs on NUMA node {next(iter(numa_nodes_used))})\n")
            elif len(numa_nodes_used) > 1:
                p(f"    NUMA affinity: SPLIT across {len(numa_nodes_used)} NUMA nodes\n")
                for nw in numa_warns:
                    p(f"      {nw}\n")
                    pod_warnings.append(f"{short}: {nw}")
            else:
                p(f"    NUMA affinity: could not determine\n")
        elif len(gpu_labels) > 1:
            p(f"    NUMA affinity: could not query\n")

        # Check PCIe root complex spread
        gpu_pcie, pcie_root_warns = _check_pcie_root_spread(pod_name, gpu_labels, connections, vendor)
        pod_topo["gpu_pcie"] = gpu_pcie
        if gpu_pcie and len(gpu_labels) > 1:
            root_ports = set(info["root_port"] for info in gpu_pcie.values()
                             if info.get("root_port") != "unknown")
            if len(root_ports) == 1:
                p(f"    PCIe root:     OK (all GPUs under root port {next(iter(root_ports))})\n")
            elif len(root_ports) > 1:
                p(f"    PCIe root:     SPREAD across {len(root_ports)} root port(s)\n")
                for pw in pcie_root_warns:
                    p(f"      {pw}\n")
                    pod_warnings.append(f"{short}: {pw}")
            else:
                p(f"    PCIe root:     could not determine root ports\n")
        elif len(gpu_labels) > 1 and not gpu_pcie:
            p(f"    PCIe root:     could not query\n")

        # Check PCIe switch topology (PXB / shared upstream links)
        switch_groups, switch_warns = _check_pcie_switch_topology(
            pod_name, gpu_labels, connections, gpu_pcie, vendor)
        pod_topo["pcie_switch_groups"] = switch_groups
        if switch_warns:
            p(f"    PCIe switches: bandwidth concern\n")
            for sw in switch_warns:
                p(f"      {sw}\n")
                pod_warnings.append(f"{short}: {sw}")
        elif len(gpu_labels) > 1 and switch_groups:
            known = {k: v for k, v in switch_groups.items() if k != "unknown"}
            if len(known) == 1:
                p(f"    PCIe switches: OK (all GPUs behind same switch {next(iter(known))})\n")
            elif len(known) == 0:
                p(f"    PCIe switches: could not determine parent switches\n")
            else:
                # Multiple switches but no PXB connections — GPUs probably under different roots
                p(f"    PCIe switches: {len(known)} switch(es) — see PCIe root above\n")

        if _get_common().VERBOSE and topo_out:
            for line in topo_out.split("\n"):
                p(f"    | {line}\n")
    else:
        p("not available\n")
        pod_topo["has_nvlink"] = None
        pod_topo["topo_summary"] = "GPU topology tool not available"

    # 2. NVLink / xGMI link status
    if vendor == "nvidia":
        link_cmd = "nvidia-smi nvlink --status"
    elif vendor == "amd":
        link_cmd = "amd-smi topology --weight" if _get_amd_tool(pod_name) == "amd-smi" else "rocm-smi --showtopoweight"
    else:
        link_cmd = "GPU link status"
    p(f"  Checking link status ({link_cmd}) ... ")
    nvlink_out, nvlink_ok = _get_gpu_link_status(pod_name, vendor)
    if nvlink_ok:
        p("OK\n")
        if vendor == "amd":
            active, inactive = _parse_rocm_link_status(nvlink_out)
        else:
            active, inactive = _parse_nvlink_status(nvlink_out)
        pod_topo["nvlink_active"] = active
        pod_topo["nvlink_inactive"] = inactive
        p(f"    NVLink connections: {active} active, {inactive} inactive\n")
        if inactive > 0:
            pod_warnings.append(f"{short}: {inactive} inactive NVLink connection(s)")
        if _get_common().VERBOSE and nvlink_out:
            for line in nvlink_out.split("\n"):
                p(f"    | {line}\n")
    else:
        p("not available\n")
        pod_topo["nvlink_active"] = 0
        pod_topo["nvlink_inactive"] = 0

    # 3. lscpu (CPU/NUMA topology)
    p("  Checking CPU topology (lscpu) ... ")
    lscpu_out, lscpu_ok = _run_topo_cmd(pod_name, ["lscpu"], "lscpu")
    lscpu_info = {}
    if lscpu_ok:
        p("OK\n")
        for line in lscpu_out.split("\n"):
            lower = line.lower()
            if any(k in lower for k in ["socket(s)", "core(s) per socket",
                                         "thread(s) per core", "numa node(s)",
                                         "model name", "cpu(s):"]):
                p(f"    {line.strip()}\n")
                # Parse key=value
                if ":" in line:
                    key, _, val = line.partition(":")
                    lscpu_info[key.strip().lower()] = val.strip()
        if _get_common().VERBOSE:
            for line in lscpu_out.split("\n"):
                p(f"    | {line}\n")
    else:
        p("not available\n")
    pod_topo["lscpu"] = lscpu_info

    # 4. lspci -tv (PCI topology tree)
    p("  Checking PCI topology (lspci -tv) ... ")
    lspci_out, lspci_ok = _run_topo_cmd(pod_name, ["lspci", "-tv"], "lspci")
    pci_gpus = []
    pci_nics = []
    pci_nvswitch = 0
    if lspci_ok:
        p("OK\n")
        # Show NVIDIA/AMD GPU and Mellanox/ConnectX lines
        gpu_lines = []
        for line in lspci_out.split("\n"):
            ll = line.lower()
            if any(k in ll for k in ["nvidia", "mellanox", "connectx", "infiniband",
                                      "advanced micro devices", "amd/ati", "instinct", "radeon"]):
                gpu_lines.append(line.rstrip())
            # Classify PCI devices
            if "nvidia" in ll and "nvswitch" in ll:
                pci_nvswitch += 1
            elif "nvidia" in ll and ("h100" in ll or "a100" in ll or "h200" in ll
                                     or "b200" in ll or "gb200" in ll or "l40" in ll
                                     or "v100" in ll or "t4" in ll or "3d controller" in ll
                                     or "gh100" in ll or "gh200" in ll or "ga100" in ll
                                     or "gb100" in ll or "gb202" in ll):
                pci_gpus.append(line.strip())
            elif ("amd" in ll or "advanced micro" in ll) and ("instinct" in ll or "mi300" in ll
                                     or "mi250" in ll or "mi210" in ll or "mi200" in ll
                                     or "mi100" in ll or "display controller" in ll
                                     or "mi308" in ll):
                pci_gpus.append(line.strip())
            if any(k in ll for k in ["mellanox", "connectx", "infiniband"]):
                # Only count physical functions, not VFs
                if "virtual function" not in ll:
                    pci_nics.append(line.strip())
        if gpu_lines:
            p(f"    GPU/NIC PCI devices ({len(gpu_lines)}):\n")
            for line in gpu_lines:
                p(f"      {line}\n")
        else:
            p("    No NVIDIA/Mellanox PCI devices found in tree\n")
        if _get_common().VERBOSE:
            for line in lspci_out.split("\n"):
                p(f"    | {line}\n")
    else:
        p("not available\n")
    pod_topo["pci_gpu_count"] = len(pci_gpus)
    pod_topo["pci_nic_count"] = len(pci_nics)
    pod_topo["pci_nvswitch_count"] = pci_nvswitch

    # 5. NCCL environment variable validation
    p("  Checking NCCL environment variables ... ")
    nccl_env = _fetch_nccl_env(pod_name)
    rdma_devs = _fetch_rdma_devices(pod_name)
    net_ifaces = _fetch_net_interfaces(pod_name)
    pod_topo["nccl_env"] = nccl_env
    pod_topo["rdma_devices"] = rdma_devs
    if nccl_env:
        p(f"found {len(nccl_env)} var(s)\n")
    else:
        p("none set\n")
    nccl_findings, nccl_warns = _validate_nccl_env(nccl_env, rdma_devs, net_ifaces)
    for f in nccl_findings:
        p(f"    {f}\n")
    for nw in nccl_warns:
        p(f"    WARNING: {nw}\n")
        pod_warnings.append(f"{short}: {nw}")

    # 5b. vLLM / CUDA / PyTorch / NIXL environment variable validation
    vllm_cuda_vars = {k: v for k, v in nccl_env.items()
                      if k.startswith(("VLLM_", "CUDA_", "TORCH_", "NIXL_", "NCCL_"))
                      or k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
    if vllm_cuda_vars:
        infer_vars = {k: v for k, v in nccl_env.items()
                      if k.startswith(("VLLM_", "CUDA_", "TORCH_", "NIXL_"))
                      or k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
        if infer_vars:
            p(f"  Checking vLLM/CUDA/PyTorch/NIXL environment ... found {len(infer_vars)} var(s)\n")
        else:
            p(f"  Checking vLLM/CUDA/PyTorch/NIXL environment ... none set\n")
        gpu_count_val = pod_topo.get("gpu_count", 0)
        has_nvlink_val = pod_topo.get("has_nvlink", False)
        vcn_findings, vcn_warns = _validate_vllm_cuda_env(
            nccl_env, gpu_count_val, rdma_devs, has_nvlink_val)
        for f in vcn_findings:
            p(f"    {f}\n")
        for vw in vcn_warns:
            p(f"    WARNING: {vw}\n")
            pod_warnings.append(f"{short}: {vw}")
    else:
        p("  Checking vLLM/CUDA/PyTorch/NIXL environment ... none set\n")

    # 6. RDMA NIC status (ibstat or sysfs fallback)
    if rdma_devs:
        p("  Checking RDMA NIC status ... ")
        nic_status = _fetch_rdma_nic_status(pod_name, rdma_devs)
        pod_topo["nic_status"] = nic_status
        if nic_status:
            source = next(iter(nic_status.values())).get("source", "unknown")
            p(f"OK (via {source}, {len(nic_status)} device(s))\n")

            # Summarize: group by state
            state_counts = {}
            down_devs = []
            for dev, info in sorted(nic_status.items()):
                state = info.get("state", "?")
                state_counts[state] = state_counts.get(state, 0) + 1
                if "active" not in state.lower() and state != "?":
                    down_devs.append(dev)

            state_str = ", ".join(f"{s}: {c}" for s, c in sorted(state_counts.items()))
            p(f"    Port states: {state_str}\n")

            if down_devs:
                if len(down_devs) <= 8:
                    p(f"    WARNING: non-active devices: {', '.join(down_devs)}\n")
                else:
                    p(f"    WARNING: {len(down_devs)} device(s) not in ACTIVE state\n")
                pod_warnings.append(
                    f"{short}: {len(down_devs)} RDMA device(s) not in ACTIVE state: "
                    f"{', '.join(down_devs[:8])}{'...' if len(down_devs) > 8 else ''}"
                )

            # Show rate for first active device
            for dev, info in sorted(nic_status.items()):
                if "active" in info.get("state", "").lower():
                    rate = info.get("rate", "?")
                    link = info.get("link_layer", "?")
                    fw = info.get("fw_ver", "?")
                    p(f"    Sample ({dev}): rate={rate}, link_layer={link}, fw={fw}\n")
                    break

            if _get_common().VERBOSE:
                for dev, info in sorted(nic_status.items()):
                    parts = [f"{k}={v}" for k, v in sorted(info.items()) if k != "source"]
                    p(f"    | {dev}: {', '.join(parts)}\n")
        else:
            p("could not query\n")
            pod_topo["nic_status"] = {}
    else:
        pod_topo["nic_status"] = {}

    return pod_topo, pod_warnings


def run_topology_validation(pods, display_names):
    """Run GPU topology checks on all pods in parallel.

    Each pod runs in its own thread.  Output is printed in pod order:
    the first pod's output streams live, then the second pod's buffered
    output is flushed (possibly already complete), and so on.
    """
    print(f"\n{'=' * 60}")
    print("  GPU TOPOLOGY VALIDATION")
    print(f"{'=' * 60}")

    n = len(pods)
    _StreamingBuffer = _get_common()._StreamingBuffer
    buffers = [_StreamingBuffer() for _ in range(n)]
    results = [None] * n      # (pod_topo, pod_warnings)
    done_events = [threading.Event() for _ in range(n)]

    def _worker(idx, pod_name, short, buf, done_evt):
        try:
            topo, warns = _discover_pod_topology(pod_name, short, buf)
            results[idx] = (topo, warns)
        except Exception as exc:
            buf.write(f"\n  ERROR discovering {short}: {exc}\n")
            results[idx] = ({}, [f"{short}: discovery failed — {exc}"])
        finally:
            done_evt.set()

    # Launch all threads
    threads = []
    for i, (pod_name, _ip) in enumerate(pods):
        short = display_names[pod_name]
        t = threading.Thread(
            target=_worker,
            args=(i, pod_name, short, buffers[i], done_events[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Print output in pod order: stream pod i, then move to pod i+1
    all_pod_results = {}
    warnings = []
    for i, (pod_name, _ip) in enumerate(pods):
        # Stream live output for this pod while its thread is running
        while not done_events[i].is_set():
            buffers[i].flush_new()
            done_events[i].wait(timeout=0.1)
        # Final flush — thread is done, print any remaining output
        buffers[i].flush_all()

        if results[i] is not None:
            pod_topo, pod_warns = results[i]
            all_pod_results[pod_name] = pod_topo
            warnings.extend(pod_warns)
        else:
            all_pod_results[pod_name] = {}

    # Wait for all threads to finish (should already be done)
    for t in threads:
        t.join(timeout=5)

    # Cross-pod consistency checks
    _nccl_key_vars = ["NCCL_IB_HCA", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
                       "NCCL_EXCLUDE_IB_HCA", "NCCL_NET", "NCCL_IB_GID_INDEX",
                       "NCCL_NET_GDR_LEVEL", "NCCL_IB_TC", "NCCL_IB_TIMEOUT",
                       "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_ALGO",
                       "NCCL_PROTO", "NCCL_MAX_CTAS", "NCCL_NVLS_ENABLE",
                       "NCCL_CROSS_NIC", "NCCL_DEBUG", "NCCL_TOPO_FILE",
                       "UCX_TLS", "UCX_NET_DEVICES", "UCX_MAX_RNDV_RAILS",
                       "UCX_RNDV_THRESH", "UCX_MEMTYPE_CACHE", "UCX_IB_GID_INDEX",
                       "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                       "CUDA_MODULE_LOADING",
                       "VLLM_HOST_IP", "VLLM_DISABLE_PYNCCL",
                       "VLLM_DP_SIZE", "VLLM_TARGET_DEVICE",
                       "VLLM_WORKER_MULTIPROC_METHOD",
                       "NIXL_ETCD_ENDPOINTS", "NIXL_ETCD_NAMESPACE"]

    env_mismatches = {}  # var -> {value -> [pod_short_names]}
    rdma_vals = {}       # tuple(device_list) -> [pod_short_names]
    if len(pods) > 1:
        for var in _nccl_key_vars:
            vals = {}
            for pod_name, _ip in pods:
                v = all_pod_results[pod_name].get("nccl_env", {}).get(var)
                vals.setdefault(v, []).append(display_names[pod_name])
            if len(vals) > 1:
                env_mismatches[var] = vals
                parts = []
                for v, pod_list in sorted(vals.items(), key=lambda x: x[0] or ""):
                    display_val = v if v is not None else "<unset>"
                    parts.append(f"{display_val} on {', '.join(pod_list)}")
                warnings.append(f"MISMATCH: {var} differs across pods: {'; '.join(parts)}")

        # Check RDMA device list consistency
        rdma_vals = {}
        for pod_name, _ip in pods:
            devs = tuple(all_pod_results[pod_name].get("rdma_devices", []))
            rdma_vals.setdefault(devs, []).append(display_names[pod_name])
        if len(rdma_vals) > 1:
            parts = []
            for devs, pod_list in rdma_vals.items():
                dev_str = ", ".join(devs) if devs else "<none>"
                parts.append(f"{len(devs)} device(s) on {', '.join(pod_list)}")
            warnings.append(f"MISMATCH: RDMA device lists differ across pods: {'; '.join(parts)}")

    # Topology connection type legend
    print(f"\n{'─' * 50}")
    print("  GPU Topology Connection Type Legend")
    print(f"{'─' * 50}")
    print("  High-bandwidth GPU interconnects:")
    print("    NV#   = NVIDIA NVLink with # lanes (e.g. NV12 = 12 lanes, NV18 = 18).")
    print("            Routed through NVSwitch on multi-GPU systems.")
    print("    NVL   = NVIDIA NVLink (alternate notation in some driver versions)")
    print("    XGMI  = AMD Infinity Fabric / xGMI link (direct GPU-GPU interconnect)")
    print()
    print("  PCIe paths (lower bandwidth, higher latency):")
    print("    PIX   = same PCIe switch (1 PCIe hop, best PCIe path)")
    print("    PXB   = cross PCIe switch, same PCIe root complex (2 PCIe hops)")
    print("    PHB   = same CPU/NUMA node but different PCIe root complex (traverses")
    print("            the CPU's internal interconnect)")
    print("    NODE  = same NUMA node (legacy alias, similar to PHB)")
    print("    SYS   = cross-socket / cross-NUMA (worst path — traverses QPI/UPI")
    print("            inter-CPU link, highest latency)")
    print()
    print("  Readiness:")
    print("    NVLink/xGMI-READY = all GPU pairs connected via NV#/NVL/XGMI — optimal")
    print("                        for collective operations (allreduce, allgather)")
    print("    PCIe only         = no high-bandwidth GPU interconnect — communication")
    print("                        goes through PCIe, significantly lower bandwidth")
    print()
    print("  Commands: nvidia-smi topo -m (NVIDIA) | rocm-smi --showtopo / amd-smi topology (AMD)")

    # Cross-pod comparison
    print(f"\n{'─' * 50}")
    print("  Topology Summary")
    print(f"{'─' * 50}")

    for pod_name, _ip in pods:
        short = display_names[pod_name]
        topo = all_pod_results[pod_name]
        gpu_count = topo.get("gpu_count", "?")
        has_nvlink = topo.get("has_nvlink")
        if has_nvlink is True:
            link_str = "NVLink/NVSwitch"
        elif has_nvlink is False:
            link_str = "PCIe only"
        else:
            link_str = "unknown"
        nvl_active = topo.get("nvlink_active", 0)
        nvl_inactive = topo.get("nvlink_inactive", 0)
        nvsw_ok = topo.get("nvswitch_uniform")
        nvsw_str = ""
        if nvsw_ok is True:
            nvsw_str = "  NVSwitch=uniform"
        elif nvsw_ok is False:
            nvsw_str = "  NVSwitch=SPLIT"
        gpu_numa = topo.get("gpu_numa", {})
        numa_nodes_used = set(gpu_numa.values()) - {-1} if gpu_numa else set()
        numa_str = ""
        if len(numa_nodes_used) == 1:
            numa_str = f"  NUMA={next(iter(numa_nodes_used))}"
        elif len(numa_nodes_used) > 1:
            numa_str = f"  NUMA=SPLIT({','.join(str(n) for n in sorted(numa_nodes_used))})"
        gpu_pcie = topo.get("gpu_pcie", {})
        root_ports = set(info["root_port"] for info in gpu_pcie.values()
                         if info.get("root_port") != "unknown") if gpu_pcie else set()
        pcie_root_str = ""
        if len(root_ports) == 1:
            pcie_root_str = f"  PCIe=1root"
        elif len(root_ports) > 1:
            pcie_root_str = f"  PCIe={len(root_ports)}roots"
        print(f"  {short:{_get_common().DISPLAY_NAME_MAX_LEN}s}  GPUs={gpu_count}  "
              f"Interconnect={link_str}  NVLinks={nvl_active} active/{nvl_inactive} inactive"
              f"{nvsw_str}{numa_str}{pcie_root_str}")

    # Check consistency across pods
    nvlink_states = set()
    gpu_counts = set()
    nvswitch_states = set()
    for topo in all_pod_results.values():
        if topo.get("has_nvlink") is not None:
            nvlink_states.add(topo["has_nvlink"])
        if "gpu_count" in topo:
            gpu_counts.add(topo["gpu_count"])
        if topo.get("nvswitch_uniform") is not None:
            nvswitch_states.add(topo["nvswitch_uniform"])

    if len(nvlink_states) > 1:
        warnings.append("MISMATCH: pods have different GPU interconnect types (some NVLink, some PCIe)")
    if len(gpu_counts) > 1:
        warnings.append(f"MISMATCH: pods have different GPU counts: {sorted(gpu_counts)}")
    if nvswitch_states == {True, False}:
        warnings.append("MISMATCH: some pods have uniform NVSwitch domains, others have split domains")

    if warnings:
        print(f"\n  Topology Warnings:")
        for w in warnings:
            print(f"    WARNING: {w}")
    else:
        print(f"\n  No topology warnings.")

    # Detailed environment consistency diff
    has_any_inconsistency = (env_mismatches or len(gpu_counts) > 1
                             or len(nvlink_states) > 1 or len(rdma_vals) > 1)
    if len(pods) > 1:
        print(f"\n{'─' * 50}")
        print("  Cross-Pod Consistency Check")
        print(f"{'─' * 50}")

        pod_shorts = [display_names[p] for p, _ in pods]
        name_w = max(len(s) for s in pod_shorts)

        if has_any_inconsistency:
            # GPU count consistency
            if len(gpu_counts) > 1:
                print(f"\n  GPU count (INCONSISTENT):")
                for pod_name, _ip in pods:
                    short = display_names[pod_name]
                    gc = all_pod_results[pod_name].get("gpu_count", "?")
                    print(f"    {short:{name_w}s}  {gc}")

            # Interconnect type consistency
            if len(nvlink_states) > 1:
                print(f"\n  GPU interconnect (INCONSISTENT):")
                for pod_name, _ip in pods:
                    short = display_names[pod_name]
                    hnv = all_pod_results[pod_name].get("has_nvlink")
                    label = "NVLink" if hnv is True else "PCIe only" if hnv is False else "unknown"
                    print(f"    {short:{name_w}s}  {label}")

            # RDMA device list consistency
            if len(rdma_vals) > 1:
                print(f"\n  RDMA devices (INCONSISTENT):")
                for pod_name, _ip in pods:
                    short = display_names[pod_name]
                    devs = all_pod_results[pod_name].get("rdma_devices", [])
                    print(f"    {short:{name_w}s}  {len(devs)} device(s): {', '.join(devs[:5])}"
                          f"{'...' if len(devs) > 5 else ''}")

            # NCCL env var diff table
            if env_mismatches:
                print(f"\n  Environment variables (INCONSISTENT):")
                for var, val_map in env_mismatches.items():
                    print(f"\n    {var}:")
                    for pod_name, _ip in pods:
                        short = display_names[pod_name]
                        v = all_pod_results[pod_name].get("nccl_env", {}).get(var)
                        display_val = v if v is not None else "<unset>"
                        # Truncate long values for readability
                        if len(display_val) > 60:
                            display_val = display_val[:57] + "..."
                        print(f"      {short:{name_w}s}  {display_val}")
        else:
            # All consistent — print summary
            checks_ok = []
            if len(gpu_counts) <= 1:
                checks_ok.append("GPU count")
            if len(nvlink_states) <= 1:
                checks_ok.append("GPU interconnect")
            if len(rdma_vals) <= 1:
                checks_ok.append("RDMA devices")
            checks_ok.append("env vars (NCCL/UCX/CUDA/vLLM/NIXL)")
            print(f"  All {len(pods)} pods are consistent: {', '.join(checks_ok)}")

    # Pod hardware profile summary
    print(f"\n{'─' * 50}")
    print("  Pod Hardware Profile")
    print(f"{'─' * 50}")

    # Classify pods by role (from pod name patterns)
    pod_roles = {}  # role -> [pod_short_names]
    for pod_name, _ip in pods:
        short = display_names[pod_name]
        name_lower = pod_name.lower()
        if "prefill" in name_lower:
            pod_roles.setdefault("prefill", []).append(short)
        elif "decode" in name_lower:
            pod_roles.setdefault("decode", []).append(short)
        else:
            pod_roles.setdefault("other", []).append(short)

    if pod_roles:
        role_parts = []
        for role in ["prefill", "decode", "other"]:
            if role in pod_roles:
                role_parts.append(f"{len(pod_roles[role])} {role}")
        print(f"  Pod types:         {', '.join(role_parts)} ({len(pods)} total)")
        if not _get_common().SSH_MODE:
            ns_flag = f" -n {_get_common().OPT_NAMESPACE}" if _get_common().OPT_NAMESPACE else ""
            explain_cmd(f"kubectl get pods{ns_flag} -l {_get_common().OPT_LABEL} -o json | jq '.items[].metadata.name'",
                        "pod names contain role hints (prefill, decode, etc.)")

    # Use first pod as representative (all should be consistent by this point)
    sample = next(iter(all_pod_results.values())) if all_pod_results else {}
    sample_pod = pods[0][0] if pods else "<POD>"
    kx = _kubectl_exec_prefix(sample_pod)
    lscpu = sample.get("lscpu", {})

    # CPU/NUMA
    cpu_model = lscpu.get("model name", "unknown")
    sockets = lscpu.get("socket(s)", "?")
    cores_per = lscpu.get("core(s) per socket", "?")
    threads_per = lscpu.get("thread(s) per core", "?")
    numa_nodes = lscpu.get("numa node(s)", "?")
    total_cpus = lscpu.get("cpu(s)", "?")
    if cpu_model != "unknown":
        print(f"  CPU:               {cpu_model}")
    print(f"  NUMA:              {numa_nodes} node(s), {sockets} socket(s), "
          f"{cores_per} cores/socket, {threads_per} threads/core, {total_cpus} logical CPUs")
    explain_cmd(f"{kx} lscpu | grep -iE 'socket|core|thread|numa|model name|^cpu\\(s\\)'",
                "parse Socket(s), Core(s) per socket, NUMA node(s), Model name")

    # GPUs
    gpu_count = sample.get("gpu_count", 0)
    pci_gpu_total = sample.get("pci_gpu_count", 0)
    pci_nvswitch = sample.get("pci_nvswitch_count", 0)
    has_nvlink = sample.get("has_nvlink")

    # Try to get GPU model from lspci data
    gpu_model = "unknown"
    sample_vendor = sample.get("gpu_vendor", "unknown")
    for pod_name, _ip in pods:
        if sample_vendor == "amd":
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", "lspci 2>/dev/null | grep -iE 'amd|instinct|advanced micro' "
                 "| grep -iv bridge | head -1; true"],
                timeout=15, use_debug=False,
            )
        else:
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", "lspci 2>/dev/null | grep -i nvidia | grep -iv nvswitch | head -1; true"],
                timeout=15, use_debug=False,
            )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip()
            for marker in ["NVIDIA Corporation ", "Advanced Micro Devices, Inc. ",
                           "AMD/ATI "]:
                if marker in line:
                    gpu_model = line.split(marker, 1)[1].strip()
                    break
            if gpu_model != "unknown":
                break

    if gpu_model != "unknown":
        print(f"  GPU:               {gpu_count} assigned to pod, model: {gpu_model}")
    else:
        print(f"  GPU:               {gpu_count} assigned to pod")
    explain_cmd(f"{kx} nvidia-smi topo -m",
                "count GPU rows (GPU0, GPU1, ...) for assigned GPUs")
    explain_cmd(f"{kx} bash -c \"lspci | grep -i nvidia | grep -iv nvswitch\"",
                "count lines for total GPUs on host PCI bus; extract model from description")
    if pci_gpu_total > gpu_count:
        print(f"                     ({pci_gpu_total} GPUs visible in PCI bus on host)")
    if pci_nvswitch > 0:
        print(f"  NVSwitch:          {pci_nvswitch} ASIC(s) on host PCI bus")
        explain_cmd(f"{kx} bash -c \"lspci | grep -i nvswitch | wc -l\"",
                    "count NVSwitch ASICs visible in PCI bus")
    if has_nvlink is True:
        print(f"  GPU interconnect:  NVLink/NVSwitch (high-bandwidth GPU-GPU fabric)")
    elif has_nvlink is False:
        print(f"  GPU interconnect:  PCIe only (no NVLink between assigned GPUs)")
    else:
        print(f"  GPU interconnect:  unknown")
    explain_cmd(f"{kx} nvidia-smi topo -m",
                "if all GPU pairs show NV## (e.g. NV12, NV18) = NVLink; PIX/PXB/PHB/SYS = PCIe")

    # RDMA / InfiniBand / RoCE
    rdma_devs = sample.get("rdma_devices", [])
    nccl_env = sample.get("nccl_env", {})
    exclude_hca = nccl_env.get("NCCL_EXCLUDE_IB_HCA", "")
    ib_hca = nccl_env.get("NCCL_IB_HCA", "")
    ib_disabled = nccl_env.get("NCCL_IB_DISABLE") == "1"

    pci_nic_count = sample.get("pci_nic_count", 0)
    if rdma_devs:
        # Determine link layer type by checking a device
        link_type = "unknown"
        for pod_name, _ip in pods:
            result = exec_in_pod(
                pod_name,
                ["bash", "-c",
                 f"cat /sys/class/infiniband/{rdma_devs[0]}/ports/1/link_layer 2>/dev/null; true"],
                timeout=10, use_debug=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                link_type = result.stdout.strip()  # "Ethernet" (RoCE) or "InfiniBand"
                break

        if link_type.lower() == "ethernet":
            rdma_type = "RoCE (RDMA over Converged Ethernet)"
        elif link_type.lower() == "infiniband":
            rdma_type = "InfiniBand"
        else:
            rdma_type = link_type

        active_count = len(rdma_devs)
        if exclude_hca:
            excluded = len([x for x in exclude_hca.split(",") if x.strip()])
            active_count = len(rdma_devs) - excluded
        elif ib_hca:
            active_count = len([x for x in ib_hca.split(",") if x.strip()])

        print(f"  RDMA:              {len(rdma_devs)} device(s), {active_count} active for NCCL")
        explain_cmd(f"{kx} bash -c \"ibv_devices 2>/dev/null || ls /sys/class/infiniband/\"",
                    "ibv_devices preferred; fallback to sysfs listing")
        explain_cmd(f"{kx} bash -c \"env | grep -E '^NCCL_(IB_HCA|EXCLUDE_IB_HCA|IB_DISABLE)'\"",
                    "NCCL_EXCLUDE_IB_HCA excludes devices; NCCL_IB_HCA selects; usable = total - excluded")
        print(f"  Link layer:        {rdma_type}")

        # RDMA NIC status from ibstat or sysfs
        sample_nic_status = sample.get("nic_status", {})
        if sample_nic_status:
            source = next(iter(sample_nic_status.values())).get("source", "unknown")
            state_counts = {}
            for dev, info in sample_nic_status.items():
                state = info.get("state", "?")
                state_counts[state] = state_counts.get(state, 0) + 1
            state_str = ", ".join(f"{s}: {c}" for s, c in sorted(state_counts.items()))
            print(f"  NIC port states:   {state_str} (via {source})")
            # Show rate from first active NIC
            for dev, info in sorted(sample_nic_status.items()):
                if "active" in info.get("state", "").lower():
                    rate = info.get("rate", "?")
                    fw = info.get("fw_ver", "?")
                    ca_type = info.get("ca_type", "")
                    rate_parts = [f"rate={rate}"]
                    if ca_type:
                        rate_parts.append(f"type={ca_type}")
                    rate_parts.append(f"fw={fw}")
                    print(f"  NIC detail:        {', '.join(rate_parts)} ({dev})")
                    break
            explain_cmd(f"{kx} ibstat {rdma_devs[0]} 2>/dev/null",
                        "ibstat shows CA type, port state, physical state, rate, link layer, firmware")
            explain_cmd(f"{kx} bash -c \"cat /sys/class/infiniband/{rdma_devs[0]}/ports/1/{{state,phys_state,rate,link_layer}}\"",
                        "sysfs fallback: read port attributes directly")
        else:
            explain_cmd(f"{kx} cat /sys/class/infiniband/{rdma_devs[0]}/ports/1/link_layer",
                        "Ethernet = RoCE; InfiniBand = native IB")

        if pci_nic_count > 0:
            print(f"  Network NICs:      {pci_nic_count} physical NIC(s) in PCI bus")
            explain_cmd(f"{kx} bash -c \"lspci -tv | grep -iE 'mellanox|connectx|infiniband' | grep -v 'Virtual Function'\"",
                        "count physical (non-VF) Mellanox/ConnectX NICs")
        if ib_disabled:
            print(f"  IB/RoCE status:    DISABLED (NCCL_IB_DISABLE=1)")
        else:
            print(f"  IB/RoCE status:    enabled")
        explain_cmd(f"{kx} bash -c \"env | grep NCCL_IB_DISABLE\"",
                    "if =1, IB/RoCE disabled; if unset or =0, enabled")
    else:
        print(f"  RDMA:              no devices found")
        explain_cmd(f"{kx} bash -c \"ls /sys/class/infiniband/ 2>/dev/null\"",
                    "empty = no RDMA devices")
        if pci_nic_count > 0:
            print(f"  Network NICs:      {pci_nic_count} physical NIC(s) in PCI bus")

    # Connection type matrix per pod type
    if len(pods) > 1 and pod_roles:
        print(f"\n{'─' * 50}")
        print("  Connection Type Matrix by Pod Type")
        print(f"{'─' * 50}")

        # Build per-role summaries
        role_order = [r for r in ["prefill", "decode", "other"] if r in pod_roles]
        # Collect per-pod-name -> role mapping
        pod_to_role = {}
        for pod_name, _ip in pods:
            name_lower = pod_name.lower()
            if "prefill" in name_lower:
                pod_to_role[pod_name] = "prefill"
            elif "decode" in name_lower:
                pod_to_role[pod_name] = "decode"
            else:
                pod_to_role[pod_name] = "other"

        for role in role_order:
            role_pods = [(pod_name, ip) for pod_name, ip in pods if pod_to_role.get(pod_name) == role]
            print(f"\n  {role.upper()} pods ({len(role_pods)}):")

            # GPU interconnect for this role
            role_conn_types = set()
            role_gpu_counts = set()
            for pod_name, _ in role_pods:
                topo = all_pod_results.get(pod_name, {})
                role_gpu_counts.add(topo.get("gpu_count", 0))
                for (g1, g2), conn in topo.get("connections", {}).items():
                    if g1 != g2 and conn != "X":
                        role_conn_types.add(conn)

            nv_conns = sorted(c for c in role_conn_types if c.startswith("NV") or c == "NVL")
            pcie_conns = sorted(c for c in role_conn_types if c in {"PIX", "PXB", "PHB", "NODE", "SYS"})

            gc_str = "/".join(str(g) for g in sorted(role_gpu_counts))
            print(f"    GPUs per pod:    {gc_str}")

            if nv_conns and not pcie_conns:
                print(f"    GPU group:       NVLink-READY ({', '.join(nv_conns)})")
            elif nv_conns and pcie_conns:
                print(f"    GPU group:       mixed NVLink ({', '.join(nv_conns)}) + PCIe ({', '.join(pcie_conns)})")
            elif pcie_conns:
                print(f"    GPU group:       PCIe only ({', '.join(pcie_conns)})")
            else:
                print(f"    GPU group:       no inter-GPU connections detected")

            # Network/RDMA for this role
            role_rdma_counts = set()
            role_active_counts = set()
            role_link_layers = set()
            for pod_name, _ in role_pods:
                topo = all_pod_results.get(pod_name, {})
                devs = topo.get("rdma_devices", [])
                role_rdma_counts.add(len(devs))
                env = topo.get("nccl_env", {})
                excl = env.get("NCCL_EXCLUDE_IB_HCA", "")
                hca = env.get("NCCL_IB_HCA", "")
                if hca:
                    role_active_counts.add(len([x for x in hca.split(",") if x.strip()]))
                elif excl:
                    role_active_counts.add(len(devs) - len([x for x in excl.split(",") if x.strip()]))
                else:
                    role_active_counts.add(len(devs))

            rdma_str = "/".join(str(r) for r in sorted(role_rdma_counts))
            active_str = "/".join(str(a) for a in sorted(role_active_counts))
            print(f"    RDMA devices:    {rdma_str} total, {active_str} active for NCCL")

            # Link layer (use first pod of this role)
            first_pn = role_pods[0][0]
            first_devs = all_pod_results.get(first_pn, {}).get("rdma_devices", [])
            if first_devs:
                result = exec_in_pod(
                    first_pn,
                    ["bash", "-c",
                     f"cat /sys/class/infiniband/{first_devs[0]}/ports/1/link_layer 2>/dev/null; true"],
                    timeout=10, use_debug=False,
                )
                ll = result.stdout.strip() if result.returncode == 0 else "unknown"
                if ll.lower() == "ethernet":
                    print(f"    Network group:   RoCE (RDMA over Converged Ethernet)")
                elif ll.lower() == "infiniband":
                    print(f"    Network group:   InfiniBand (native IB fabric)")
                else:
                    print(f"    Network group:   {ll}")
            else:
                print(f"    Network group:   no RDMA devices")

            ib_dis = any(
                all_pod_results.get(pod_name, {}).get("nccl_env", {}).get("NCCL_IB_DISABLE") == "1"
                for pod_name, _ in role_pods
            )
            if ib_dis:
                print(f"    IB/RoCE:         DISABLED on some/all pods")
            else:
                print(f"    IB/RoCE:         enabled")

    # High-level findings summary
    print(f"\n{'─' * 50}")
    print("  High-Level Findings")
    print(f"{'─' * 50}")

    # Collect aggregate data
    all_gpu_counts = [t.get("gpu_count", 0) for t in all_pod_results.values() if "gpu_count" in t]
    all_has_nvlink = [t.get("has_nvlink") for t in all_pod_results.values()]
    all_nvl_active = [t.get("nvlink_active", 0) for t in all_pod_results.values()]
    all_nvswitch_uniform = [t.get("nvswitch_uniform") for t in all_pod_results.values()]

    # Collect all connection types observed across pods (excluding self/X)
    all_conn_types = set()
    for t in all_pod_results.values():
        for (g1, g2), conn in t.get("connections", {}).items():
            if g1 != g2 and conn != "X":
                all_conn_types.add(conn)

    # Check lspci for NVSwitch ASICs across pods
    nvswitch_in_pci = False
    for pod_name, _ip in pods:
        result = exec_in_pod(
            pod_name,
            ["bash", "-c", "lspci 2>/dev/null | grep -ci nvswitch; true"],
            timeout=15, use_debug=False,
        )
        if result.returncode == 0:
            try:
                count = int(result.stdout.strip())
                if count > 0:
                    nvswitch_in_pci = True
                    break
            except ValueError:
                pass

    num_pods = len(pods)

    # GPU count
    if all_gpu_counts:
        if len(set(all_gpu_counts)) == 1:
            print(f"  GPU allocation:    {all_gpu_counts[0]} GPU(s) per pod, consistent across all {num_pods} pod(s)")
        else:
            print(f"  GPU allocation:    varies across pods — {dict(zip([display_names[p] for p, _ in pods], all_gpu_counts))}")
    explain_cmd(f"{kx} nvidia-smi topo -m | grep -c '^GPU'",
                "count topo matrix rows to determine how many accelerators are assigned to the pod")

    # Interconnect + NVLink-readiness assessment
    nvlink_pods = sum(1 for v in all_has_nvlink if v is True)
    pcie_pods = sum(1 for v in all_has_nvlink if v is False)

    # Classify observed connection types
    nv_types = sorted(c for c in all_conn_types if c.startswith("NV") or c == "NVL")
    pcie_types_seen = sorted(c for c in all_conn_types if c in {"PIX", "PXB", "PHB", "NODE", "SYS"})
    other_types = sorted(all_conn_types - set(nv_types) - set(pcie_types_seen))

    if all_conn_types:
        print(f"  Connection types:  {', '.join(sorted(all_conn_types))} observed across all GPU pairs")
    else:
        if all_gpu_counts and max(all_gpu_counts) <= 1:
            print(f"  Connection types:  N/A — single GPU per pod, no inter-GPU connections")
        else:
            print(f"  Connection types:  none detected")
    explain_cmd(f"{kx} nvidia-smi topo -m",
                "read the matrix values between GPU rows: NV## = NVLink, PIX/PXB/PHB/SYS = PCIe paths, X = self")

    if nvlink_pods == num_pods:
        print(f"  GPU interconnect:  NVLink/NVSwitch on all {num_pods} pod(s)")
        # Check if ALL GPU pairs use NVLink (fully NVLink-ready)
        if pcie_types_seen:
            print(f"                     PARTIAL: some GPU pairs still use PCIe paths ({', '.join(pcie_types_seen)})")
            print(f"                     Not all GPU pairs are NVLink-connected")
        else:
            # All pairs connected via NV#
            if nv_types:
                widths = []
                for nt in nv_types:
                    suffix = nt[2:]
                    if suffix.isdigit():
                        widths.append(f"{nt} = {suffix} NVLink lanes")
                    elif nt == "NVL":
                        widths.append("NVL = NVLink")
                width_desc = ", ".join(widths) if widths else ", ".join(nv_types)
                print(f"                     NVLink-READY: all GPU pairs connected via NVLink ({width_desc})")
            else:
                print(f"                     NVLink-READY: all GPU pairs connected via NVLink")
    elif pcie_pods == num_pods:
        print(f"  GPU interconnect:  PCIe only on all {num_pods} pod(s)")
        if nvswitch_in_pci:
            print(f"                     NOTE: NVSwitch hardware detected in PCI bus but NVLink")
            print(f"                     is NOT exposed to the container GPU slice — the host has")
            print(f"                     NVSwitch ASICs but pods only see a subset of GPUs without")
            print(f"                     NVLink connectivity between them")
    elif nvlink_pods > 0:
        print(f"  GPU interconnect:  mixed — {nvlink_pods} pod(s) NVLink, {pcie_pods} pod(s) PCIe only")

    # Hop distance
    any_hop_issue = any("excessive hops" in w for w in warnings)
    if any_hop_issue:
        print(f"  Hop distance:      sub-optimal — some GPU pairs traverse high-latency PCIe paths")
        print(f"                     (SYS = cross-socket, PHB = cross PCIe root complex)")
    else:
        print(f"  Hop distance:      OK — no excessive hops between GPU pairs")
    explain_cmd(f"{kx} nvidia-smi topo -m | grep -E 'SYS|PHB'",
                "SYS = cross-socket hop (worst), PHB = cross PCIe root (bad); absence = good")

    # NUMA affinity
    any_numa_split = any("span" in w and "NUMA" in w for w in warnings)
    if any_numa_split:
        split_pods = sum(1 for t in all_pod_results.values()
                         if len(set(t.get("gpu_numa", {}).values()) - {-1}) > 1)
        print(f"  NUMA affinity:     SPLIT — {split_pods} pod(s) have GPUs on different NUMA nodes")
        print(f"                     Cross-socket PCIe traffic adds latency and contention")
        print(f"                     to collective ops (allreduce, allgather, send/recv)")
    else:
        all_numa = [set(t.get("gpu_numa", {}).values()) - {-1} for t in all_pod_results.values()
                    if t.get("gpu_numa")]
        if all_numa and all(len(ns) <= 1 for ns in all_numa):
            print(f"  NUMA affinity:     OK — all GPUs on same NUMA node per pod")
        elif not all_numa:
            print(f"  NUMA affinity:     could not determine")
        else:
            print(f"  NUMA affinity:     OK")
    explain_cmd(
        f"{kx} bash -c \"nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | "
        f"while IFS=, read idx bus; do bus=$(echo $bus | tr -d ' ' | sed 's/^0\\{{4,\\}}/0000/'); "
        f"echo GPU$idx: NUMA $(cat /sys/bus/pci/devices/$bus/numa_node 2>/dev/null); done\"",
        "maps each GPU to its NUMA node via PCI bus ID; all same = OK, different = SPLIT")

    # PCIe root spread
    any_pcie_spread = any("PCIe root port" in w for w in warnings)
    if any_pcie_spread:
        spread_pods = sum(1 for t in all_pod_results.values()
                          if len(set(info["root_port"] for info in t.get("gpu_pcie", {}).values()
                                     if info.get("root_port") != "unknown")) > 1)
        print(f"  PCIe root spread:  {spread_pods} pod(s) have GPUs under different PCIe root ports")
        print(f"                     Cross-root P2P transfers route through CPU fabric, reducing bandwidth")
    else:
        all_roots = [set(info["root_port"] for info in t.get("gpu_pcie", {}).values()
                         if info.get("root_port") != "unknown")
                     for t in all_pod_results.values() if t.get("gpu_pcie")]
        if all_roots and all(len(rs) <= 1 for rs in all_roots):
            print(f"  PCIe root spread:  OK — all GPUs under same root port per pod")
        elif not all_roots:
            print(f"  PCIe root spread:  could not determine")
        else:
            print(f"  PCIe root spread:  OK")
    explain_cmd(
        f"{kx} bash -c \"nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | "
        f"while IFS=, read idx bus; do bus=$(echo $bus | tr -d ' ' | sed 's/^0\\{{4,\\}}/0000/'); "
        f"chain=$(readlink -f /sys/bus/pci/devices/$bus | grep -oE '[0-9a-f]{{4}}:[0-9a-f]{{2}}:[0-9a-f]{{2}}\\.[0-9a-f]' | head -2); "
        f"echo GPU$idx: root=$(echo $chain | awk '{{print $2}}'); done\"",
        "shows PCI root port for each GPU; all same = OK, different = GPUs span root complexes")

    # PCIe switch topology (PXB)
    any_pxb = any("PXB" in w for w in warnings)
    if any_pxb:
        pxb_pods = sum(1 for t in all_pod_results.values()
                       if len({sw for sw, gpus in t.get("pcie_switch_groups", {}).items()
                               if sw != "unknown"}) > 1)
        print(f"  PCIe switch topo:  {pxb_pods} pod(s) have GPUs under different PCIe switches")
        print(f"                     PXB = cross-switch but same root complex — GPU pairs share")
        print(f"                     a bottlenecked upstream link, reducing P2P bandwidth")
    else:
        all_switches = [
            {sw for sw in t.get("pcie_switch_groups", {}) if sw != "unknown"}
            for t in all_pod_results.values() if t.get("pcie_switch_groups")
        ]
        if all_switches and all(len(sws) <= 1 for sws in all_switches):
            print(f"  PCIe switch topo:  OK — all GPUs behind same PCIe switch per pod")
        elif not all_switches:
            print(f"  PCIe switch topo:  could not determine")
        else:
            print(f"  PCIe switch topo:  OK — no PXB cross-switch connections detected")
    explain_cmd(f"{kx} nvidia-smi topo -m | grep PXB",
                "PXB in matrix = cross PCIe switch (shared upstream); absence = all behind same switch")
    explain_cmd(
        f"{kx} bash -c \"nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | "
        f"while IFS=, read idx bus; do bus=$(echo $bus | tr -d ' ' | sed 's/^0\\{{4,\\}}/0000/'); "
        f"dev=$(readlink -f /sys/bus/pci/devices/$bus); parent=$(dirname $dev); "
        f"sw=$(basename $parent); class=$(cat $parent/class 2>/dev/null); "
        f"echo GPU$idx: switch=$sw class=$class; done\"",
        "class 0x0604 = PCI bridge; same parent bridge = co-located, different = separate switches")

    # NVSwitch domain
    if any(v is True for v in all_has_nvlink):
        if all(v is True for v in all_nvswitch_uniform if v is not None):
            print(f"  NVSwitch domain:   uniform — all GPUs in same NVLink group per pod")
        elif any(v is False for v in all_nvswitch_uniform):
            print(f"  NVSwitch domain:   SPLIT — GPUs span different NVSwitch domains in some pods")
        else:
            print(f"  NVSwitch domain:   uniform — no split detected")
    else:
        print(f"  NVSwitch domain:   N/A — no NVLink connections active in pods")
    explain_cmd(f"{kx} nvidia-smi topo -m",
                "uniform = all NV## values identical between GPU pairs; different NV## values = split domains")

    # NVLink status
    total_active = sum(all_nvl_active)
    total_inactive = sum(t.get("nvlink_inactive", 0) for t in all_pod_results.values())
    if total_active > 0 or total_inactive > 0:
        print(f"  NVLink status:     {total_active} active, {total_inactive} inactive across all pods")
        if total_inactive > 0:
            print(f"                     WARNING: inactive NVLink connections may indicate hardware issues")
    else:
        print(f"  NVLink status:     no active NVLink connections in any pod")
    explain_cmd(f"{kx} nvidia-smi nvlink --status | grep -cE '(active|inactive)'",
                "count active/inactive NVLink connections per GPU")

    # NCCL config
    nccl_envs = [t.get("nccl_env", {}) for t in all_pod_results.values()]
    any_nccl_mismatch = any("MISMATCH:" in w and "differs across pods" in w for w in warnings)
    ib_disabled_pods = sum(1 for e in nccl_envs if e.get("NCCL_IB_DISABLE") == "1")
    hca_set_pods = sum(1 for e in nccl_envs if e.get("NCCL_IB_HCA"))
    exclude_set_pods = sum(1 for e in nccl_envs if e.get("NCCL_EXCLUDE_IB_HCA"))
    socket_set_pods = sum(1 for e in nccl_envs if e.get("NCCL_SOCKET_IFNAME"))

    print(f"  NCCL config:")
    explain_cmd(f"{kx} bash -c \"env | grep -E '^NCCL_' | sort\"",
                "all NCCL environment variables controlling collective transport")
    if ib_disabled_pods > 0:
        print(f"    IB/RoCE:         DISABLED on {ib_disabled_pods}/{num_pods} pod(s) — RDMA will not be used")
    else:
        print(f"    IB/RoCE:         enabled (NCCL_IB_DISABLE not set or =0)")
    if hca_set_pods > 0:
        sample_hca = next((e["NCCL_IB_HCA"] for e in nccl_envs if e.get("NCCL_IB_HCA")), "")
        print(f"    NCCL_IB_HCA:     set on {hca_set_pods}/{num_pods} pod(s) ({sample_hca})")
    elif exclude_set_pods > 0:
        sample_excl = next((e["NCCL_EXCLUDE_IB_HCA"] for e in nccl_envs if e.get("NCCL_EXCLUDE_IB_HCA")), "")
        rdma_count = len(list(all_pod_results.values())[0].get("rdma_devices", []))
        excl_count = len([x for x in sample_excl.split(",") if x.strip()])
        active_count = rdma_count - excl_count
        print(f"    Device filter:   NCCL_EXCLUDE_IB_HCA excludes {excl_count} of {rdma_count} RDMA devices "
              f"({active_count} active)")
    else:
        rdma_count = len(list(all_pod_results.values())[0].get("rdma_devices", [])) if all_pod_results else 0
        if rdma_count > 0:
            print(f"    Device filter:   none — NCCL will use all {rdma_count} RDMA device(s)")
        else:
            print(f"    Device filter:   no RDMA devices found")
    if socket_set_pods > 0:
        sample_sock = next((e["NCCL_SOCKET_IFNAME"] for e in nccl_envs if e.get("NCCL_SOCKET_IFNAME")), "")
        print(f"    Socket iface:    {sample_sock} on {socket_set_pods}/{num_pods} pod(s)")
    else:
        print(f"    Socket iface:    auto-select (NCCL_SOCKET_IFNAME not set)")
    if any_nccl_mismatch:
        print(f"    Consistency:     MISMATCH — env vars differ across pods (see warnings above)")
    else:
        print(f"    Consistency:     OK — env vars match across all pods")

    # UCX config summary
    ucx_tls_pods = sum(1 for e in nccl_envs if e.get("UCX_TLS"))
    ucx_net_pods = sum(1 for e in nccl_envs if e.get("UCX_NET_DEVICES"))
    ucx_any_pods = sum(1 for e in nccl_envs if any(k.startswith("UCX_") for k in e))
    nccl_net_ucx = sum(1 for e in nccl_envs if (e.get("NCCL_NET") or "").lower() == "ucx")

    if ucx_any_pods > 0 or nccl_net_ucx > 0:
        print(f"  UCX config:")
        explain_cmd(f"{kx} bash -c \"env | grep -E '^UCX_' | sort\"",
                    "all UCX environment variables controlling transport selection and RDMA behavior")
        if nccl_net_ucx > 0:
            print(f"    NCCL transport:  UCX plugin (NCCL_NET=UCX on {nccl_net_ucx}/{num_pods} pod(s))")
        if ucx_tls_pods > 0:
            sample_tls = next((e["UCX_TLS"] for e in nccl_envs if e.get("UCX_TLS")), "")
            print(f"    UCX_TLS:         {sample_tls}")
            tls_list = [t.strip().lower() for t in sample_tls.split(",")]
            rdma_tls = {"rc", "rc_v", "rc_x", "ud", "ud_v", "ud_x", "dc", "dc_x"}
            gpu_tls = {"cuda_copy", "cuda_ipc", "gdr_copy"}
            active_rdma = sorted(set(tls_list) & rdma_tls)
            active_gpu = sorted(set(tls_list) & gpu_tls)
            if active_rdma:
                print(f"                     RDMA transports: {', '.join(active_rdma)}")
            if active_gpu:
                print(f"                     GPU transports: {', '.join(active_gpu)}")
            if "tcp" in tls_list:
                print(f"                     WARNING: tcp fallback enabled — slow for GPU workloads")
        else:
            print(f"    UCX_TLS:         not set (UCX auto-selects)")
        if ucx_net_pods > 0:
            sample_net = next((e["UCX_NET_DEVICES"] for e in nccl_envs if e.get("UCX_NET_DEVICES")), "")
            print(f"    UCX_NET_DEVICES: {sample_net}")
        else:
            print(f"    UCX_NET_DEVICES: not set (UCX auto-selects RDMA devices)")
        # Show other notable UCX vars
        for var_name, desc in [
            ("UCX_MAX_RNDV_RAILS", "multi-rail parallelism"),
            ("UCX_RNDV_THRESH", "rendezvous threshold"),
            ("UCX_MEMTYPE_CACHE", "GPU memory type cache"),
        ]:
            val_pods = sum(1 for e in nccl_envs if e.get(var_name))
            if val_pods > 0:
                sample_val = next((e[var_name] for e in nccl_envs if e.get(var_name)), "")
                print(f"    {var_name}: {sample_val} ({desc})")

    # vLLM / CUDA / NIXL config summary
    vllm_pods = sum(1 for e in nccl_envs if any(k.startswith("VLLM_") for k in e))
    cuda_pods = sum(1 for e in nccl_envs if any(k.startswith("CUDA_") for k in e))
    nixl_pods = sum(1 for e in nccl_envs if any(k.startswith("NIXL_") for k in e))

    if vllm_pods > 0 or cuda_pods > 0 or nixl_pods > 0:
        print(f"\n  Inference runtime config:")
        explain_cmd(f"{kx} bash -c \"env | grep -E '^(VLLM_|CUDA_|TORCH_|NIXL_|MASTER_|RANK=|WORLD_SIZE=|LOCAL_RANK=)' | sort\"",
                    "all vLLM, CUDA, PyTorch distributed, and NIXL environment variables")

        # CUDA config
        if cuda_pods > 0:
            sample_cuda_order = next((e.get("CUDA_DEVICE_ORDER") for e in nccl_envs
                                      if e.get("CUDA_DEVICE_ORDER")), None)
            sample_cuda_module = next((e.get("CUDA_MODULE_LOADING") for e in nccl_envs
                                       if e.get("CUDA_MODULE_LOADING")), None)
            sample_cuda_visible = next((e.get("CUDA_VISIBLE_DEVICES") for e in nccl_envs
                                        if e.get("CUDA_VISIBLE_DEVICES")), None)
            if sample_cuda_visible:
                print(f"    CUDA_VISIBLE_DEVICES: {sample_cuda_visible}")
            if sample_cuda_order:
                print(f"    CUDA_DEVICE_ORDER: {sample_cuda_order}")
            elif any(t.get("gpu_count", 0) > 1 for t in all_pod_results.values()):
                print(f"    CUDA_DEVICE_ORDER: not set (recommend PCI_BUS_ID for stable ordering)")
            if sample_cuda_module:
                print(f"    CUDA_MODULE_LOADING: {sample_cuda_module}")
            else:
                print(f"    CUDA_MODULE_LOADING: not set (defaults to EAGER; LAZY reduces memory)")

            # Check for debugging vars in production
            any_blocking = any(e.get("CUDA_LAUNCH_BLOCKING") == "1" for e in nccl_envs)
            if any_blocking:
                print(f"    WARNING: CUDA_LAUNCH_BLOCKING=1 on some pods — severe perf penalty")

        explain_cmd(f"{kx} bash -c \"env | grep -E '^CUDA_'\"",
                    "CUDA environment: device visibility, ordering, module loading mode")

        # vLLM config
        if vllm_pods > 0:
            sample_vllm = next((e for e in nccl_envs if any(k.startswith("VLLM_") for k in e)), {})
            dp_size = sample_vllm.get("VLLM_DP_SIZE")
            host_ip = sample_vllm.get("VLLM_HOST_IP")
            target_dev = sample_vllm.get("VLLM_TARGET_DEVICE")
            disable_pynccl = sample_vllm.get("VLLM_DISABLE_PYNCCL")

            if host_ip:
                print(f"    VLLM_HOST_IP:    {host_ip}")
            if dp_size:
                print(f"    VLLM_DP_SIZE:    {dp_size}")
            if target_dev:
                print(f"    VLLM_TARGET_DEVICE: {target_dev}")
            if disable_pynccl == "1":
                print(f"    WARNING: VLLM_DISABLE_PYNCCL=1 — PyNCCL disabled, slower fallback")

            # Show other notable vLLM vars
            for var_name, desc in [
                ("VLLM_CPU_KVCACHE_SPACE", "CPU KV cache offload (GiB)"),
                ("VLLM_ENGINE_ITERATION_TIMEOUT_S", "engine iteration timeout"),
                ("VLLM_USE_NCCL_SYMM_MEM", "NCCL symmetric memory"),
                ("VLLM_NCCL_SO_PATH", "custom NCCL library path"),
                ("VLLM_WORKER_MULTIPROC_METHOD", "worker process method"),
                ("VLLM_FLOAT32_MATMUL_PRECISION", "matmul precision"),
            ]:
                val = sample_vllm.get(var_name)
                if val:
                    print(f"    {var_name}: {val} ({desc})")

        explain_cmd(f"{kx} bash -c \"env | grep -E '^VLLM_'\"",
                    "vLLM inference engine configuration variables")

        # NIXL / llm-d disaggregated KV transfer
        if nixl_pods > 0:
            sample_nixl = next((e for e in nccl_envs if any(k.startswith("NIXL_") for k in e)), {})
            nixl_etcd = sample_nixl.get("NIXL_ETCD_ENDPOINTS")
            nixl_ns = sample_nixl.get("NIXL_ETCD_NAMESPACE")
            print(f"    llm-d / NIXL:    configured on {nixl_pods}/{num_pods} pod(s)")
            if nixl_etcd:
                print(f"    NIXL_ETCD_ENDPOINTS: {nixl_etcd}")
            if nixl_ns:
                print(f"    NIXL_ETCD_NAMESPACE: {nixl_ns}")
            # Check if RDMA is available for KV transfer
            sample_rdma = sample.get("rdma_devices", [])
            if not sample_rdma:
                print(f"    WARNING: NIXL configured but no RDMA devices — KV transfer will use TCP")
            else:
                print(f"    KV transfer:     RDMA available ({len(sample_rdma)} device(s))")

        explain_cmd(f"{kx} bash -c \"env | grep -E '^NIXL_'\"",
                    "NIXL agent discovery and KV cache transfer configuration for llm-d")

        # NCCL tuning for inference
        any_p2p_disable = any(e.get("NCCL_P2P_DISABLE") == "1" for e in nccl_envs)
        any_shm_disable = any(e.get("NCCL_SHM_DISABLE") == "1" for e in nccl_envs)
        sample_max_ctas = next((e.get("NCCL_MAX_CTAS") for e in nccl_envs
                                if e.get("NCCL_MAX_CTAS")), None)
        if any_p2p_disable:
            print(f"    WARNING: NCCL_P2P_DISABLE=1 on some pods — GPU P2P transfers disabled")
        if any_shm_disable:
            print(f"    WARNING: NCCL_SHM_DISABLE=1 on some pods — shared memory transport disabled")
        if sample_max_ctas:
            print(f"    NCCL_MAX_CTAS:   {sample_max_ctas} (limits GPU contention from NCCL kernels)")

        # PyTorch distributed
        master_set = sum(1 for e in nccl_envs if e.get("MASTER_ADDR"))
        if master_set > 0:
            sample_master = next((e.get("MASTER_ADDR") for e in nccl_envs if e.get("MASTER_ADDR")), "")
            sample_world = next((e.get("WORLD_SIZE") for e in nccl_envs if e.get("WORLD_SIZE")), "?")
            print(f"    PyTorch distributed: MASTER_ADDR={sample_master}, WORLD_SIZE={sample_world}")
        explain_cmd(f"{kx} bash -c \"env | grep -E '^(MASTER_|WORLD_SIZE|RANK=|LOCAL_RANK=)'\"",
                    "PyTorch distributed coordination: master address, world size, rank")

    # Print verification summary if -x was used
    if _get_common().EXPLAIN_VERIFY:
        _print_verify_summary()

    print()
    return all_pod_results


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-discovery.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _USAGE = """\
Usage: uv run run-tests-discovery.py [options]

Run GPU topology discovery on Kubernetes inference pods.

Equivalent to: run-tests.sh -d

Options:
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -h, --help            Show this help message.
  -i, --install-deps    Install missing tools (ibstat, ibv_devices) if needed.
  -l, --label SELECTOR  Label selector to discover pods
                        (default: "llm-d.ai/inferenceServing=true").
  -n, --namespace NS    Kubernetes namespace for all kubectl commands.
  -v, --verbose         Print kubectl commands as they run.
  -x, --explain-verify  Run each explain command and verify output.
                        Implies --explain.
"""
    if "-h" in sys.argv or "--help" in sys.argv:
        print(_USAGE)
        sys.exit(0)

    _c = _get_common()
    _cfg = _c._parse_common_args()
    _cfg["DISCOVER_ONLY"] = True
    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    run_topology_validation(_pods, _display_names)
