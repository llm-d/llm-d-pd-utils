"""Shared utilities and configuration for run-tests*.py modules.

Provides global configuration, kubectl helpers, command execution,
pod discovery, explain/verify logic, and server process cleanup.

All submodules (run-tests-discovery.py, run-tests-perftest.py,
run-tests-iperf3.py) import from this module via importlib to
avoid circular dependencies with the main run-tests.py entry point.
"""

import atexit
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Global configuration — defaults set here, overridden by configure()
# ---------------------------------------------------------------------------
VERBOSE = False
DISCOVER_ONLY = False
EXPLAIN = False
EXPLAIN_VERIFY = False

USE_DEBUG_CONTAINER = False
DEBUG_IMAGE = ""
DEBUG_SUFFIX = "-debug"

INSTALL_DEPS = False
OPT_DEVICE = None
OPT_GID_INDEX = None
OPT_LABEL = "llm-d.ai/model"
OPT_NAMESPACE = None

SSH_COMMAND = None       # e.g. ["ssh", "-i", "/path/key"]
SSH_HOSTS = []           # [(display_name, ip), ...] — same shape as discover_pods()
SSH_MODE = False         # derived: True when SSH_HOSTS is non-empty
_ssh_target_map = {}     # display_name -> ssh_target (e.g. "user@host")

DISPLAY_NAME_MAX_LEN = 32

PERFTEST_DURATION = 5
PERFTEST_ITERATIONS = 1000
PERFTEST_PORT = 18515

RDMA_BLOCK_SIZES = [1073741824]  # 1G default
RDMA_LATENCY_SIZE = 2
OPT_PERCENTAGE_MARGIN = 10.0


def configure(**kwargs):
    """Set global config values.  Called by main() after CLI parsing."""
    global SSH_MODE
    g = globals()
    for key, val in kwargs.items():
        if key not in g:
            raise ValueError(f"Unknown config key: {key}")
        g[key] = val
    SSH_MODE = bool(SSH_HOSTS)


def _parse_common_args(argv=None, extra_flags=None):
    """Parse CLI flags shared by all submodule entry points.

    Returns a dict suitable for passing to configure().

    *extra_flags* is an optional dict mapping flag tuples to
    (config_key, needs_value) pairs so callers can extend parsing
    without duplicating the loop.  Example::

        extra_flags={
            ("-r", "--rdma-device"): ("OPT_DEVICE", True),
            ("-i", "--install-deps"): ("INSTALL_DEPS", False),
        }

    Boolean (needs_value=False) flags store True when present.
    Value flags store the next argument or the part after '='.
    """
    if argv is None:
        argv = sys.argv[1:]

    if extra_flags is None:
        extra_flags = {}

    # Built-in shared flags
    all_flags = {
        ("-v", "--verbose"):            ("VERBOSE", False),
        ("-e", "--explain"):            ("EXPLAIN", False),
        ("-x", "--explain-verify"):     ("EXPLAIN_VERIFY", False),
        ("-i", "--install-deps"):       ("INSTALL_DEPS", False),
        ("-l", "--label"):              ("OPT_LABEL", True),
        ("-n", "--namespace"):          ("OPT_NAMESPACE", True),
        ("-D", "--debug-image"):        ("DEBUG_IMAGE", True),
        ("-p", "--percentage-margin"):  ("_PCT_MARGIN_RAW", True),
        ("--ssh-command", "--ssh-command"):  ("_SSH_COMMAND_RAW", True),
        ("--ssh-hosts", "--ssh-hosts"):      ("_SSH_HOSTS_RAW", True),
    }
    all_flags.update(extra_flags)

    # Build lookup: short/long flag -> (config_key, needs_value, long_form)
    lookup = {}
    for flag_tuple, (key, needs_val) in all_flags.items():
        long_form = flag_tuple[1]  # e.g. "--label"
        for f in flag_tuple:
            lookup[f] = (key, needs_val, long_form)

    cfg = {}
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        # Check --flag=value form
        if "=" in arg:
            prefix = arg.split("=", 1)[0]
            if prefix in lookup:
                key, needs_val, _ = lookup[prefix]
                if needs_val:
                    cfg[key] = arg.split("=", 1)[1]
                continue

        if arg in lookup:
            key, needs_val, _ = lookup[arg]
            if needs_val:
                if i + 1 < len(argv):
                    cfg[key] = argv[i + 1]
                    skip_next = True
            else:
                cfg[key] = True

    # -x implies -e
    if cfg.get("EXPLAIN_VERIFY"):
        cfg["EXPLAIN"] = True

    # Derive USE_DEBUG_CONTAINER from DEBUG_IMAGE
    if "DEBUG_IMAGE" in cfg:
        cfg["USE_DEBUG_CONTAINER"] = bool(cfg["DEBUG_IMAGE"])
        cfg["DEBUG_SUFFIX"] = "-debug"

    # Convert percentage margin to float
    raw_pct = cfg.pop("_PCT_MARGIN_RAW", None)
    if raw_pct is not None:
        cfg["OPT_PERCENTAGE_MARGIN"] = float(raw_pct)

    # Strip trailing bare '=' from label (e.g. "llm-d.ai/model=" → "llm-d.ai/model")
    lbl = cfg.get("OPT_LABEL", "")
    if lbl.endswith("=") and "=" not in lbl[:-1]:
        cfg["OPT_LABEL"] = lbl[:-1]

    # Handle SSH flags
    ssh_hosts_raw = cfg.pop("_SSH_HOSTS_RAW", None)
    ssh_cmd_raw = cfg.pop("_SSH_COMMAND_RAW", None)
    if ssh_hosts_raw:
        cfg["SSH_COMMAND"] = shlex.split(ssh_cmd_raw) if ssh_cmd_raw else ["ssh"]
        cfg["SSH_HOSTS"], cfg["_ssh_target_map"] = _parse_ssh_hosts(ssh_hosts_raw)
    elif ssh_cmd_raw:
        print("Error: --ssh-command requires --ssh-hosts", file=sys.stderr)
        sys.exit(1)

    return cfg


def _parse_ssh_hosts(hosts_str):
    """Parse comma-separated host list into (display_name, ip) tuples.

    Supported formats per entry:
      host                    → display_name=host, ip=host, target=host
      user@host               → display_name=host, ip=host, target=user@host
      host:ip                 → display_name=host, ip=ip,   target=host
      user@host:ip            → display_name=host, ip=ip,   target=user@host

    Returns (hosts_list, target_map) where hosts_list matches
    discover_pods() format [(display_name, ip), ...] and
    target_map maps display_name -> ssh_target.
    """
    hosts = []
    target_map = {}
    for entry in hosts_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Split off explicit IP after ':'
        if ":" in entry:
            ssh_part, ip = entry.rsplit(":", 1)
        else:
            ssh_part, ip = entry, None
        # Extract display name (strip user@ prefix)
        if "@" in ssh_part:
            display_name = ssh_part.split("@", 1)[1]
        else:
            display_name = ssh_part
        if ip is None:
            ip = display_name
        hosts.append((display_name, ip))
        target_map[display_name] = ssh_part
    return hosts, target_map


def _is_latency_title(title):
    t = title.lower()
    return "latency" in t or "jitter" in t or "loss" in t


def print_results_summary(pods, all_results, display_names, margin_pct=None):
    """Print average per metric, flag any result outside margin_pct of the average."""
    if margin_pct is None:
        margin_pct = OPT_PERCENTAGE_MARGIN
    n = len(pods)
    metrics = list(all_results.keys())
    if not metrics:
        return

    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY (margin: \u00b1{margin_pct:.0f}%)")
    print(f"{'=' * 60}")

    any_outlier = False
    for m in metrics:
        matrix = all_results[m]
        is_lat = _is_latency_title(m)

        vals = []
        entries = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = matrix[i][j]
                if v is not None:
                    vals.append(v)
                    entries.append((i, j, v))

        if not vals:
            print(f"\n  {m}: no results")
            continue

        avg = sum(vals) / len(vals)
        min_v = min(vals)
        max_v = max(vals)
        fmt = ".4f" if is_lat else ".2f"
        print(f"\n  {m}:")
        print(f"    Average: {avg:{fmt}}  Min: {min_v:{fmt}}  Max: {max_v:{fmt}}  "
              f"({len(vals)} result(s))")

        if avg == 0:
            continue
        threshold = margin_pct / 100.0
        outliers = []
        for i, j, v in entries:
            deviation = abs(v - avg) / avg
            if deviation > threshold:
                outliers.append((i, j, v, deviation))

        if outliers:
            any_outlier = True
            print(f"    OUTLIERS (>{margin_pct:.0f}% from average):")
            for i, j, v, dev in sorted(outliers, key=lambda x: -x[3]):
                src = display_names[pods[i][0]]
                dst = display_names[pods[j][0]]
                src_role = get_pod_role(pods[i][0])
                dst_role = get_pod_role(pods[j][0])
                role_str = ""
                if src_role or dst_role:
                    role_str = f" ({src_role or '?'}->{dst_role or '?'})"
                sign = "+" if v > avg else "-"
                print(f"      {src} -> {dst}{role_str}: "
                      f"{v:{fmt}} ({sign}{dev * 100:.1f}% from avg {avg:{fmt}})")
        else:
            print(f"    All results within \u00b1{margin_pct:.0f}% of average")

    if not any_outlier:
        print(f"\n  All metrics within \u00b1{margin_pct:.0f}% margin \u2014 results are consistent.")
    print()


def _discover_and_display(label=None):
    """Discover pods, detect roles, and print summary.  Returns (pods, display_names)."""
    lbl = label or OPT_LABEL
    print(f"Discovering pods with label: {lbl} ...")
    pods = discover_pods(lbl)
    display_names = make_display_names([p[0] for p in pods])
    discover_pod_roles(lbl)
    print(f"Found {len(pods)} pod(s):")
    for name, ip in pods:
        role = get_pod_role(name)
        role_tag = f"  [{role}]" if role else ""
        print(f"  {display_names[name]:{DISPLAY_NAME_MAX_LEN}s}  {name}  ({ip}){role_tag}")

    # Show cross-role filtering status
    test_pairs = _cross_role_pairs(pods)
    all_pairs_count = len(pods) * (len(pods) - 1)
    if len(test_pairs) < all_pairs_count:
        prefill_count = sum(1 for n, _ in pods if get_pod_role(n) == "prefill")
        decode_count = sum(1 for n, _ in pods if get_pod_role(n) == "decode")
        print(f"  Cross-role testing: {prefill_count} prefill + {decode_count} decode pods "
              f"→ {len(test_pairs)} cross-role pairs (skipping {all_pairs_count - len(test_pairs)} same-role pairs)")

    return pods, display_names


# ---------------------------------------------------------------------------
# Global server process cleanup (thread-safe)
# ---------------------------------------------------------------------------
_server_procs = []
_server_procs_lock = threading.Lock()


def _server_procs_append(proc):
    with _server_procs_lock:
        _server_procs.append(proc)


def _server_procs_remove(proc):
    with _server_procs_lock:
        if proc in _server_procs:
            _server_procs.remove(proc)


def _cleanup_servers():
    with _server_procs_lock:
        procs = list(_server_procs)
    for proc in procs:
        try:
            proc.terminate()
        except OSError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
    with _server_procs_lock:
        _server_procs.clear()


atexit.register(_cleanup_servers)
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, lambda s, f: sys.exit(1))


# ---------------------------------------------------------------------------
# Threading utilities (shared by perftest, iperf3, discovery)
# ---------------------------------------------------------------------------
class _StreamingBuffer:
    """Thread-safe buffer that supports incremental flushing to stdout.

    The producer thread calls write().  The consumer (main thread) calls
    flush_new() to print everything written since the last flush, or
    flush_all() to print the complete buffer.
    """

    def __init__(self):
        self._buf = io.StringIO()
        self._lock = threading.Lock()
        self._flushed = 0  # character offset already printed

    def write(self, s):
        with self._lock:
            self._buf.write(s)

    def flush_new(self):
        """Print any new content since last flush. Returns True if anything was printed."""
        with self._lock:
            val = self._buf.getvalue()
        new = val[self._flushed:]
        if new:
            print(new, end="", flush=True)
            self._flushed += len(new)
            return True
        return False

    def flush_all(self):
        """Print entire buffer (anything not yet flushed)."""
        self.flush_new()

    def getvalue(self):
        with self._lock:
            return self._buf.getvalue()


def _schedule_parallel_pairs(n, pairs=None):
    """Schedule directed pairs into waves of non-conflicting pairs.

    If *pairs* is given, only those (i, j) tuples are scheduled.
    Otherwise all n*(n-1) directed pairs are used.

    Returns a list of waves, where each wave is a list of (i, j) index tuples.
    Within a wave, no index appears as source or destination more than once,
    so pairs in the same wave can safely run in parallel.
    """
    if pairs is None:
        remaining = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        remaining = list(pairs)
    waves = []
    while remaining:
        wave = []
        used = set()
        still_remaining = []
        for pair in remaining:
            i, j = pair
            if i not in used and j not in used:
                wave.append(pair)
                used.add(i)
                used.add(j)
            else:
                still_remaining.append(pair)
        waves.append(wave)
        remaining = still_remaining
    return waves


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------
def _parse_size(s):
    """Parse a size string like '65536', '64K', '1M', '1G' into bytes (int)."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def _human_size(n):
    """Return a human-readable size string."""
    if n >= 1024**3 and n % 1024**3 == 0:
        return f"{n // 1024**3}G"
    if n >= 1024**2 and n % 1024**2 == 0:
        return f"{n // 1024**2}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def _strip_ansi(text):
    """Remove ANSI escape codes (colors, underline, bold, etc.) from text."""
    return _ANSI_RE.sub('', text)


# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------
def _kubectl_ns_args():
    """Return ['-n', namespace] if OPT_NAMESPACE is set, else []."""
    if OPT_NAMESPACE:
        return ["-n", OPT_NAMESPACE]
    return []


def _kubectl_exec_prefix(pod_name="<POD>"):
    """Build the exec prefix string for explain output (kubectl or SSH)."""
    if SSH_MODE:
        ssh_cmd_str = " ".join(SSH_COMMAND) if SSH_COMMAND else "ssh"
        target = _ssh_target_map.get(pod_name, pod_name)
        return f"{ssh_cmd_str} {target}"
    ns = f" -n {OPT_NAMESPACE}" if OPT_NAMESPACE else ""
    return f"kubectl exec{ns} {pod_name} --"


# ---------------------------------------------------------------------------
# Explain / verify logic
# ---------------------------------------------------------------------------
def explain(text, indent="    "):
    """Print an explain block (command + parsing hint) when --explain is active."""
    if not EXPLAIN:
        return
    for line in text.strip().split("\n"):
        print(f"{indent}# {line}", flush=True)


_verify_counts = {"passed": 0, "warned": 0, "failed": 0}


def _verify_cmd_inline(cmd, hint, indent):
    """Execute a command and verify output inline. Called by explain_cmd when -x is active."""
    hint_lower = (hint or "").lower()

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"{indent}  [FAIL: timed out after 30s]", flush=True)
        if hint:
            print(f"{indent}  expected: {hint}", flush=True)
        _verify_counts["failed"] += 1
        return
    except Exception as exc:
        print(f"{indent}  [FAIL: {exc}]", flush=True)
        if hint:
            print(f"{indent}  expected: {hint}", flush=True)
        _verify_counts["failed"] += 1
        return

    stdout = _strip_ansi(result.stdout.strip())

    # grep returns 1 when no matches; kubectl may add benign stderr
    stderr_clean = "\n".join(
        l for l in (result.stderr or "").strip().split("\n")
        if l.strip() and not l.strip().startswith("Defaulted container")
        and "command terminated with exit code" not in l
    ).strip()
    grep_no_match = (result.returncode == 1
                     and ("grep" in cmd or "wc" in cmd)
                     and not stderr_clean)
    absence_ok = any(w in hint_lower for w in ["absence", "unset", "not set",
                                                 "if =1", "empty ="])

    # Tool not available: command has 2>/dev/null, exit 1, no output
    tool_not_found = (result.returncode != 0 and "2>/dev/null" in cmd
                      and not stdout and not stderr_clean)

    # FAIL: non-zero exit (not a grep no-match, not a missing optional tool)
    if result.returncode != 0 and not grep_no_match and not tool_not_found:
        print(f"{indent}  [FAIL: exit code {result.returncode}]", flush=True)
        if hint:
            print(f"{indent}  expected: {hint}", flush=True)
        stderr_out = (result.stderr or "").strip()
        if stderr_out:
            for line in stderr_out.split("\n"):
                print(f"{indent}  stderr: {line}", flush=True)
        if stdout:
            for line in stdout.split("\n"):
                print(f"{indent}  stdout: {line}", flush=True)
        _verify_counts["failed"] += 1
        return

    if tool_not_found:
        print(f"{indent}  [executed — tool not available (install with -i/--install-deps)]", flush=True)
        _verify_counts["warned"] += 1
        return

    # Empty output
    if grep_no_match or not stdout:
        if absence_ok or grep_no_match:
            print(f"{indent}  [executed and output verified — no output, absence confirms finding]", flush=True)
            _verify_counts["passed"] += 1
        else:
            print(f"{indent}  [WARN: command succeeded but produced no output]", flush=True)
            if hint:
                print(f"{indent}  expected: {hint}", flush=True)
            _verify_counts["warned"] += 1
        return

    # Has output — verify hint keywords
    hint_checks = []
    stdout_lower = stdout.lower()
    if hint:
        if "count" in hint_lower:
            has_number = any(c.isdigit() for c in stdout)
            hint_checks.append(("has numeric output", has_number))
        check_terms = []
        if "active" in hint_lower:
            check_terms.append("active")
        if "nvlink" in hint_lower:
            check_terms.append(("nv", "nvlink", "nvl"))
        if "pcie" in hint_lower or "pxb" in hint_lower or "pix" in hint_lower:
            check_terms.append(("pix", "pxb", "phb", "sys", "pcie"))
        if "ethernet" in hint_lower or "roce" in hint_lower:
            check_terms.append(("ethernet", "roce", "infiniband"))
        if "socket" in hint_lower or "numa" in hint_lower:
            check_terms.append(("socket", "numa", "node"))
        if "gpu" in hint_lower:
            check_terms.append(("gpu", "nvidia"))
        if "nccl" in hint_lower:
            check_terms.append(("nccl",))
        for term in check_terms:
            if isinstance(term, tuple):
                found = any(t in stdout_lower for t in term)
                hint_checks.append((f"contains {'/'.join(term)}", found))
            else:
                found = term in stdout_lower
                hint_checks.append((f"contains '{term}'", found))

    all_checks_ok = all(ok for _, ok in hint_checks) if hint_checks else True

    if all_checks_ok:
        print(f"{indent}  [executed and output verified]", flush=True)
        if VERBOSE:
            for line in stdout.split("\n"):
                display = line if VERBOSE or len(line) <= 120 else line[:117] + "..."
                print(f"{indent}  | {display}", flush=True)
        _verify_counts["passed"] += 1
    else:
        check_strs = [f"{'OK' if ok else 'MISS'}: {desc}" for desc, ok in hint_checks]
        print(f"{indent}  [WARN: {'; '.join(check_strs)}]", flush=True)
        if hint:
            print(f"{indent}  expected: {hint}", flush=True)
        for line in stdout.split("\n"):
            display = line if len(line) <= 120 else line[:117] + "..."
            print(f"{indent}  | {display}", flush=True)
        _verify_counts["warned"] += 1


def _print_verify_summary():
    """Print the verification summary totals."""
    p = _verify_counts["passed"]
    w = _verify_counts["warned"]
    f = _verify_counts["failed"]
    total = p + w + f
    if total == 0:
        return
    print(f"\n{'─' * 50}")
    print(f"  Summary of running explanation commands: {p} passed, {w} warned, {f} failed "
          f"(out of {total} commands)")
    print(f"{'─' * 50}")
    if f > 0:
        print(f"  NOTE: FAIL means the command could not run or returned an error.")
        print(f"  This may indicate missing tools (install with -i/--install-deps)")
        print(f"  or permission issues in the pod.")
    if w > 0:
        print(f"  NOTE: WARN means the command ran but output was empty or didn't")
        print(f"  contain expected keywords — results may still be valid.")


def explain_cmd(cmd, parse_hint=None, indent="    "):
    """Print a reproducible command when --explain is active.

    With --explain-verify (-x), also executes the command inline and
    verifies the output.
    """
    if not EXPLAIN:
        return
    print(f"{indent}$ {cmd}", flush=True)
    if parse_hint:
        print(f"{indent}  # {parse_hint}", flush=True)
    if EXPLAIN_VERIFY:
        _verify_cmd_inline(cmd, parse_hint, indent)


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------
def run_cmd(cmd, capture=True, timeout=120):
    if VERBOSE:
        print(f"  $ {' '.join(cmd)}", flush=True)
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    else:
        return subprocess.run(cmd, timeout=timeout)


def exec_in_pod(pod_name, cmd_args, timeout=300, use_debug=None):
    """Execute a command in the pod (or on a remote host via SSH).

    use_debug: None = use global USE_DEBUG_CONTAINER setting,
               True/False = override for this call.
    """
    if SSH_MODE:
        target = _ssh_target_map.get(pod_name, pod_name)
        remote_cmd = shlex.join(cmd_args)
        cmd = list(SSH_COMMAND) + [target, remote_cmd]
        return run_cmd(cmd, timeout=timeout)
    debug = USE_DEBUG_CONTAINER if use_debug is None else use_debug
    ns = _kubectl_ns_args()
    if debug:
        container_name = pod_name + DEBUG_SUFFIX
        cmd = ["kubectl", "exec"] + ns + [pod_name, f"--container={container_name}", "--"] + cmd_args
    else:
        cmd = ["kubectl", "exec"] + ns + [pod_name, "--"] + cmd_args
    return run_cmd(cmd, timeout=timeout)


def _build_remote_cmd(pod_name, cmd_args, use_debug=None):
    """Build the full subprocess command list for remote execution.

    Used by callers that need subprocess.Popen (background servers).
    """
    if SSH_MODE:
        target = _ssh_target_map.get(pod_name, pod_name)
        return list(SSH_COMMAND) + [target, shlex.join(cmd_args)]
    debug = USE_DEBUG_CONTAINER if use_debug is None else use_debug
    ns = _kubectl_ns_args()
    if debug:
        container_name = pod_name + DEBUG_SUFFIX
        return ["kubectl", "exec"] + ns + [pod_name, f"--container={container_name}", "--"] + cmd_args
    return ["kubectl", "exec"] + ns + [pod_name, "--"] + cmd_args


# ---------------------------------------------------------------------------
# Pod discovery
# ---------------------------------------------------------------------------
def discover_pods(label):
    """Get pod names and IPs for a given label selector (or SSH hosts list)."""
    if SSH_MODE:
        if not SSH_HOSTS:
            print("No SSH hosts configured", file=sys.stderr)
            sys.exit(1)
        return list(SSH_HOSTS)
    cmd = ["kubectl", "get", "pods"] + _kubectl_ns_args() + ["-l", label, "-o", "json"]
    if VERBOSE:
        print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error discovering pods with label {label}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(result.stdout)
    pods = []
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        ip = item.get("status", {}).get("podIP")
        if ip:
            pods.append((name, ip))
    if not pods:
        print(f"No pods found with label {label}", file=sys.stderr)
        sys.exit(1)
    pods.sort(key=lambda x: x[0])
    return pods


# ---------------------------------------------------------------------------
# Pod role detection and cross-role pair filtering
# ---------------------------------------------------------------------------
_ROLE_LABEL = "llm-d.ai/role"

_pod_roles = {}   # pod_name -> "prefill" | "decode" | None


def discover_pod_roles(label):
    """Fetch the llm-d.ai/role label for all pods matching *label*.

    Populates the global _pod_roles dict.  Returns a dict
    {pod_name: role_str_or_None}.
    In SSH mode returns empty dict (no Kubernetes roles).
    """
    global _pod_roles
    if SSH_MODE:
        return _pod_roles
    cmd = ["kubectl", "get", "pods"] + _kubectl_ns_args() + [
        "-l", label, "-o",
        "jsonpath={range .items[*]}{.metadata.name}={.metadata.labels.llm-d\\.ai/role} {end}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return _pod_roles  # best-effort
    for token in result.stdout.strip().split():
        if "=" in token:
            name, role = token.split("=", 1)
            _pod_roles[name] = role.strip() if role.strip() else None
    return _pod_roles


def get_pod_role(pod_name):
    """Return the cached role for *pod_name*, or None."""
    return _pod_roles.get(pod_name)


def _cross_role_pairs(pods):
    """Return list of (i, j) index pairs restricted to cross-role testing.

    If both prefill and decode pods exist, only pairs where one is prefill
    and the other is decode are returned (in both directions).
    If all pods share the same role (or roles are unknown), returns all
    n*(n-1) pairs so no tests are skipped.
    """
    n = len(pods)
    roles = [_pod_roles.get(pods[i][0]) for i in range(n)]
    role_set = {r for r in roles if r}

    # Need both prefill and decode present to filter
    if not ({"prefill", "decode"} <= role_set):
        return [(i, j) for i in range(n) for j in range(n) if i != j]

    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ri, rj = roles[i], roles[j]
            if ri and rj and ri != rj:
                pairs.append((i, j))
    return pairs


def make_display_names(pod_names, max_len=None):
    """Create short, unique display names from a list of pod names."""
    if max_len is None:
        max_len = DISPLAY_NAME_MAX_LEN
    if not pod_names:
        return {}
    if len(pod_names) == 1:
        n = pod_names[0]
        return {n: n[:max_len] if len(n) > max_len else n}

    prefix_len = 0
    for chars in zip(*pod_names):
        if len(set(chars)) == 1:
            prefix_len += 1
        else:
            break

    reversed_names = [n[::-1] for n in pod_names]
    suffix_len = 0
    for chars in zip(*reversed_names):
        if len(set(chars)) == 1:
            suffix_len += 1
        else:
            break

    tails = []
    for n in pod_names:
        end = len(n) - suffix_len if suffix_len > 0 else len(n)
        tails.append(n[prefix_len:end])

    shared_prefix = pod_names[0][:prefix_len]
    result = {}
    for full, tail in zip(pod_names, tails):
        if not tail:
            tail = full[-max_len:]
        if len(tail) >= max_len:
            tail_keep = max(4, max_len // 2)
            head_keep = max_len - tail_keep - 3
            if head_keep < 1:
                head_keep = 1
                tail_keep = max_len - 4
            display = tail[:head_keep] + "..." + tail[-tail_keep:]
        elif len(tail) < max_len and prefix_len > 0:
            budget = max_len - len(tail)
            if budget >= prefix_len + 3:
                display = shared_prefix[:budget] + tail
            elif budget >= 4:
                keep = budget - 3
                display = shared_prefix[:keep] + "..." + tail
            else:
                display = tail
        else:
            display = tail
        result[full] = display

    seen = {}
    for full in pod_names:
        d = result[full]
        if d in seen:
            seen[d] += 1
            suffix = str(seen[d])
            result[full] = d[:max_len - len(suffix)] + suffix
        else:
            seen[d] = 1
    return result


# ---------------------------------------------------------------------------
# Debug containers
# ---------------------------------------------------------------------------
def debug_container_running(pod_name, container_name):
    cmd = [
        "kubectl", "get", "pod",
    ] + _kubectl_ns_args() + [
        pod_name,
        "-o", f"jsonpath={{.status.ephemeralContainerStatuses[?(@.name==\"{container_name}\")].state.running}}",
    ]
    if VERBOSE:
        print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def create_debug_containers(pods):
    """Create ephemeral debug containers on all pods if needed."""
    needed = []
    for name, _ip in pods:
        container_name = name + DEBUG_SUFFIX
        if debug_container_running(name, container_name):
            print(f"  Debug container '{container_name}' already running in {name}, reusing.", flush=True)
        else:
            needed.append((name, container_name))

    for name, container_name in needed:
        print(f"  Creating debug container '{container_name}' in {name} ...", flush=True)
        cmd = [
            "kubectl", "debug",
        ] + _kubectl_ns_args() + [
            "-i", name,
            f"--image={DEBUG_IMAGE}",
            f"--container={container_name}",
            "--profile=netadmin",
            "--", "sleep", "inf",
        ]
        if VERBOSE:
            print(f"  $ {' '.join(cmd)}", flush=True)
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if needed:
        print("  Waiting for debug containers to be ready ...", flush=True)
        for _ in range(60):
            all_ready = all(
                debug_container_running(name, cname) for name, cname in needed
            )
            if all_ready:
                print("  All debug containers ready.")
                return
            time.sleep(1)
        print("  Warning: timed out waiting for some debug containers.", file=sys.stderr)
    else:
        print("  All debug containers already running.")
