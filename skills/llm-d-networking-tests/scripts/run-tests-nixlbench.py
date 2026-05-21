# /// script
# requires-python = ">=3.10"
# ///
"""NIXLBench runner for Kubernetes inference pods.

Runs nixlbench (NIXL data transfer benchmark) between pods using ETCD for
worker coordination.  Measures RDMA/GPU memory transfer throughput and latency
between pod pairs.

With --install-deps, builds nixlbench from source inside pods (requires CUDA
and UCX to be present in the container image).  Otherwise expects nixlbench
to be pre-installed.

Public API:
    run_nixlbench(pods, display_names) -> dict
    ensure_nixlbench(pod_name) -> None
    install_etcd(pod_name) -> None
    build_nixlbench_from_source(pod_name) -> bool
"""

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


# ---------------------------------------------------------------------------
# NIXLBench constants
# ---------------------------------------------------------------------------
ETCD_VER = "v3.5.21"
ETCD_DOWNLOAD_URL = (
    f"https://github.com/etcd-io/etcd/releases/download/{ETCD_VER}"
    f"/etcd-{ETCD_VER}-linux-amd64.tar.gz"
)
ETCD_PORT = 2379
NIXLBENCH_BINARY = "nixlbench"
NIXLBENCH_TIMEOUT = 600  # 10 minutes per pair
NIXL_REPO = "https://github.com/ai-dynamo/nixl.git"
NIXL_BUILD_DIR = "/tmp/nixl-build"
NIXL_INSTALL_PREFIX = "/usr/local/nixl"
NIXLBENCH_INSTALL_PREFIX = "/usr/local/nixlbench"
ETCD_CPP_REPO = "https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git"
PROTOBUF_VER = "v21.12"
GRPC_VER = "v1.46.7"

# Non-root fallback paths (used when /usr/local is not writable)
NONROOT_PREFIX = "/tmp/local"
NONROOT_BIN = "/tmp/.local/bin"
NONROOT_NIXL_PREFIX = "/tmp/local/nixl"
NONROOT_NIXLBENCH_PREFIX = "/tmp/nixlbench"
NIXL_PIP_LIBS = "/usr/local/lib/python3.12/dist-packages/.nixl_cu12.mesonpy.libs"
NIXL_PIP_UCX_LIBS = "/usr/local/lib/python3.12/dist-packages/nixl_cu12.libs"

# Source builds for non-root pods (gdrcopy + UCX from source)
GDRCOPY_VER = "2.5"
GDRCOPY_URL = f"https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v{GDRCOPY_VER}.tar.gz"
NONROOT_GDR_HOME = f"{NONROOT_PREFIX}/gdrcopy"
UCX_VER = "1.18.0"
UCX_SRC_URL = f"https://github.com/openucx/ucx/releases/download/v{UCX_VER}/ucx-{UCX_VER}.tar.gz"
NONROOT_UCX_HOME = f"{NONROOT_PREFIX}/ucx"

# Defaults — overridden via configure() from run-tests.py
NIXLBENCH_BACKEND = "UCX"
NIXLBENCH_SEG_TYPE = "VRAM"
NIXLBENCH_BUFFER_SIZE = "8G"


# ---------------------------------------------------------------------------
# Non-root environment detection and helpers
# ---------------------------------------------------------------------------
_nonroot_cache = {}  # pod_name -> bool


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


def _nonroot_env_prefix(pod_name):
    """Build shell env prefix for non-root pods (HOME=/tmp, /tmp paths)."""
    if not _detect_nonroot(pod_name):
        return ""
    return (
        f'export HOME=/tmp && '
        f'export PATH={NONROOT_BIN}:{NONROOT_NIXLBENCH_PREFIX}/bin:'
        f'{NONROOT_NIXL_PREFIX}/bin:{NONROOT_PREFIX}/bin:$PATH && '
        f'export LD_LIBRARY_PATH={NONROOT_NIXLBENCH_PREFIX}/lib:'
        f'{NONROOT_NIXLBENCH_PREFIX}/lib64:'
        f'{NONROOT_NIXL_PREFIX}/lib64:{NONROOT_NIXL_PREFIX}/lib64/plugins:'
        f'{NONROOT_NIXL_PREFIX}/lib/$(uname -m)-linux-gnu:'
        f'{NONROOT_NIXL_PREFIX}/lib/$(uname -m)-linux-gnu/plugins:'
        f'{NONROOT_UCX_HOME}/lib:{NONROOT_GDR_HOME}/lib:'
        f'{NONROOT_PREFIX}/lib64:{NONROOT_PREFIX}/lib:'
        f'{NIXL_PIP_LIBS}:{NIXL_PIP_UCX_LIBS}:'
        f'/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}} && '
        f'export PKG_CONFIG_PATH={NONROOT_PREFIX}/lib64/pkgconfig:'
        f'{NONROOT_PREFIX}/lib/pkgconfig:${{PKG_CONFIG_PATH:-}} && '
    )


def _get_install_prefix(pod_name):
    """Return (nixl_prefix, nixlbench_prefix, local_prefix) based on root detection."""
    if _detect_nonroot(pod_name):
        return NONROOT_NIXL_PREFIX, NONROOT_NIXLBENCH_PREFIX, NONROOT_PREFIX
    return NIXL_INSTALL_PREFIX, NIXLBENCH_INSTALL_PREFIX, "/usr/local"


def _get_bin_dir(pod_name):
    """Return the bin directory for installing standalone tools (etcd, git)."""
    if _detect_nonroot(pod_name):
        return NONROOT_BIN
    return "/usr/local/bin"


# ---------------------------------------------------------------------------
# Binary checks
# ---------------------------------------------------------------------------
def _check_binary(pod_name, binary):
    """Check if a binary is available on a pod. Returns True if found."""
    nixl_pfx, bench_pfx, _local_pfx = _get_install_prefix(pod_name)
    nonroot_pfx = _nonroot_env_prefix(pod_name)
    env_prefix = (
        f'{nonroot_pfx}'
        f'export PATH={bench_pfx}/bin:{nixl_pfx}/bin:'
        f'{NONROOT_BIN}:/usr/local/bin:$PATH && '
        f'export LD_LIBRARY_PATH={bench_pfx}/lib:'
        f'{bench_pfx}/lib64:'
        f'{nixl_pfx}/lib64:{nixl_pfx}/lib64/plugins:'
        f'{nixl_pfx}/lib/$(uname -m)-linux-gnu:'
        f'{nixl_pfx}/lib/$(uname -m)-linux-gnu/plugins:'
        f'{NIXL_PIP_LIBS}:{NIXL_PIP_UCX_LIBS}:'
        f'/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}'
    )
    if binary == NIXLBENCH_BINARY:
        # For nixlbench: verify both found AND shared libs resolve
        check_cmd = (
            f'{env_prefix} && '
            f'if ! command -v {binary} >/dev/null 2>&1; then echo "MISSING"; '
            f'elif ldd $(command -v {binary}) 2>&1 | grep -q "not found"; then echo "MISSING_LIBS"; '
            f'else echo "FOUND"; fi'
        )
    else:
        check_cmd = (
            f'{env_prefix} && '
            f'command -v {binary} >/dev/null 2>&1 && echo "FOUND" || echo "MISSING"'
        )
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", check_cmd],
        use_debug=False,
    )
    if binary == NIXLBENCH_BINARY:
        return result.returncode == 0 and "FOUND" in result.stdout and "MISSING_LIBS" not in result.stdout
    return result.returncode == 0 and "FOUND" in result.stdout


def _check_etcd_runtime(pod_name):
    """Check if nixlbench was built with ETCD runtime support."""
    nixl_pfx, bench_pfx, _local_pfx = _get_install_prefix(pod_name)
    nonroot_pfx = _nonroot_env_prefix(pod_name)
    env_prefix = (
        f'{nonroot_pfx}'
        f'export PATH={bench_pfx}/bin:{nixl_pfx}/bin:'
        f'{NONROOT_BIN}:/usr/local/bin:$PATH && '
        f'export LD_LIBRARY_PATH={bench_pfx}/lib:'
        f'{bench_pfx}/lib64:'
        f'{nixl_pfx}/lib64:{nixl_pfx}/lib64/plugins:'
        f'{NIXL_PIP_LIBS}:{NIXL_PIP_UCX_LIBS}:'
        f'/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}'
    )
    # Run nixlbench with --etcd_endpoints dummy to check if ETCD runtime is valid
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f'{env_prefix} && nixlbench --etcd_endpoints http://127.0.0.1:1 '
         f'--benchmark_group test 2>&1 | head -5'],
        timeout=10, use_debug=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    # "Invalid runtime: ETCD" means ETCD runtime was not compiled in
    return "Invalid runtime" not in output


def ensure_nixlbench(pod_name, out=None):
    """Verify nixlbench binary is available; build from source if --install-deps."""
    _out = out or print
    if _check_binary(pod_name, NIXLBENCH_BINARY):
        if _check_etcd_in_binary(pod_name):
            _out(f"  nixlbench binary available on {pod_name} (with ETCD runtime).")
        else:
            _out(f"  nixlbench binary available on {pod_name} (ASIO runtime, no ETCD).")
        return
    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: nixlbench binary not found on {pod_name}.")
        _out(f"  Re-run with --install-deps (-i) to build from source,")
        _out(f"  or use a pod image with nixlbench pre-installed.")
        _out(f"  See https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench")
        sys.exit(1)
    _out(f"  nixlbench not found on {pod_name}, building from source ...")
    if not build_nixlbench_from_source(pod_name, out=_out):
        _out(f"  Error: failed to build nixlbench on {pod_name}.")
        sys.exit(1)
    if not _check_binary(pod_name, NIXLBENCH_BINARY):
        _out(f"  Error: nixlbench still not available after build on {pod_name}.")
        sys.exit(1)
    _out(f"  nixlbench built and installed successfully on {pod_name}.")


# ---------------------------------------------------------------------------
# Build from source
# ---------------------------------------------------------------------------
def _run_build_step(pod_name, cmd, label, out, timeout=600, ignore_error=False):
    """Run a build step inside the pod. Returns True on success."""
    _out = out or print
    _out(f"    [{label}] ...")
    nonroot_pfx = _nonroot_env_prefix(pod_name)
    _nixl_pfx, _bench_pfx, local_pfx = _get_install_prefix(pod_name)
    cmd_with_path = (
        f'{nonroot_pfx}'
        f'export PATH={NONROOT_BIN}:/usr/local/bin:$PATH && '
        f'export PKG_CONFIG_PATH={local_pfx}/lib64/pkgconfig:{local_pfx}/lib/pkgconfig:${{PKG_CONFIG_PATH:-}} && '
        f'{cmd}'
    )
    result = exec_in_pod(pod_name, ["bash", "-c", cmd_with_path], timeout=timeout, use_debug=False)
    if _get_common().VERBOSE and result.stdout.strip():
        _out(result.stdout[-500:])
    if result.returncode != 0:
        if ignore_error:
            _out(f"    [{label}] warning (exit {result.returncode}, ignored)")
            return True
        _out(f"    [{label}] FAILED (exit {result.returncode})")
        stderr = result.stderr.strip()
        if stderr:
            _out(f"    stderr: {stderr[-500:]}")
        return False
    return True


def _install_deps_nonroot(pod_name, out=None):
    """Install build dependencies for non-root pods using pip and curl."""
    _out = out or print
    _out(f"  Detected non-root environment on {pod_name}")
    _out(f"  Installing build tools via pip to /tmp/.local/bin ...")

    # Python build tools via HOME=/tmp pip install --user
    pip_pkgs = "meson pybind11 tomlkit cmake pkgconf"
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f"export HOME=/tmp && "
         f"pip install --user {pip_pkgs} 2>&1 || "
         f"pip3 install --user {pip_pkgs} 2>&1 || "
         f"python3 -m pip install --user {pip_pkgs} 2>&1"],
        timeout=180, use_debug=False,
    )
    if result.returncode != 0:
        _out(f"    Warning: pip install failed: {(result.stderr or result.stdout or '').strip()[:300]}")
    else:
        _out(f"    Installed Python build tools (meson, cmake, pybind11).")

    # Verify pkg-config is available (installed via pkgconf above)
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f"export HOME=/tmp && export PATH={NONROOT_BIN}:$PATH && "
         f"command -v pkg-config >/dev/null 2>&1 && echo OK || echo MISSING"],
        timeout=30, use_debug=False,
    )
    if "OK" in (result.stdout or ""):
        _out(f"    pkg-config available at {NONROOT_BIN}/pkg-config")
    else:
        _out(f"    Warning: pkg-config not found on PATH")

    # Check for git — download static binary if missing
    git_check = exec_in_pod(
        pod_name,
        ["bash", "-c", f"export PATH={NONROOT_BIN}:$PATH && command -v git"],
        use_debug=False,
    )
    if git_check.returncode != 0:
        _out(f"    git not found; will use curl for tarball downloads.")

    # Ensure /tmp/.local/bin and build dirs exist
    exec_in_pod(
        pod_name,
        ["bash", "-c", f"mkdir -p {NONROOT_BIN} {NONROOT_PREFIX} {NIXL_BUILD_DIR}"],
        use_debug=False,
    )

    # Verify essential tools
    verify_cmd = (
        f"export HOME=/tmp && export PATH={NONROOT_BIN}:/usr/local/bin:$PATH && "
        f"echo 'meson:' $(meson --version 2>/dev/null || echo MISSING) && "
        f"echo 'cmake:' $(cmake --version 2>/dev/null | head -1 || echo MISSING) && "
        f"echo 'ninja:' $(ninja --version 2>/dev/null || echo MISSING) && "
        f"echo 'gcc:' $(gcc --version 2>/dev/null | head -1 || echo MISSING)"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", verify_cmd], use_debug=False)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            _out(f"    {line}")

    _out(f"  Build dependencies ready on {pod_name}.")
    return True


def _build_gdrcopy_nonroot(pod_name, out=None):
    """Build gdrcopy from source for non-root pods (userspace lib only, skips insmod)."""
    _out = out or print
    # Check if already built
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", f"test -f {NONROOT_GDR_HOME}/lib/libgdrapi.so && echo FOUND"],
        use_debug=False,
    )
    if result.returncode == 0 and "FOUND" in (result.stdout or ""):
        _out(f"    gdrcopy already installed at {NONROOT_GDR_HOME}")
        return True

    _out(f"  Building gdrcopy v{GDRCOPY_VER} on {pod_name} ...")
    build_cmd = (
        f"export HOME=/tmp && export PATH={NONROOT_BIN}:$PATH && "
        f"mkdir -p /tmp/nixl_installer && cd /tmp/nixl_installer && "
        f"curl -sL {GDRCOPY_URL} -o gdrcopy.tar.gz && "
        f"tar xzf gdrcopy.tar.gz && rm gdrcopy.tar.gz && "
        f"cd gdrcopy-{GDRCOPY_VER} && "
        f"make prefix={NONROOT_GDR_HOME} CUDA=/usr/local/cuda all install 2>&1 | tail -5 && "
        f"test -f {NONROOT_GDR_HOME}/lib/libgdrapi.so && echo GDRCOPY_OK"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", build_cmd], timeout=300, use_debug=False)
    if "GDRCOPY_OK" not in (result.stdout or ""):
        _out(f"    Warning: gdrcopy build failed (non-fatal, UCX will build without GDR)")
        if _get_common().VERBOSE and result.stderr:
            _out(f"    stderr: {result.stderr.strip()[-300:]}")
        return False
    _out(f"    gdrcopy installed at {NONROOT_GDR_HOME} (skipped insmod — host kernel module)")
    return True


def _build_ucx_from_source(pod_name, out=None):
    """Build UCX from source for non-root pods when no existing UCX is available."""
    _out = out or print
    # Check if already built
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", f"test -f {NONROOT_UCX_HOME}/lib/libucp.so && echo FOUND"],
        use_debug=False,
    )
    if result.returncode == 0 and "FOUND" in (result.stdout or ""):
        _out(f"    UCX already installed at {NONROOT_UCX_HOME}")
        return NONROOT_UCX_HOME

    _out(f"  Building UCX v{UCX_VER} from source on {pod_name} ...")

    # Detect Mellanox/IB for configure flags
    mlx_check = exec_in_pod(
        pod_name,
        ["bash", "-c",
         "(command -v lspci >/dev/null 2>&1 && lspci | grep -qi mellanox && echo MLX) || "
         "(command -v ibstat >/dev/null 2>&1 && echo MLX) || echo NONE"],
        use_debug=False,
    )
    mlx_opts = ""
    if "MLX" in (mlx_check.stdout or ""):
        mlx_opts = "--with-rdmacm --with-mlx5-dv --with-ib-hw-tm"
        _out(f"    Mellanox/IB detected, enabling RDMA options")

    # gdrcopy flag
    gdr_opt = ""
    gdr_check = exec_in_pod(
        pod_name,
        ["bash", "-c", f"test -f {NONROOT_GDR_HOME}/lib/libgdrapi.so && echo FOUND"],
        use_debug=False,
    )
    if "FOUND" in (gdr_check.stdout or ""):
        gdr_opt = f"--with-gdrcopy={NONROOT_GDR_HOME}"

    build_cmd = (
        f"export HOME=/tmp && export PATH={NONROOT_BIN}:$PATH && "
        f"export LD_LIBRARY_PATH={NONROOT_GDR_HOME}/lib:${{LD_LIBRARY_PATH:-}} && "
        f"mkdir -p /tmp/nixl_installer && cd /tmp/nixl_installer && "
        f"curl -sL {UCX_SRC_URL} -o ucx.tar.gz && "
        f"tar xzf ucx.tar.gz && rm ucx.tar.gz && "
        f"cd ucx-{UCX_VER} && "
        f"./configure --prefix={NONROOT_UCX_HOME} "
        f"--enable-shared --disable-static --disable-doxygen-doc "
        f"--enable-optimizations --enable-cma --enable-devel-headers "
        f"--with-cuda=/usr/local/cuda --with-dm --with-verbs --enable-mt "
        f"{gdr_opt} {mlx_opts} 2>&1 | tail -10 && "
        f"make -j$(nproc) 2>&1 | tail -5 && "
        f"make install-strip 2>&1 | tail -5 && "
        f"test -f {NONROOT_UCX_HOME}/lib/libucp.so && echo UCX_OK"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", build_cmd], timeout=600, use_debug=False)
    if "UCX_OK" not in (result.stdout or ""):
        _out(f"    UCX build failed")
        if _get_common().VERBOSE and result.stderr:
            _out(f"    stderr: {result.stderr.strip()[-300:]}")
        return None
    _out(f"    UCX installed at {NONROOT_UCX_HOME} (using LD_LIBRARY_PATH)")
    return NONROOT_UCX_HOME


def install_nixlbench_deps(pod_name, out=None):
    """Install system packages and Python build tools needed for nixlbench."""
    _out = out or print
    _out(f"  Installing build dependencies on {pod_name} ...")

    # Non-root path: skip apt-get/dnf entirely, use pip + curl
    if _detect_nonroot(pod_name):
        return _install_deps_nonroot(pod_name, out=_out)

    # Package groups by distro family — installed one group at a time so a
    # failure in one doesn't block the rest.
    rpm_groups = [
        ("gcc gcc-c++ ninja-build cmake pkgconfig git python3-pip",
         "core build tools"),
        ("rdma-core-devel",
         "RDMA core development headers"),
        ("gflags-devel",
         "gflags (required by nixlbench)"),
        ("openssl-devel",
         "OpenSSL development headers (for etcd-cpp-apiv3)"),
        ("re2-devel zlib-devel",
         "re2 and zlib development headers (for gRPC)"),
    ]
    apt_groups = [
        ("build-essential ninja-build cmake pkg-config git python3-pip",
         "core build tools"),
        ("libibverbs-dev librdmacm-dev rdma-core ibverbs-utils "
         "libibumad-dev ibverbs-providers libnuma-dev",
         "RDMA/IB development headers"),
        ("libgflags-dev libaio-dev liburing-dev libz-dev pybind11-dev",
         "I/O and misc libraries"),
        ("libssl-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc "
         "libgrpc++-dev libcpprest-dev",
         "protobuf/gRPC/OpenSSL (for etcd support)"),
    ]

    # Detect package manager
    for pkg_mgr, install_cmd in [
        ("dnf", "dnf install -y"),
        ("yum", "yum install -y"),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            _out(f"    Detected {pkg_mgr} package manager")
            for pkgs, label in rpm_groups:
                result = exec_in_pod(
                    pod_name, ["bash", "-c", f"{install_cmd} {pkgs}"],
                    timeout=300, use_debug=False,
                )
                if result.returncode != 0:
                    _out(f"    Warning: failed to install {label}: {result.stderr.strip()[:200]}")
                else:
                    stdout = (result.stdout or "").lower()
                    if "already installed" in stdout or "nothing to do" in stdout:
                        _out(f"    Skipped {label} (already installed).")
                    else:
                        _out(f"    Installed {label}.")
            break
    else:
        # Debian/Ubuntu — apt-get
        exec_in_pod(pod_name, ["apt-get", "update", "-y"], timeout=120, use_debug=False)
        for pkgs, label in apt_groups:
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", f"DEBIAN_FRONTEND=noninteractive apt-get install -y {pkgs}"],
                timeout=300, use_debug=False,
            )
            if result.returncode != 0:
                _out(f"    Warning: failed to install {label}: {result.stderr.strip()[:200]}")
            else:
                stdout = (result.stdout or "").lower()
                if "already the newest" in stdout:
                    _out(f"    Skipped {label} (already installed).")
                else:
                    _out(f"    Installed {label}.")

    # Python build tools — try pip3 first, then python3 -m pip
    _out(f"    Installing Python build tools (meson, pybind11, tomlkit) ...")
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         "pip3 install meson pybind11 tomlkit 2>/dev/null || "
         "python3 -m pip install meson pybind11 tomlkit"],
        timeout=120, use_debug=False,
    )
    if result.returncode != 0:
        _out(f"    Warning: pip install failed: {result.stderr.strip()[:200]}")
    else:
        _out(f"    Installed Python build tools.")

    _out(f"  Build dependencies installed on {pod_name}.")
    return True


def _check_lib_installed(pod_name, pkg_config_name, lib_paths):
    """Check if a library is installed via pkg-config or file existence."""
    checks = [f"pkg-config --exists {pkg_config_name} 2>/dev/null"]
    for p in lib_paths:
        checks.append(f"test -f {p}")
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", " || ".join(checks)],
        use_debug=False,
    )
    return result.returncode == 0


def _clean_build_dir(pod_name, build_dir, out):
    """Clean a build directory, handling stubborn permissions."""
    _run_build_step(
        pod_name,
        f"chmod -R u+rwx {build_dir} 2>/dev/null; rm -rf {build_dir}",
        "clean", out, ignore_error=True,
    )


def _clone_or_download(pod_name, repo_url, branch_or_tag, dest_dir, out):
    """Clone a git repo, or fall back to curl tarball download if git is unavailable."""
    _out = out or print
    # In non-root mode we install a git stub (for meson version queries only),
    # so always use curl for actual cloning in that case.
    use_git = False
    if not _detect_nonroot(pod_name):
        git_check = exec_in_pod(
            pod_name,
            ["bash", "-c", f"export PATH={NONROOT_BIN}:$PATH && "
             f"git --version 2>/dev/null | grep -q 'git version'"],
            use_debug=False,
        )
        use_git = git_check.returncode == 0
    if use_git:
        return f"git clone --depth 1 --branch {branch_or_tag} {repo_url} {dest_dir}"
    # Fallback: download tarball via curl (try tag first, then branch)
    base_url = repo_url.replace(".git", "")
    return (
        f"mkdir -p {dest_dir} && "
        f"(curl -sfL {base_url}/archive/refs/tags/{branch_or_tag}.tar.gz -o /tmp/_dl.tar.gz "
        f"|| curl -sfL {base_url}/archive/refs/heads/{branch_or_tag}.tar.gz -o /tmp/_dl.tar.gz "
        f"|| curl -sfL {base_url}/archive/{branch_or_tag}.tar.gz -o /tmp/_dl.tar.gz) && "
        f"tar xzf /tmp/_dl.tar.gz --strip-components=1 -C {dest_dir} && "
        f"rm -f /tmp/_dl.tar.gz"
    )


def build_protobuf(pod_name, out=None):
    """Build protobuf from source (required by gRPC and etcd-cpp-apiv3)."""
    _out = out or print
    _nixl_pfx, _bench_pfx, local_pfx = _get_install_prefix(pod_name)
    nonroot = _detect_nonroot(pod_name)
    if _check_lib_installed(pod_name, "protobuf",
                            [f"{local_pfx}/lib64/libprotobuf.so",
                             f"{local_pfx}/lib/libprotobuf.so",
                             "/usr/local/lib64/libprotobuf.so",
                             "/usr/local/lib/libprotobuf.so"]):
        _out(f"    protobuf already installed.")
        return True
    _out(f"  Building protobuf on {pod_name} ...")
    build_dir = f"{NIXL_BUILD_DIR}/protobuf"
    _clean_build_dir(pod_name, build_dir, _out)
    clone_cmd = _clone_or_download(
        pod_name, "https://github.com/protocolbuffers/protobuf.git",
        PROTOBUF_VER, build_dir, _out)
    install_cmd = f"cd {build_dir}/build && make install"
    if not nonroot:
        install_cmd += " && ldconfig"
    steps = [
        (clone_cmd, "clone protobuf", 120),
        (f"cd {build_dir} && git submodule update --init --recursive 2>/dev/null || true",
         "init submodules", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX={local_pfx} "
         f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON "
         f"-Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 600),
        (install_cmd, "install", 120),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: protobuf build failed at '{label}'.")
            return False
    _out(f"  protobuf built and installed on {pod_name}.")
    return True


def build_grpc(pod_name, out=None):
    """Build gRPC from source with bundled abseil (required by etcd-cpp-apiv3)."""
    _out = out or print
    _nixl_pfx, _bench_pfx, local_pfx = _get_install_prefix(pod_name)
    nonroot = _detect_nonroot(pod_name)
    if _check_lib_installed(pod_name, "grpc++",
                            [f"{local_pfx}/lib/libgrpc++.so",
                             f"{local_pfx}/lib64/libgrpc++.so",
                             "/usr/local/lib/libgrpc++.so",
                             "/usr/local/lib64/libgrpc++.so"]):
        _out(f"    gRPC already installed.")
        return True
    _out(f"  Building gRPC on {pod_name} (this may take several minutes) ...")
    build_dir = f"{NIXL_BUILD_DIR}/grpc"
    _clean_build_dir(pod_name, build_dir, _out)
    clone_cmd = _clone_or_download(
        pod_name, "https://github.com/grpc/grpc.git",
        GRPC_VER, build_dir, _out)
    install_cmd = f"cd {build_dir}/build && cmake --install ."
    if not nonroot:
        install_cmd += " && ldconfig"
    steps = [
        (clone_cmd, "clone gRPC", 120),
        (f"cd {build_dir} && git submodule update --init "
         f"third_party/abseil-cpp third_party/cares/cares 2>/dev/null || true",
         "init submodules", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX={local_pfx} "
         f"-DBUILD_SHARED_LIBS=ON -DgRPC_INSTALL=ON "
         f"-DgRPC_BUILD_TESTS=OFF "
         f"-DgRPC_PROTOBUF_PROVIDER=package "
         f"-DgRPC_ABSL_PROVIDER=module "
         f"-DgRPC_CARES_PROVIDER=module "
         f"-DgRPC_RE2_PROVIDER=package "
         f"-DgRPC_SSL_PROVIDER=package "
         f"-DgRPC_ZLIB_PROVIDER=package "
         f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 900),
        (install_cmd, "install", 120),
        (f"rm -f {local_pfx}/lib/pkgconfig/absl_*.pc "
         f"{local_pfx}/lib64/pkgconfig/absl_*.pc 2>/dev/null; "
         f"dnf remove -y abseil-cpp abseil-cpp-devel 2>/dev/null; true",
         "clean abseil pkg-config", 60),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: gRPC build failed at '{label}'.")
            return False
    _out(f"  gRPC built and installed on {pod_name}.")
    return True


def build_etcd_cpp_api(pod_name, out=None):
    """Build etcd-cpp-apiv3 from source (required for nixlbench etcd runtime).

    Requires protobuf and gRPC to be installed first.  Builds them from source
    if not already present (RHEL 9 UBI repos lack these C++ packages).
    """
    _out = out or print
    _nixl_pfx, _bench_pfx, local_pfx = _get_install_prefix(pod_name)
    nonroot = _detect_nonroot(pod_name)

    # Check if already installed
    if _check_lib_installed(pod_name, "etcd-cpp-api",
                            [f"{local_pfx}/lib64/libetcd-cpp-api-core.so",
                             f"{local_pfx}/lib/libetcd-cpp-api-core.so",
                             "/usr/local/lib64/libetcd-cpp-api-core.so",
                             "/usr/local/lib/libetcd-cpp-api-core.so"]):
        _out(f"    etcd-cpp-apiv3 already available.")
        return True

    _out(f"  Building etcd stack (protobuf -> gRPC -> etcd-cpp-apiv3) ...")

    # Step 1: Build protobuf from source
    if not build_protobuf(pod_name, out=_out):
        _out(f"  Cannot build etcd-cpp-apiv3 without protobuf.")
        return False

    # Step 2: Build gRPC from source
    if not build_grpc(pod_name, out=_out):
        _out(f"  Cannot build etcd-cpp-apiv3 without gRPC.")
        return False

    # Step 3: Build etcd-cpp-apiv3
    _out(f"  Building etcd-cpp-apiv3 on {pod_name} ...")
    build_dir = f"{NIXL_BUILD_DIR}/etcd-cpp-apiv3"
    _clean_build_dir(pod_name, build_dir, _out)
    clone_cmd = _clone_or_download(
        pod_name, "https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git",
        "master", build_dir, _out)
    install_cmd = f"cd {build_dir}/build && make install"
    if not nonroot:
        install_cmd += " && ldconfig"
    steps = [
        (clone_cmd, "clone etcd-cpp-apiv3", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX={local_pfx} -DBUILD_SHARED_LIBS=ON "
         f"-DBUILD_ETCD_TESTS=OFF -DBUILD_ETCD_CORE_ONLY=ON "
         f"-DCMAKE_PREFIX_PATH={local_pfx}",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 300),
        (install_cmd, "install", 120),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: etcd-cpp-apiv3 build failed at '{label}'.")
            _out(f"  nixlbench will be built without etcd runtime support.")
            return False

    # Create pkg-config file (meson needs it to find etcd-cpp-api)
    pkgconfig_cmd = (
        f'mkdir -p {local_pfx}/lib64/pkgconfig && '
        f'cat > {local_pfx}/lib64/pkgconfig/etcd-cpp-api.pc << "PKGEOF"\n'
        f'prefix={local_pfx}\n'
        'exec_prefix=${prefix}\n'
        'libdir=${prefix}/lib64\n'
        'includedir=${prefix}/include\n'
        '\n'
        'Name: etcd-cpp-api\n'
        'Description: etcd C++ client API (core only)\n'
        'Version: 0.15.4\n'
        'Libs: -L${libdir} -letcd-cpp-api-core\n'
        'Cflags: -I${includedir}\n'
        'PKGEOF'
    )
    _run_build_step(pod_name, pkgconfig_cmd, "pkg-config file", _out, ignore_error=True)

    _out(f"  etcd-cpp-apiv3 built and installed on {pod_name}.")
    return True


def _setup_ucx_prefix_from_pip(pod_name, out=None):
    """Create a proper UCX prefix from pip package libs + downloaded headers.

    The pip nixl_cu12 package bundles UCX runtime libs with versioned names.
    This function creates a standard UCX prefix layout at NONROOT_PREFIX/ucx with:
    - lib/libucp.so etc (symlinks to pip versioned libs)
    - lib/ucx/ (symlink to pip ucx plugins dir)
    - include/ucp/, include/ucs/, include/uct/ (downloaded from UCX release)
    """
    _out = out or print
    ucx_pfx = f"{NONROOT_PREFIX}/ucx"
    pip_ucx = NIXL_PIP_UCX_LIBS

    # Create lib symlinks (versioned -> unversioned)
    setup_cmd = (
        f"mkdir -p {ucx_pfx}/lib {ucx_pfx}/include && "
        f"for f in {pip_ucx}/libucp-*.so*; do ln -sf \"$f\" {ucx_pfx}/lib/libucp.so; done && "
        f"for f in {pip_ucx}/libucs-*.so*; do ln -sf \"$f\" {ucx_pfx}/lib/libucs.so; done && "
        f"for f in {pip_ucx}/libuct-*.so*; do ln -sf \"$f\" {ucx_pfx}/lib/libuct.so; done && "
        f"for f in {pip_ucx}/libucm-*.so*; do ln -sf \"$f\" {ucx_pfx}/lib/libucm.so; done && "
        f"ln -sfn {pip_ucx}/ucx {ucx_pfx}/lib/ucx && "
        f"echo LIBS_OK"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", setup_cmd], timeout=30, use_debug=False)
    if "LIBS_OK" not in (result.stdout or ""):
        _out(f"    Warning: failed to create UCX lib symlinks")
        return None

    # Download UCX headers from release tarball (just the src/ucp, src/ucs, src/uct include dirs)
    # UCX 1.18.0 matches the bundled version (check via lib version)
    headers_cmd = (
        f"if [ ! -f {ucx_pfx}/include/ucp/api/ucp.h ]; then "
        f"  curl -sfL https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz "
        f"  -o /tmp/_ucx_headers.tar.gz && "
        f"  tar xzf /tmp/_ucx_headers.tar.gz -C /tmp "
        f"  ucx-1.18.0/src/ucp/api ucx-1.18.0/src/ucs/type "
        f"  ucx-1.18.0/src/ucs/sys ucx-1.18.0/src/ucs/memory "
        f"  ucx-1.18.0/src/ucs/config ucx-1.18.0/src/uct/api 2>/dev/null && "
        f"  mkdir -p {ucx_pfx}/include/ucp/api {ucx_pfx}/include/ucs/type "
        f"  {ucx_pfx}/include/ucs/sys {ucx_pfx}/include/ucs/memory "
        f"  {ucx_pfx}/include/ucs/config {ucx_pfx}/include/uct/api && "
        f"  cp /tmp/ucx-1.18.0/src/ucp/api/*.h {ucx_pfx}/include/ucp/api/ && "
        f"  cp /tmp/ucx-1.18.0/src/ucs/type/*.h {ucx_pfx}/include/ucs/type/ && "
        f"  cp /tmp/ucx-1.18.0/src/ucs/sys/*.h {ucx_pfx}/include/ucs/sys/ && "
        f"  cp /tmp/ucx-1.18.0/src/ucs/memory/*.h {ucx_pfx}/include/ucs/memory/ && "
        f"  cp /tmp/ucx-1.18.0/src/ucs/config/*.h {ucx_pfx}/include/ucs/config/ && "
        f"  cp /tmp/ucx-1.18.0/src/uct/api/*.h {ucx_pfx}/include/uct/api/ && "
        f"  rm -rf /tmp/ucx-1.18.0 /tmp/_ucx_headers.tar.gz && "
        f"  echo HEADERS_OK; "
        f"else echo HEADERS_OK; fi"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", headers_cmd], timeout=120, use_debug=False)
    if "HEADERS_OK" not in (result.stdout or ""):
        _out(f"    Warning: failed to download UCX headers")
        return None

    _out(f"    UCX prefix set up at {ucx_pfx} (headers downloaded, libs from pip)")
    return ucx_pfx


def _detect_ucx_path(pod_name):
    """Detect UCX installation path on the pod."""
    # Check common locations including pip package bundled UCX
    search_paths = ["/opt/ucx", "/usr/local", "/usr"]
    if _detect_nonroot(pod_name):
        search_paths = [f"{NONROOT_PREFIX}/ucx"] + search_paths
    for path in search_paths:
        result = exec_in_pod(
            pod_name,
            ["bash", "-c", f"test -f {path}/lib/libucp.so && echo FOUND"],
            use_debug=False,
        )
        if result.returncode == 0 and "FOUND" in result.stdout:
            return path
    # Check for UCX bundled in the NIXL pip package (versioned .so names)
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", f"ls {NIXL_PIP_UCX_LIBS}/libucp-*.so* 2>/dev/null && echo FOUND"],
        use_debug=False,
    )
    if result.returncode == 0 and "FOUND" in result.stdout:
        return NIXL_PIP_UCX_LIBS
    return None


def build_nixl(pod_name, out=None):
    """Clone and build NIXL library from source (UCX plugin only)."""
    _out = out or print
    nixl_pfx, _bench_pfx, local_pfx = _get_install_prefix(pod_name)
    nonroot = _detect_nonroot(pod_name)
    _out(f"  Building NIXL library on {pod_name} ...")

    # Detect UCX path for meson
    ucx_path = _detect_ucx_path(pod_name)
    if ucx_path == NIXL_PIP_UCX_LIBS and nonroot:
        # pip UCX has versioned .so names and no headers — create a proper prefix
        ucx_path = _setup_ucx_prefix_from_pip(pod_name, out=_out)
    if ucx_path:
        _out(f"    Found UCX at {ucx_path}")
    elif nonroot and _get_common().INSTALL_DEPS:
        # No UCX found on non-root pod — build gdrcopy + UCX from source
        _build_gdrcopy_nonroot(pod_name, out=_out)
        ucx_path = _build_ucx_from_source(pod_name, out=_out)
        if ucx_path:
            _out(f"    Built UCX from source at {ucx_path}")
        else:
            _out(f"    Warning: UCX build from source failed, NIXL build may fail.")
    else:
        _out(f"    Warning: UCX not found, NIXL build may fail.")

    nixl_src = f"{NIXL_BUILD_DIR}/nixl"
    meson_args = (
        f"--prefix={nixl_pfx} --buildtype=release "
        f"-Denable_plugins=UCX -Dinstall_headers=true"
    )
    if ucx_path and ucx_path not in ("/usr", "/usr/local"):
        meson_args += f" -Ducx_path={ucx_path}"

    # Clean stale directory — chmod first to fix meson subproject perms, then rm
    _run_build_step(
        pod_name,
        f"chmod -R u+rwx {nixl_src} 2>/dev/null; rm -rf {nixl_src}",
        "clean", _out, timeout=60, ignore_error=True,
    )
    clone_cmd = _clone_or_download(
        pod_name, NIXL_REPO, "main", nixl_src, _out)
    steps = [
        (clone_cmd, "clone NIXL", 300),
        (f"cd {nixl_src} && meson setup build {meson_args}",
         "meson configure", 900),
        (f"cd {nixl_src}/build && ninja", "build", 900),
        (f"cd {nixl_src}/build && ninja install", "install", 120),
    ]
    for cmd, label, step_timeout in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=step_timeout):
            return False

    if not nonroot:
        # Configure linker to find NIXL libraries (root only)
        ldconfig_cmd = (
            f'NIXL_LIBDIR=$(find {nixl_pfx} -maxdepth 1 -name "lib*" -type d | head -1) && '
            f'echo "$NIXL_LIBDIR" > /etc/ld.so.conf.d/nixl.conf && '
            f'[ -d "$NIXL_LIBDIR/plugins" ] && echo "$NIXL_LIBDIR/plugins" >> /etc/ld.so.conf.d/nixl.conf; '
            f'ldconfig'
        )
        if not _run_build_step(pod_name, ldconfig_cmd, "ldconfig", _out):
            _out(f"    Warning: ldconfig failed, may need LD_LIBRARY_PATH at runtime.")

        symlink_cmd = (
            f'NIXL_LIBDIR=$(find {nixl_pfx} -maxdepth 1 -name "lib*" -type d | head -1) && '
            f'SYSLIB=$([ -d /usr/local/lib64 ] && echo /usr/local/lib64 || echo /usr/local/lib) && '
            f'mkdir -p $SYSLIB && '
            f'for f in $NIXL_LIBDIR/lib*.so*; do '
            f'  [ -f "$f" ] && ln -sf "$f" $SYSLIB/$(basename "$f") || true; '
            f'done; '
            f'for f in $NIXL_LIBDIR/plugins/lib*.so*; do '
            f'  [ -f "$f" ] && ln -sf "$f" $SYSLIB/$(basename "$f") || true; '
            f'done; ldconfig'
        )
        if not _run_build_step(pod_name, symlink_cmd, "library symlinks", _out):
            _out(f"    Warning: symlink creation failed.")
    else:
        _out(f"    Non-root: skipping ldconfig/symlinks (using LD_LIBRARY_PATH)")
        # Create lib64 symlink for nixlbench (meson uses libdir=lib64 for find_library)
        _run_build_step(
            pod_name,
            f"rm -rf {nixl_pfx}/lib64 && "
            f"NIXL_LIBDIR=$(find {nixl_pfx}/lib -maxdepth 2 -name 'libnixl.so' -exec dirname {{}} \\; | head -1) && "
            f"[ -n \"$NIXL_LIBDIR\" ] && ln -sf \"$NIXL_LIBDIR\" {nixl_pfx}/lib64",
            "create lib64 symlink", _out, timeout=10, ignore_error=True,
        )

    _out(f"  NIXL built and installed at {nixl_pfx} on {pod_name}.")
    return True


def _build_nixlbench_binary(pod_name, out=None):
    """Build nixlbench binary against installed NIXL."""
    _out = out or print
    nixl_pfx, bench_pfx, local_pfx = _get_install_prefix(pod_name)
    nonroot = _detect_nonroot(pod_name)
    _out(f"  Building nixlbench on {pod_name} ...")

    nixlbench_src = f"{NIXL_BUILD_DIR}/nixl/benchmark/nixlbench"
    pkg_env = f"export PKG_CONFIG_PATH={local_pfx}/lib64/pkgconfig:{local_pfx}/lib/pkgconfig:$PKG_CONFIG_PATH"
    meson_args = (
        f"-Dnixl_path={nixl_pfx}/ "
        f"-Dprefix={bench_pfx} "
        f"-Dlibdir=lib64 "
        f"-Detcd_inc_path={local_pfx}/include "
        f"-Detcd_lib_path={local_pfx}/lib64 "
        f"--buildtype=release"
    )

    # Ensure gflags wrap is available (needed for non-root where system libgflags-dev is missing)
    if nonroot:
        _run_build_step(
            pod_name,
            f"cd {nixlbench_src} && "
            f"(test -f subprojects/gflags.wrap || meson wrap install gflags)",
            "install gflags wrap", _out, timeout=60, ignore_error=True,
        )
        # Create etcd include/lib dirs even when empty (meson.build references them unconditionally)
        exec_in_pod(pod_name, ["bash", "-c", f"mkdir -p {local_pfx}/include {local_pfx}/lib64"], use_debug=False)

    # Patch worker meson.build to add asio_dep (upstream bug: worker includes asio_runtime.h
    # which needs asio.hpp but asio_dep is not in worker_deps)
    _run_build_step(
        pod_name,
        f"cd {nixlbench_src} && "
        f"sed -i 's/tomlplusplus_dep$/tomlplusplus_dep,\\n  asio_dep/' src/worker/meson.build",
        "patch worker deps", _out, timeout=30, ignore_error=True,
    )

    steps = [
        (f"{pkg_env} && cd {nixlbench_src} && meson setup build {meson_args}",
         "meson configure", 300),
        (f"cd {nixlbench_src}/build && ninja", "build", 600),
        (f"cd {nixlbench_src}/build && ninja install", "install", 120),
    ]
    for cmd, label, step_timeout in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=step_timeout):
            return False

    if not nonroot:
        # Root path: set up profile.d and symlinks
        env_setup = (
            f'echo "export PATH={bench_pfx}/bin:{nixl_pfx}/bin:\\$PATH" '
            f'> /etc/profile.d/nixlbench.sh && '
            f'echo "export LD_LIBRARY_PATH={bench_pfx}/lib:'
            f'{bench_pfx}/lib64:'
            f'{nixl_pfx}/lib64:{nixl_pfx}/lib64/plugins:'
            f'{nixl_pfx}/lib/\\$(uname -m)-linux-gnu:'
            f'{nixl_pfx}/lib/\\$(uname -m)-linux-gnu/plugins:'
            f'/usr/local/lib64:/usr/local/lib:/opt/ucx/lib:'
            f'\\$LD_LIBRARY_PATH" >> /etc/profile.d/nixlbench.sh && '
            f'chmod +x /etc/profile.d/nixlbench.sh && '
            f'ln -sf {bench_pfx}/bin/nixlbench /usr/local/bin/nixlbench'
        )
        if not _run_build_step(pod_name, env_setup, "environment setup", _out):
            _out(f"    Warning: PATH setup failed. nixlbench may not be found via command -v.")
    else:
        _out(f"    Non-root: nixlbench installed at {bench_pfx}/bin/ (using LD_LIBRARY_PATH)")

    _out(f"  nixlbench built and installed at {bench_pfx} on {pod_name}.")
    return True


def _check_pip_nixl_libs(pod_name):
    """Check if NIXL libs are available from the pip package."""
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", f"test -f {NIXL_PIP_LIBS}/libnixl.so && echo FOUND"],
        use_debug=False,
    )
    return result.returncode == 0 and "FOUND" in (result.stdout or "")


def _setup_pip_nixl_for_build(pod_name, out=None):
    """Set up NIXL from pip package for building nixlbench against it.

    Creates a fake NIXL install prefix at NONROOT_NIXL_PREFIX with:
    - lib64/ symlinked to pip package libs (libnixl.so, libnixl_build.so, libserdes.so)
    - include/ from the cloned NIXL source (src/api/cpp/ headers)
    """
    _out = out or print
    nixl_pfx = NONROOT_NIXL_PREFIX if _detect_nonroot(pod_name) else NIXL_INSTALL_PREFIX
    nixl_src = f"{NIXL_BUILD_DIR}/nixl"

    _out(f"  Reusing NIXL libs from pip package (nixl_cu12) ...")

    # nixlbench meson.build looks for: nixl_path/include/ and nixl_path/<libdir>/
    setup_cmd = (
        f"mkdir -p {nixl_pfx}/lib64 {nixl_pfx}/include && "
        # Symlink .so files from pip .mesonpy.libs (libnixl.so, libnixl_build.so, libserdes.so etc)
        f"for f in {NIXL_PIP_LIBS}/lib*.so*; do "
        f"  [ -f \"$f\" ] && ln -sf \"$f\" {nixl_pfx}/lib64/$(basename \"$f\") || true; "
        f"done && "
        # Symlink plugins directory (UCX backend etc)
        f"ln -sfn {NIXL_PIP_LIBS}/plugins {nixl_pfx}/lib64/plugins && "
        # Copy API headers from cloned source (nixlbench uses nixl_path/include/)
        f"cp -r {nixl_src}/src/api/cpp/*.h {nixl_pfx}/include/ 2>/dev/null || true && "
        # Also copy backend headers if present
        f"mkdir -p {nixl_pfx}/include/backend && "
        f"cp -r {nixl_src}/src/api/cpp/backend/*.h {nixl_pfx}/include/backend/ 2>/dev/null || true && "
        # Copy infra headers (nixl_descriptors.h deps)
        f"cp -r {nixl_src}/src/infra/*.h {nixl_pfx}/include/ 2>/dev/null || true && "
        # Copy all internal utility headers recursively (nixlbench uses utils/common/, utils/serdes/ etc)
        f"cp -a {nixl_src}/src/utils {nixl_pfx}/include/ 2>/dev/null || true && "
        f"echo DONE"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", setup_cmd], timeout=30, use_debug=False)
    if result.returncode != 0 or "DONE" not in (result.stdout or ""):
        _out(f"    Warning: failed to set up pip NIXL libs: {(result.stderr or '').strip()[:200]}")
        return False

    # Verify the critical files
    verify = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f"test -f {nixl_pfx}/lib64/libnixl.so && "
         f"test -f {nixl_pfx}/lib64/libnixl_build.so && "
         f"test -f {nixl_pfx}/lib64/libserdes.so && "
         f"test -f {nixl_pfx}/include/nixl.h && echo OK"],
        use_debug=False,
    )
    if "OK" not in (verify.stdout or ""):
        _out(f"    Warning: NIXL prefix verification failed — missing libs or headers")
        return False

    _out(f"    NIXL prefix set up at {nixl_pfx} (libs from pip, headers from source)")
    return True


def build_nixlbench_from_source(pod_name, out=None):
    """Full build pipeline: deps -> etcd-cpp-api -> NIXL -> nixlbench."""
    _out = out or print
    _out(f"  === Building nixlbench from source on {pod_name} ===")
    nonroot = _detect_nonroot(pod_name)

    # Ensure build directory exists
    exec_in_pod(pod_name, ["mkdir", "-p", NIXL_BUILD_DIR], use_debug=False)

    # Step 1: Install system and Python build dependencies
    if not install_nixlbench_deps(pod_name, out=_out):
        return False

    # Step 1.5: Create git stub if git is not available (meson needs it for version info)
    if nonroot:
        exec_in_pod(
            pod_name,
            ["bash", "-c",
             f"mkdir -p {NONROOT_BIN} && "
             f"if ! command -v git >/dev/null 2>&1; then "
             f"cat > {NONROOT_BIN}/git << 'GITEOF'\n"
             "#!/bin/bash\n"
             "if [[ \"$*\" == *\"rev-parse\"* ]]; then echo \"00000000\"; exit 0; "
             "elif [[ \"$*\" == *\"describe\"* ]]; then echo \"v0.0.0\"; exit 0; "
             "elif [[ \"$*\" == *\"submodule\"* ]]; then exit 0; "
             "elif [[ \"$*\" == *\"clone\"* ]]; then echo \"git stub: clone not supported\" >&2; exit 1; "
             "else exit 0; fi\n"
             f"GITEOF\n"
             f"chmod +x {NONROOT_BIN}/git; fi"],
            use_debug=False,
        )

    # Step 2: Build etcd-cpp-apiv3 (optional but needed for multi-node)
    etcd_ok = build_etcd_cpp_api(pod_name, out=_out)
    if not etcd_ok:
        _out(f"  Continuing without etcd-cpp-apiv3 (nixlbench etcd runtime will be disabled).")

    # Step 3: Build NIXL from source (full build for consistent headers + libs)
    # Note: pip NIXL libs have version mismatches with the public source headers,
    # so we always do a full build from source for ABI compatibility.
    if not build_nixl(pod_name, out=_out):
        _out(f"  NIXL build failed. Cannot continue.")
        return False

    # Step 4: Build nixlbench binary
    if not _build_nixlbench_binary(pod_name, out=_out):
        _out(f"  nixlbench build failed.")
        return False

    _out(f"  === nixlbench build complete on {pod_name} ===")
    return True


def _check_etcd(pod_name):
    """Check if etcd and etcdctl are available on a pod."""
    return _check_binary(pod_name, "etcd") and _check_binary(pod_name, "etcdctl")


def install_etcd(pod_name, out=None):
    """Download and install etcd binary on a pod."""
    _out = out or print
    if _check_etcd(pod_name):
        _out(f"  etcd already available on {pod_name}.")
        return True

    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: etcd not found on {pod_name}.")
        _out(f"  Re-run with --install-deps to automatically install etcd.")
        sys.exit(1)

    bin_dir = _get_bin_dir(pod_name)
    _out(f"  Installing etcd {ETCD_VER} on {pod_name} (to {bin_dir}) ...")
    install_script = (
        f'mkdir -p {bin_dir} && '
        f'curl -L {ETCD_DOWNLOAD_URL} -o /tmp/etcd.tar.gz'
        f' && tar xzf /tmp/etcd.tar.gz -C /tmp'
        f' && cp /tmp/etcd-{ETCD_VER}-linux-amd64/etcd {bin_dir}/'
        f' && cp /tmp/etcd-{ETCD_VER}-linux-amd64/etcdctl {bin_dir}/'
        f' && rm -rf /tmp/etcd*'
        f' && echo "ETCD_INSTALLED"'
    )
    result = exec_in_pod(pod_name, ["bash", "-c", install_script], timeout=120, use_debug=False)
    if result.returncode != 0 or "ETCD_INSTALLED" not in result.stdout:
        _out(f"  Error: failed to install etcd: {(result.stderr or result.stdout or '').strip()[:200]}")
        return False
    _out(f"  etcd {ETCD_VER} installed on {pod_name}.")
    return True


# ---------------------------------------------------------------------------
# ETCD lifecycle
# ---------------------------------------------------------------------------
def start_etcd_server(pod_name, pod_ip, out=None):
    """Start etcd server in background on a pod. Returns Popen process."""
    _out = out or print
    _c = _get_common()

    bin_dir = _get_bin_dir(pod_name)
    etcd_args = [
        "bash", "-c",
        f"export PATH={bin_dir}:/usr/local/bin:$PATH && "
        f"etcd "
        f"--data-dir=/tmp/etcd-nixlbench-data "
        f"--listen-client-urls=http://0.0.0.0:{ETCD_PORT} "
        f"--advertise-client-urls=http://{pod_ip}:{ETCD_PORT} "
        f"--listen-peer-urls=http://0.0.0.0:2380 "
        f"--initial-advertise-peer-urls=http://{pod_ip}:2380 "
        f"--initial-cluster=default=http://{pod_ip}:2380",
    ]
    cmd = _c._build_remote_cmd(pod_name, etcd_args)

    _out(f"  Starting etcd server on {pod_name} ({pod_ip}:{ETCD_PORT}) ...")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _c._server_procs_append(proc)

    # Wait for etcd to be ready
    time.sleep(2)
    health_result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f"export PATH={bin_dir}:/usr/local/bin:$PATH && "
         f"etcdctl endpoint health --endpoints=http://localhost:{ETCD_PORT}"],
        use_debug=False,
    )
    if health_result.returncode != 0:
        _out(f"  Warning: etcd health check failed, waiting 3 more seconds ...")
        time.sleep(3)

    _out(f"  etcd server started on {pod_name}.")
    return proc


def stop_etcd_server(proc, out=None):
    """Stop etcd server process."""
    _out = out or print
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
    _out("  etcd server stopped.")


def cleanup_etcd_state(pod_name, etcd_endpoint, group=None, out=None):
    """Clean up etcd state after a benchmark run."""
    _out = out or print
    prefix = f"xferbench/{group}" if group else "xferbench"
    exec_in_pod(
        pod_name,
        ["bash", "-c",
         f"export PATH={NONROOT_BIN}:/usr/local/bin:$PATH && "
         f"etcdctl del {prefix} --prefix=true --endpoints={etcd_endpoint}"],
        use_debug=False,
    )
    if _get_common().VERBOSE:
        _out(f"  Cleaned etcd state: prefix={prefix}")


# ---------------------------------------------------------------------------
# NIXLBench execution
# ---------------------------------------------------------------------------
NIXLBENCH_ASIO_PORT = 12345


def _check_etcd_in_binary(pod_name):
    """Check if nixlbench was built with working etcd runtime (not just listed in help)."""
    nixl_pfx, bench_pfx, _local_pfx = _get_install_prefix(pod_name)
    nonroot_pfx = _nonroot_env_prefix(pod_name)
    env_prefix = (
        f'{nonroot_pfx}'
        f'export PATH={bench_pfx}/bin:{nixl_pfx}/bin:'
        f'{NONROOT_BIN}:/usr/local/bin:$PATH && '
        f'export LD_LIBRARY_PATH={bench_pfx}/lib:'
        f'{bench_pfx}/lib64:'
        f'{nixl_pfx}/lib64:{nixl_pfx}/lib64/plugins:'
        f'{nixl_pfx}/lib/$(uname -m)-linux-gnu:'
        f'{NIXL_PIP_LIBS}:{NIXL_PIP_UCX_LIBS}:'
        f'/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}'
    )
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f'{env_prefix} && nixlbench --etcd_endpoints http://127.0.0.1:1 '
         f'--benchmark_group test 2>&1 | head -5'],
        timeout=10, use_debug=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return "Invalid runtime" not in output


def _build_nixlbench_cmd(etcd_endpoint, benchmark_group, pod_name=None,
                         runtime_type=None, asio_address=None):
    """Build full shell command for nixlbench with proper PATH/LD_LIBRARY_PATH."""
    buffer_bytes = _get_common()._parse_size(NIXLBENCH_BUFFER_SIZE)

    if runtime_type == "ASIO":
        nixl_args = (
            f"-runtime_type ASIO "
            f"-asio_address {asio_address or '127.0.0.1'} "
            f"-asio_port {NIXLBENCH_ASIO_PORT} "
            f"--backend {NIXLBENCH_BACKEND} "
            f"--initiator_seg_type {NIXLBENCH_SEG_TYPE} "
            f"--target_seg_type {NIXLBENCH_SEG_TYPE} "
            f"--op_type WRITE "
            f"--total_buffer_size {buffer_bytes}"
        )
    else:
        nixl_args = (
            f"--etcd_endpoints {etcd_endpoint} "
            f"--benchmark_group {benchmark_group} "
            f"--backend {NIXLBENCH_BACKEND} "
            f"--initiator_seg_type {NIXLBENCH_SEG_TYPE} "
            f"--target_seg_type {NIXLBENCH_SEG_TYPE} "
            f"--op_type WRITE "
            f"--total_buffer_size {buffer_bytes}"
        )
    # Include both standard and non-root paths for maximum compatibility
    env_setup = (
        f"export HOME=/tmp && "
        f"export PATH={NONROOT_NIXLBENCH_PREFIX}/bin:{NONROOT_NIXL_PREFIX}/bin:"
        f"{NONROOT_BIN}:"
        f"{NIXLBENCH_INSTALL_PREFIX}/bin:{NIXL_INSTALL_PREFIX}/bin:"
        f"/usr/local/bin:$PATH && "
        f"export LD_LIBRARY_PATH={NONROOT_NIXLBENCH_PREFIX}/lib:"
        f"{NONROOT_NIXLBENCH_PREFIX}/lib64:"
        f"{NONROOT_NIXL_PREFIX}/lib64:{NONROOT_NIXL_PREFIX}/lib64/plugins:"
        f"{NONROOT_NIXL_PREFIX}/lib/$(uname -m)-linux-gnu:"
        f"{NONROOT_NIXL_PREFIX}/lib/$(uname -m)-linux-gnu/plugins:"
        f"{NONROOT_PREFIX}/lib64:{NONROOT_PREFIX}/lib:"
        f"{NIXL_PIP_LIBS}:{NIXL_PIP_UCX_LIBS}:"
        f"{NIXLBENCH_INSTALL_PREFIX}/lib:{NIXLBENCH_INSTALL_PREFIX}/lib64:"
        f"{NIXL_INSTALL_PREFIX}/lib64:{NIXL_INSTALL_PREFIX}/lib64/plugins:"
        f"{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu:"
        f"{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu/plugins:"
        f"/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}"
    )
    return ["bash", "-c", f"{env_setup} && {NIXLBENCH_BINARY} {nixl_args}"]


def run_nixlbench_pair(src_pod, src_ip, dst_pod, dst_ip, etcd_endpoint, group,
                       out=None, use_asio=False):
    """Run nixlbench between two pods. Returns parsed metrics dict or None.

    Launches target first (Popen), then initiator (subprocess.run blocking).
    When use_asio=True, uses direct ASIO socket communication (no etcd needed).
    Otherwise uses etcd for rank coordination.
    """
    _out = out or print
    _c = _get_common()

    if use_asio:
        # ASIO: both use dst_ip as address. First to start binds (rank 0),
        # second fails bind and connects (rank 1). Target starts first.
        target_cmd_args = _build_nixlbench_cmd(
            etcd_endpoint, group, runtime_type="ASIO", asio_address=dst_ip)
        initiator_cmd_args = _build_nixlbench_cmd(
            etcd_endpoint, group, runtime_type="ASIO", asio_address=dst_ip)
        short_desc = f"nixlbench --backend {NIXLBENCH_BACKEND} -runtime_type ASIO"
    else:
        target_cmd_args = _build_nixlbench_cmd(etcd_endpoint, group)
        initiator_cmd_args = _build_nixlbench_cmd(etcd_endpoint, group)
        short_desc = f"nixlbench --backend {NIXLBENCH_BACKEND} --benchmark_group {group}"

    # Start target (background)
    target_cmd = _c._build_remote_cmd(dst_pod, target_cmd_args)
    _out(f"  Target: {short_desc}")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(target_cmd)}")
    target_proc = subprocess.Popen(target_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _c._server_procs_append(target_proc)

    # Brief delay for target to start listening
    time.sleep(3 if use_asio else 2)

    # Start initiator (blocking)
    initiator_cmd = _c._build_remote_cmd(src_pod, initiator_cmd_args)
    _out(f"  Initiator: {short_desc}")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(initiator_cmd)}")

    try:
        result = subprocess.run(
            initiator_cmd, capture_output=True, text=True, timeout=NIXLBENCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        _out(f"    TIMEOUT after {NIXLBENCH_TIMEOUT}s")
        result = None

    # Clean up target and capture its output
    try:
        target_proc.terminate()
    except OSError:
        pass
    try:
        target_stdout, target_stderr = target_proc.communicate(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        target_stdout, target_stderr = b"", b""
        try:
            target_proc.kill()
        except OSError:
            pass
    _c._server_procs_remove(target_proc)

    if _c.VERBOSE and target_stdout:
        target_out = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
        _out(f"  nixlbench target output ({len(target_out)} chars):\n{target_out[-2000:]}")

    # Clean up etcd state for this group (only when using etcd runtime)
    if not use_asio:
        cleanup_etcd_state(_etcd_pod_name, etcd_endpoint, group=group, out=_out)

    if result is None or result.returncode != 0:
        stderr_msg = result.stderr.strip()[:200] if result else "timeout"
        _out(f"    FAILED: {stderr_msg}")
        # Even if initiator fails, try parsing target output (it may have results)
        if target_stdout:
            target_text = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
            parsed = parse_nixlbench_output(target_text, out=_out)
            if parsed:
                return parsed
        return None

    # Try initiator output first, then target output
    parsed = parse_nixlbench_output(result.stdout, out=_out)
    if not parsed and target_stdout:
        target_text = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
        parsed = parse_nixlbench_output(target_text, out=_out)
    return parsed


# Module-level state for etcd pod (set during run_nixlbench)
_etcd_pod_name = None


def parse_nixlbench_output(stdout, out=None):
    """Parse nixlbench columnar output.

    nixlbench outputs fixed-column tables.  Known column orders:
      Short (8 cols): Block Size (B), Batch Size, B/W (GB/Sec), Avg Lat. (us),
                      Avg Prep (us), P99 Prep (us), Avg Post (us), P99 Post (us)
      Long (12 cols): adds Aggregate B/W (GB/Sec), Network Util (%),
                      Avg Tx (us), P99 Tx (us)  after B/W

    Data rows are whitespace-separated numbers.  The header row (multi-word
    column names) is detected but NOT split — we use positional mapping
    directly since the column order is fixed.

    Returns dict with bw_gbs, lat_avg_us, lat_p99_us (prep) for the largest
    block-size row, or None.
    """
    _out = out or print

    if not stdout or not stdout.strip():
        _out("    No output from nixlbench")
        return None

    if _get_common().VERBOSE:
        _out(f"  nixlbench raw output ({len(stdout)} chars):\n{stdout[-3000:]}")

    lines = stdout.strip().split("\n")

    # Collect candidate data rows — lines where ALL tokens are numeric.
    # Also note whether we see a header line (to confirm it's nixlbench output).
    saw_header = False
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if "block size" in lower or "b/w" in lower or "gb/sec" in lower:
            saw_header = True
            continue

        # A data row: first char is a digit and all whitespace-separated
        # tokens parse as floats.
        if stripped[0].isdigit():
            parts = stripped.split()
            try:
                vals = [float(p) for p in parts]
                if len(vals) >= 3:  # need at least block_size, batch, bw
                    data_rows.append(vals)
            except ValueError:
                pass

    if not data_rows:
        _out("    Warning: could not parse nixlbench output (no data rows)")
        return None

    # Use last data row (largest block size)
    row = data_rows[-1]
    ncols = len(row)
    metrics = {}

    # 10-col format (observed):
    #   0=Block Size  1=Batch  2=B/W  3=Avg Lat  4=Avg Prep  5=P99 Prep
    #   6=Avg Post  7=P99 Post  8=Avg Tx  9=P99 Tx
    # 8-col format (short — no Avg Tx / P99 Tx):
    #   0=Block Size  1=Batch  2=B/W  3=Avg Lat  4=Avg Prep  5=P99 Prep
    #   6=Avg Post  7=P99 Post
    # 12-col format (long — adds Aggregate B/W + Network Util after B/W):
    #   0=Block Size 1=Batch 2=B/W 3=Agg B/W 4=Net Util 5=Avg Lat
    #   6=Avg Prep 7=P99 Prep 8=Avg Post 9=P99 Post 10=Avg Tx 11=P99 Tx
    if ncols >= 8 and ncols <= 10:
        metrics["bw_gbs"] = row[2]
        metrics["lat_avg_us"] = row[3]     # Avg Lat
        metrics["lat_p99_us"] = row[7]     # P99 Post (completion/ack)
    elif ncols >= 11:
        metrics["bw_gbs"] = row[2]
        metrics["lat_avg_us"] = row[5]     # Avg Lat
        metrics["lat_p99_us"] = row[9]     # P99 Post
    else:
        metrics["bw_gbs"] = row[2] if ncols > 2 else row[-1]

    if not saw_header and _get_common().VERBOSE:
        _out(f"    Note: parsed {ncols}-column data row without header confirmation")

    block_size = int(row[0]) if row[0] == int(row[0]) else row[0]
    _out(f"    Parsed: block={block_size}  B/W={metrics.get('bw_gbs', '?')} GB/s"
         f"  avg_lat={metrics.get('lat_avg_us', '?')} us"
         f"  p99_lat={metrics.get('lat_p99_us', '?')} us")

    return metrics


def _run_nixlbench_matrix(pods, display_names, etcd_endpoint, use_asio=False):
    """Run nixlbench for all pod pairs. Returns dict of metric_name -> NxN matrix."""
    _c = _get_common()
    n = len(pods)
    test_pairs = _c._cross_role_pairs(pods)
    total_pairs = len(test_pairs)

    print(f"\n{'─' * 50}")
    print(f"  Running nixlbench — backend={NIXLBENCH_BACKEND} "
          f"seg_type={NIXLBENCH_SEG_TYPE} buffer={NIXLBENCH_BUFFER_SIZE} "
          f"— {total_pairs} pair(s)")
    print(f"{'─' * 50}")

    waves = _c._schedule_parallel_pairs(n, pairs=test_pairs)

    # Metric matrices — populated as results come in
    bw_matrix = [[None] * n for _ in range(n)]
    lat_avg_matrix = [[None] * n for _ in range(n)]
    lat_p99_matrix = [[None] * n for _ in range(n)]

    done = 0

    for wave_idx, wave in enumerate(waves):
        pair_labels = ", ".join(
            f"{display_names[pods[i][0]]}->{display_names[pods[j][0]]}"
            for i, j in wave
        )
        print(f"\n  [wave {wave_idx + 1}/{len(waves)}] "
              f"{len(wave)} pair(s) in parallel: {pair_labels}")

        buffers = []
        results_slot = [None] * len(wave)
        done_events = [threading.Event() for _ in wave]

        def _worker(slot, i, j, buf, done_evt):
            try:
                src_name, src_ip = pods[i]
                dst_name, dst_ip = pods[j]
                src_short = display_names[src_name]
                dst_short = display_names[dst_name]

                def _out(msg):
                    buf.write(msg + "\n")

                _out(f"\n    {src_short} -> {dst_short}")
                group = f"pair_{i}_{j}"
                pair_metrics = run_nixlbench_pair(
                    src_name, src_ip, dst_name, dst_ip,
                    etcd_endpoint, group, out=_out,
                    use_asio=use_asio,
                )
                if pair_metrics:
                    for k, v in pair_metrics.items():
                        _out(f"    {k}: {v}")
                else:
                    _out(f"    FAIL")
                results_slot[slot] = (i, j, pair_metrics)
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

        # Store results in matrices
        for slot in range(len(wave)):
            if results_slot[slot] is not None:
                i, j, pair_metrics = results_slot[slot]
                if pair_metrics:
                    if "bw_gbs" in pair_metrics:
                        bw_matrix[i][j] = pair_metrics["bw_gbs"]
                    if "lat_avg_us" in pair_metrics:
                        lat_avg_matrix[i][j] = pair_metrics["lat_avg_us"]
                    if "lat_p99_us" in pair_metrics:
                        lat_p99_matrix[i][j] = pair_metrics["lat_p99_us"]
                done += 1

    print(f"\n  Completed {done}/{total_pairs} pair(s)")

    # Build results dict — only include metrics that have at least one value
    results = {}
    if any(bw_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write BW (GB/s)"] = bw_matrix
    if any(lat_avg_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write Latency avg (usec)"] = lat_avg_matrix
    if any(lat_p99_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write Latency P99 (usec)"] = lat_p99_matrix

    return results


def run_nixlbench(pods, display_names):
    """Run nixlbench between all pod pairs.

    1. Verify nixlbench on all pods (parallel)
    2. Install/verify etcd on pod 0
    3. Start etcd on pod 0
    4. Run nixlbench for all pairs
    5. Stop etcd
    """
    global _etcd_pod_name
    _c = _get_common()
    n = len(pods)

    print(f"\n{'=' * 60}")
    print("  NIXLBENCH (NIXL Data Transfer Benchmark)")
    print(f"{'=' * 60}")
    print(f"Settings: backend={NIXLBENCH_BACKEND}, seg_type={NIXLBENCH_SEG_TYPE}, "
          f"buffer_size={NIXLBENCH_BUFFER_SIZE}")

    # --- Step 1: Verify nixlbench on all pods in parallel ---
    print("\nEnsuring nixlbench is available on all pods (parallel) ...")
    ensure_bufs = [_c._StreamingBuffer() for _ in range(n)]
    ensure_done = [threading.Event() for _ in range(n)]

    ensure_failed = [False] * n

    def _ensure_worker(idx, pod_name, buf, done_evt):
        try:
            ensure_nixlbench(pod_name, out=lambda msg: buf.write(msg + "\n"))
        except SystemExit:
            buf.write(f"  FATAL: nixlbench not found on {pod_name}\n")
            ensure_failed[idx] = True
        except Exception as exc:
            buf.write(f"  ERROR on {pod_name}: {exc}\n")
            ensure_failed[idx] = True
        finally:
            done_evt.set()

    ensure_threads = []
    for idx, (name, _ip) in enumerate(pods):
        t = threading.Thread(
            target=_ensure_worker,
            args=(idx, name, ensure_bufs[idx], ensure_done[idx]),
            daemon=True,
        )
        ensure_threads.append(t)
        t.start()

    for idx in range(n):
        while not ensure_done[idx].is_set():
            ensure_bufs[idx].flush_new()
            ensure_done[idx].wait(timeout=0.1)
        ensure_bufs[idx].flush_all()

    for t in ensure_threads:
        t.join(timeout=5)

    if any(ensure_failed):
        failed_pods = [display_names[pods[i][0]] for i in range(n) if ensure_failed[i]]
        print(f"\n  Aborting: nixlbench not available on {', '.join(failed_pods)}")
        return {}

    # --- Step 2: Detect runtime mode (ASIO if etcd support not compiled in) ---
    etcd_pod_name, etcd_pod_ip = pods[0]
    _etcd_pod_name = etcd_pod_name
    use_asio = not _check_etcd_in_binary(etcd_pod_name)
    if use_asio:
        print(f"\n  Using ASIO runtime (nixlbench built without etcd-cpp-api)")
        etcd_endpoint = ""
        etcd_proc = None
    else:
        print(f"\nSetting up etcd on {display_names[etcd_pod_name]} ...")
        install_etcd(etcd_pod_name)
        # --- Step 3: Start etcd server ---
        etcd_proc = start_etcd_server(etcd_pod_name, etcd_pod_ip)
        etcd_endpoint = f"http://{etcd_pod_ip}:{ETCD_PORT}"

    try:
        # --- Step 4: Run nixlbench matrix ---
        results = _run_nixlbench_matrix(pods, display_names, etcd_endpoint,
                                        use_asio=use_asio)
    finally:
        # --- Step 5: Stop etcd ---
        if etcd_proc:
            stop_etcd_server(etcd_proc)

    return results


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-nixlbench.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _USAGE = """\
Usage: uv run run-tests-nixlbench.py [options]

Run NIXLBench (NIXL data transfer benchmark) between Kubernetes inference pods.

Equivalent to: run-tests.sh -t nixlbench

Options:
  --nixlbench-backend BACKEND
                        NIXLBench backend (default: UCX).
  --nixlbench-seg-type TYPE
                        Segment type for initiator/target (default: VRAM).
  --nixlbench-buffer-size SIZE
                        Total buffer size (default: 8G).
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -h, --help            Show this help message.
  -i, --install-deps    Build nixlbench from source and install etcd
                        if missing.  Requires CUDA and UCX in the pod image.
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
    _cfg = _c._parse_common_args(extra_flags={
        ("--nixlbench-backend", "--nixlbench-backend"):       ("_NIXLBENCH_BACKEND", True),
        ("--nixlbench-seg-type", "--nixlbench-seg-type"):     ("_NIXLBENCH_SEG_TYPE", True),
        ("--nixlbench-buffer-size", "--nixlbench-buffer-size"): ("_NIXLBENCH_BUFFER_SIZE", True),
    })

    # Apply nixlbench-specific config
    _raw_backend = _cfg.pop("_NIXLBENCH_BACKEND", None)
    if _raw_backend:
        NIXLBENCH_BACKEND = _raw_backend
    _raw_seg = _cfg.pop("_NIXLBENCH_SEG_TYPE", None)
    if _raw_seg:
        NIXLBENCH_SEG_TYPE = _raw_seg
    _raw_buf = _cfg.pop("_NIXLBENCH_BUFFER_SIZE", None)
    if _raw_buf:
        NIXLBENCH_BUFFER_SIZE = _raw_buf

    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    _results = run_nixlbench(_pods, _display_names)
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
