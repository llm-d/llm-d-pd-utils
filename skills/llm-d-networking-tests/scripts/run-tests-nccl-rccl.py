# /// script
# requires-python = ">=3.10"
# ///
"""NCCL/RCCL collective operation test runner for Kubernetes inference pods.

Detects GPU vendor per pod (NVIDIA vs AMD), builds nccl-tests or rccl-tests
from source if needed, sets up passwordless SSH for mpirun, runs
all_reduce_perf across same-vendor pods, and displays results.

Public API:
    run_nccl_rccl(pods, display_names) -> dict
"""

import subprocess
import sys
import threading

from importlib import import_module

# Import shared utilities from the common module.
_common = None
_discovery = None


def _get_common():
    global _common
    if _common is None:
        _common = import_module("run-tests-common")
    return _common


def _get_discovery():
    global _discovery
    if _discovery is None:
        _discovery = import_module("run-tests-discovery")
    return _discovery


# Delegated to run-tests-common
def exec_in_pod(*args, **kwargs):
    return _get_common().exec_in_pod(*args, **kwargs)


def _kubectl_ns_args():
    return _get_common()._kubectl_ns_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests.git"
RCCL_TESTS_REPO = "https://github.com/ROCm/rccl-tests.git"
NCCL_TESTS_DIR = "/tmp/nccl-tests"
RCCL_TESTS_DIR = "/tmp/rccl-tests"
NCCL_BINARY = "/tmp/nccl-tests/build/all_reduce_perf"
RCCL_BINARY = "/tmp/rccl-tests/build/all_reduce_perf"

OPENMPI_VERSION = "4.1.6"
OPENMPI_URL = (f"https://download.open-mpi.org/release/open-mpi/v4.1/"
               f"openmpi-{OPENMPI_VERSION}.tar.gz")
OPENMPI_BUILD_DIR = "/tmp/openmpi-build"

# Default all_reduce_perf arguments
AR_MIN_BYTES = "8"
AR_MAX_BYTES = "128M"
AR_FACTOR = "2"
AR_GPUS_PER_PROC = "1"

# SSH port for sshd in pods
SSHD_PORT = 22


# ---------------------------------------------------------------------------
# GPU vendor detection (reuses run-tests-discovery.py)
# ---------------------------------------------------------------------------
def detect_gpu_vendor(pod_name):
    """Detect GPU vendor on a pod: 'nvidia', 'amd', or 'unknown'."""
    return _get_discovery()._detect_gpu_vendor(pod_name)


def detect_all_gpu_vendors(pods):
    """Detect GPU vendor for all pods in parallel.

    Returns dict: {pod_name: 'nvidia'|'amd'|'unknown'}
    """
    _c = _get_common()
    vendors = {}
    lock = threading.Lock()
    bufs = [_c._StreamingBuffer() for _ in range(len(pods))]
    done_events = [threading.Event() for _ in range(len(pods))]

    def _worker(idx, pod_name, buf, done_evt):
        try:
            v = detect_gpu_vendor(pod_name)
            buf.write(f"  {pod_name}: {v} GPU\n")
            with lock:
                vendors[pod_name] = v
        except Exception as exc:
            buf.write(f"  {pod_name}: ERROR detecting GPU vendor: {exc}\n")
            with lock:
                vendors[pod_name] = "unknown"
        finally:
            done_evt.set()

    threads = []
    for idx, (name, _ip) in enumerate(pods):
        t = threading.Thread(
            target=_worker,
            args=(idx, name, bufs[idx], done_events[idx]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for idx in range(len(pods)):
        while not done_events[idx].is_set():
            bufs[idx].flush_new()
            done_events[idx].wait(timeout=0.1)
        bufs[idx].flush_all()

    for t in threads:
        t.join(timeout=5)

    return vendors


# ---------------------------------------------------------------------------
# GPU count detection
# ---------------------------------------------------------------------------
def detect_gpu_count(pod_name, vendor):
    """Detect number of GPUs on a pod. Returns int or 0 on failure."""
    if vendor == "nvidia":
        result = exec_in_pod(pod_name, ["bash", "-c", "nvidia-smi -L 2>/dev/null | wc -l"],
                             timeout=15, use_debug=False)
    elif vendor == "amd":
        result = exec_in_pod(pod_name, ["bash", "-c", "rocm-smi --showid 2>/dev/null | grep -c GPU || echo 0"],
                             timeout=15, use_debug=False)
    else:
        return 0
    if result.returncode == 0:
        try:
            return int(result.stdout.strip())
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# Binary availability and build
# ---------------------------------------------------------------------------
def _tests_available(pod_name, binary_path):
    """Check if all_reduce_perf binary exists at the given path."""
    result = exec_in_pod(pod_name, ["test", "-x", binary_path], timeout=10, use_debug=False)
    return result.returncode == 0


def nccl_tests_available(pod_name):
    return _tests_available(pod_name, NCCL_BINARY)


def rccl_tests_available(pod_name):
    return _tests_available(pod_name, RCCL_BINARY)


def install_build_deps(pod_name, out=None):
    """Install build dependencies for nccl/rccl-tests and OpenMPI.

    Installs git, make, gcc, curl, then builds OpenMPI from source if
    mpirun is not already available (RHEL UBI containers typically lack
    openmpi packages in their repos).
    """
    _out = out or print
    _out(f"  Installing build dependencies on {pod_name} ...")

    # Step 1: Install basic build tools via package manager
    rpm_pkgs = ["git", "make", "gcc", "gcc-c++", "curl", "bzip2"]
    apt_pkgs = ["git", "make", "gcc", "g++", "curl", "bzip2"]

    for pkg_mgr, install_cmd in [
        ("dnf", ["dnf", "install", "-y"]),
        ("yum", ["yum", "install", "-y"]),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            result = exec_in_pod(pod_name, install_cmd + rpm_pkgs,
                                 timeout=300, use_debug=False)
            if result.returncode != 0:
                _out(f"    Warning: failed to install build tools: "
                     f"{result.stderr.strip()[:200]}")
            else:
                _out(f"    Installed build tools via {pkg_mgr}.")
            # Also try installing openmpi package (works on Fedora, not UBI)
            result = exec_in_pod(pod_name, install_cmd + ["openmpi", "openmpi-devel"],
                                 timeout=120, use_debug=False)
            if result.returncode == 0:
                # Package install worked — symlink to standard PATH
                exec_in_pod(pod_name, ["bash", "-c",
                    "if [ -d /usr/lib64/openmpi/bin ] && ! command -v mpirun >/dev/null 2>&1; then "
                    "  ln -sf /usr/lib64/openmpi/bin/* /usr/local/bin/ 2>/dev/null; "
                    "  ln -sf /usr/lib64/openmpi/lib/* /usr/local/lib/ 2>/dev/null; "
                    "  ldconfig 2>/dev/null; "
                    "fi; true"
                ], use_debug=False)
            break
    else:
        exec_in_pod(pod_name, ["apt-get", "update"], timeout=60, use_debug=False)
        result = exec_in_pod(pod_name, ["apt-get", "install", "-y"] + apt_pkgs,
                             timeout=300, use_debug=False)
        if result.returncode != 0:
            _out(f"    Warning: failed to install build tools: "
                 f"{result.stderr.strip()[:200]}")
        else:
            _out(f"    Installed build tools via apt.")
        # Try package openmpi on Debian/Ubuntu
        result = exec_in_pod(pod_name, ["apt-get", "install", "-y",
                                         "openmpi-bin", "libopenmpi-dev"],
                             timeout=120, use_debug=False)

    # Step 2: If mpirun still not available, build OpenMPI from source
    mpirun = find_mpirun(pod_name)
    if mpirun:
        _out(f"    mpirun already available: {mpirun}")
        return True

    _out(f"    OpenMPI not found in packages — building v{OPENMPI_VERSION} from source ...")
    build_script = (
        f"rm -rf {OPENMPI_BUILD_DIR} && mkdir -p {OPENMPI_BUILD_DIR} && "
        f"cd {OPENMPI_BUILD_DIR} && "
        f"curl -sL {OPENMPI_URL} | tar xz && "
        f"cd openmpi-{OPENMPI_VERSION} && "
        f"./configure --prefix=/usr/local --disable-man-pages "
        f"  --with-cuda=/usr/local/cuda 2>&1 | tail -3 && "
        f"make -j$(nproc) 2>&1 | tail -3 && "
        f"make install 2>&1 | tail -3 && "
        f"ldconfig"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", build_script],
                         timeout=600, use_debug=False)
    if result.returncode != 0:
        _out(f"    OpenMPI build failed: {result.stderr.strip()[:500]}")
        return False

    mpirun = find_mpirun(pod_name)
    if mpirun:
        _out(f"    OpenMPI built successfully: {mpirun}")
        return True

    _out(f"    Warning: OpenMPI build completed but mpirun not found.")
    return False


def _find_nccl_paths(pod_name):
    """Find NCCL include and lib paths on a pod.

    Prefers the pip nvidia-nccl package over the system NCCL, because
    the system NCCL at /lib64/ may have been compiled for a newer CUDA
    than the installed driver supports (e.g. NCCL 2.29.7+cuda13.2 on a
    CUDA 12.9 driver).  The pip package is typically matched to the
    container's CUDA toolkit.

    Returns (nccl_inc, nccl_lib) or (None, None).
    """
    result = exec_in_pod(pod_name, ["bash", "-c",
        # 1. Prefer pip nvidia-nccl package (matched to container CUDA)
        'PIP_INC=$(find /opt -path "*/nvidia/nccl/include/nccl.h" -type f 2>/dev/null | head -1); '
        'PIP_LIB=$(find /opt -path "*/nvidia/nccl/lib/libnccl.so*" -type f 2>/dev/null | head -1); '
        'if [ -n "$PIP_INC" ] && [ -n "$PIP_LIB" ]; then '
        '  echo "INC=$(dirname $PIP_INC)"; '
        '  echo "LIB=$(dirname $PIP_LIB)"; '
        '  exit 0; '
        'fi; '
        # 2. Standard CUDA paths
        'if [ -f /usr/local/cuda/include/nccl.h ]; then '
        '  echo "INC=/usr/local/cuda/include"; '
        '  echo "LIB=/usr/local/cuda/lib64"; '
        'elif [ -f /usr/include/nccl.h ]; then '
        '  echo "INC=/usr/include"; '
        '  echo "LIB=/usr/lib64"; '
        'else '
        '  if [ -n "$PIP_INC" ]; then echo "INC=$(dirname $PIP_INC)"; fi; '
        '  if [ -f /usr/lib64/libnccl.so ]; then '
        '    echo "LIB=/usr/lib64"; '
        '  elif [ -n "$PIP_LIB" ]; then '
        '    echo "LIB=$(dirname $PIP_LIB)"; '
        '  fi; '
        'fi'
    ], timeout=15, use_debug=False)

    nccl_inc = nccl_lib = None
    for line in (result.stdout or "").strip().split("\n"):
        if line.startswith("INC="):
            nccl_inc = line[4:]
        elif line.startswith("LIB="):
            nccl_lib = line[4:]
    return nccl_inc, nccl_lib


def build_nccl_tests(pod_name, out=None):
    """Clone and build nccl-tests from source. Returns True on success."""
    _out = out or print

    # Find NCCL headers and libs
    nccl_inc, nccl_lib = _find_nccl_paths(pod_name)
    if not nccl_inc or not nccl_lib:
        _out(f"  Error: NCCL headers/libs not found on {pod_name} "
             f"(inc={nccl_inc}, lib={nccl_lib}).")
        return False
    _out(f"  NCCL paths: inc={nccl_inc}, lib={nccl_lib}")

    _out(f"  Cloning nccl-tests on {pod_name} ...")
    exec_in_pod(pod_name, ["rm", "-rf", NCCL_TESTS_DIR], use_debug=False)
    result = exec_in_pod(pod_name, ["git", "clone", NCCL_TESTS_REPO, NCCL_TESTS_DIR],
                         timeout=120, use_debug=False)
    if result.returncode != 0:
        _out(f"  Error cloning nccl-tests: {result.stderr.strip()[:300]}")
        return False

    # Verify clone produced a Makefile
    check = exec_in_pod(pod_name, ["test", "-f", f"{NCCL_TESTS_DIR}/Makefile"],
                        timeout=10, use_debug=False)
    if check.returncode != 0:
        _out(f"  Error: clone succeeded but {NCCL_TESTS_DIR}/Makefile not found.")
        return False

    _out(f"  Building nccl-tests on {pod_name} (make MPI=1) ...")
    build_cmd = (
        f"cd {NCCL_TESTS_DIR} && "
        f"make MPI=1 MPI_HOME=/usr/local CUDA_HOME=/usr/local/cuda "
        f"NCCL_INC={nccl_inc} NCCL_LIB={nccl_lib} -j$(nproc)"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", build_cmd], timeout=300, use_debug=False)
    if _get_common().VERBOSE and result.stdout.strip():
        _out(result.stdout[-500:])
    if result.returncode != 0:
        _out(f"  Build failed: {result.stderr.strip()[:500]}")
        return False
    _out(f"  nccl-tests built successfully on {pod_name}.")
    return True


def build_rccl_tests(pod_name, out=None):
    """Clone and build rccl-tests from source. Returns True on success."""
    _out = out or print
    _out(f"  Cloning rccl-tests on {pod_name} ...")
    exec_in_pod(pod_name, ["rm", "-rf", RCCL_TESTS_DIR], use_debug=False)
    result = exec_in_pod(pod_name, ["git", "clone", RCCL_TESTS_REPO, RCCL_TESTS_DIR],
                         timeout=120, use_debug=False)
    if result.returncode != 0:
        _out(f"  Error cloning rccl-tests: {result.stderr.strip()[:300]}")
        return False

    _out(f"  Building rccl-tests on {pod_name} (make MPI=1) ...")
    build_cmd = (
        f"cd {RCCL_TESTS_DIR} && "
        f"make MPI=1 HIP_HOME=/opt/rocm RCCL_HOME=/opt/rocm -j$(nproc)"
    )
    result = exec_in_pod(pod_name, ["bash", "-c", build_cmd], timeout=300, use_debug=False)
    if _get_common().VERBOSE and result.stdout.strip():
        _out(result.stdout[-500:])
    if result.returncode != 0:
        _out(f"  Build failed: {result.stderr.strip()[:500]}")
        return False
    _out(f"  rccl-tests built successfully on {pod_name}.")
    return True


def ensure_collective_tests(pod_name, vendor, out=None):
    """Ensure all_reduce_perf is available, building if needed."""
    _out = out or print
    if vendor == "nvidia":
        if nccl_tests_available(pod_name):
            _out(f"  nccl-tests already available on {pod_name}.")
            return True
        if not _get_common().INSTALL_DEPS:
            _out(f"  Error: nccl-tests not found on {pod_name}. Re-run with --install-deps.")
            return False
        _out(f"  nccl-tests not found on {pod_name}, building ...")
        return build_nccl_tests(pod_name, out=_out)
    elif vendor == "amd":
        if rccl_tests_available(pod_name):
            _out(f"  rccl-tests already available on {pod_name}.")
            return True
        if not _get_common().INSTALL_DEPS:
            _out(f"  Error: rccl-tests not found on {pod_name}. Re-run with --install-deps.")
            return False
        _out(f"  rccl-tests not found on {pod_name}, building ...")
        return build_rccl_tests(pod_name, out=_out)
    else:
        _out(f"  Skipping {pod_name}: unknown GPU vendor.")
        return False


# ---------------------------------------------------------------------------
# OpenSSH setup / teardown
# ---------------------------------------------------------------------------
def install_ssh(pod_name, out=None):
    """Install SSH server and client on a pod.

    Prefers dropbear (lightweight, no privsep/chroot needed in containers)
    and falls back to openssh-server if dropbear is unavailable.
    """
    _out = out or print
    _out(f"  Installing SSH on {pod_name} ...")

    # Try rpm-based first, then apt
    for pkg_mgr, install_cmd in [
        ("dnf", ["dnf", "install", "-y"]),
        ("yum", ["yum", "install", "-y"]),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            # Prefer dropbear (works in containers without SYS_CHROOT)
            result = exec_in_pod(pod_name, install_cmd + ["dropbear", "openssh-clients"],
                                 timeout=120, use_debug=False)
            if result.returncode == 0:
                _out(f"    dropbear + ssh client installed via {pkg_mgr}.")
                return True
            # Fallback to openssh
            result = exec_in_pod(pod_name, install_cmd + ["openssh-server", "openssh-clients"],
                                 timeout=120, use_debug=False)
            if result.returncode == 0:
                _out(f"    openssh installed via {pkg_mgr}.")
                return True
            _out(f"    Warning: {pkg_mgr} install failed: {result.stderr.strip()[:200]}")
            return False

    # apt-based
    exec_in_pod(pod_name, ["apt-get", "update"], timeout=60, use_debug=False)
    result = exec_in_pod(pod_name, ["apt-get", "install", "-y", "dropbear", "openssh-client"],
                         timeout=120, use_debug=False)
    if result.returncode == 0:
        _out(f"    dropbear + ssh client installed via apt.")
        return True
    result = exec_in_pod(pod_name, ["apt-get", "install", "-y",
                                     "openssh-server", "openssh-client"],
                         timeout=120, use_debug=False)
    if result.returncode == 0:
        _out(f"    openssh installed via apt.")
        return True
    _out(f"    Warning: apt install failed: {result.stderr.strip()[:200]}")
    return False


# Well-known locations where mpirun may be installed on RPM-based systems.
_MPIRUN_SEARCH_PATHS = [
    "/usr/lib64/openmpi/bin",
    "/usr/lib/openmpi/bin",
    "/usr/local/bin",
    "/opt/amazon/openmpi/bin",
    "/opt/hpcx/ompi/bin",
]


def find_mpirun(pod_name):
    """Find the full path to mpirun on a pod.

    Returns the path string, or None if not found.
    """
    # First try: mpirun on PATH
    result = exec_in_pod(pod_name, ["bash", "-c", "command -v mpirun 2>/dev/null"],
                         timeout=10, use_debug=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    # Search well-known locations
    search = " || ".join(
        f'([ -x {p}/mpirun ] && echo {p}/mpirun)'
        for p in _MPIRUN_SEARCH_PATHS
    )
    result = exec_in_pod(pod_name, ["bash", "-c", f"{search} || true"],
                         timeout=10, use_debug=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")[0].strip()

    return None


def install_mpi(pod_name, out=None):
    """Install OpenMPI on a pod. Returns path to mpirun or None."""
    _out = out or print

    # Check if already available
    mpirun = find_mpirun(pod_name)
    if mpirun:
        _out(f"  mpirun already available on {pod_name}: {mpirun}")
        return mpirun

    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: mpirun not found on {pod_name}. Re-run with --install-deps.")
        return None

    _out(f"  Installing OpenMPI on {pod_name} ...")

    for pkg_mgr, install_cmd, packages in [
        ("dnf", ["dnf", "install", "-y"], ["openmpi", "openmpi-devel"]),
        ("yum", ["yum", "install", "-y"], ["openmpi", "openmpi-devel"]),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            result = exec_in_pod(pod_name, install_cmd + packages,
                                 timeout=180, use_debug=False)
            if result.returncode == 0:
                _out(f"    OpenMPI installed via {pkg_mgr}.")
            else:
                _out(f"    Warning: {pkg_mgr} install failed: {result.stderr.strip()[:200]}")
            # Even on failure, check if mpirun appeared
            mpirun = find_mpirun(pod_name)
            if mpirun:
                _out(f"    mpirun found at: {mpirun}")
                return mpirun
            _out(f"    Warning: mpirun still not found after install.")
            return None

    # apt-based
    exec_in_pod(pod_name, ["apt-get", "update"], timeout=60, use_debug=False)
    result = exec_in_pod(pod_name, ["apt-get", "install", "-y",
                                     "openmpi-bin", "libopenmpi-dev"],
                         timeout=180, use_debug=False)
    if result.returncode == 0:
        _out(f"    OpenMPI installed via apt.")
    else:
        _out(f"    Warning: apt install failed: {result.stderr.strip()[:200]}")

    mpirun = find_mpirun(pod_name)
    if mpirun:
        _out(f"    mpirun found at: {mpirun}")
        return mpirun
    _out(f"    Warning: mpirun still not found after install.")
    return None


def setup_ssh_cluster(group_pods, out=None):
    """Set up passwordless SSH across a group of pods for mpirun.

    - Generate a key pair on the first pod
    - Distribute public key to all pods
    - Start sshd on all pods
    - Configure StrictHostKeyChecking=no

    Returns True on success.
    """
    _out = out or print
    if not group_pods:
        return False

    first_pod = group_pods[0][0]

    # Install openssh on all pods in parallel
    _c = _get_common()
    n = len(group_pods)
    bufs = [_c._StreamingBuffer() for _ in range(n)]
    done_events = [threading.Event() for _ in range(n)]
    install_ok = [False] * n

    def _install_worker(idx, pod_name, buf, done_evt):
        try:
            install_ok[idx] = install_ssh(pod_name, out=lambda msg: buf.write(msg + "\n"))
        except Exception as exc:
            buf.write(f"  ERROR installing openssh on {pod_name}: {exc}\n")
        finally:
            done_evt.set()

    threads = []
    for idx, (name, _ip) in enumerate(group_pods):
        t = threading.Thread(
            target=_install_worker,
            args=(idx, name, bufs[idx], done_events[idx]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for idx in range(n):
        while not done_events[idx].is_set():
            bufs[idx].flush_new()
            done_events[idx].wait(timeout=0.1)
        bufs[idx].flush_all()

    for t in threads:
        t.join(timeout=5)

    if not all(install_ok):
        _out("  Warning: openssh installation failed on some pods.")

    # Determine the actual HOME directory (may differ from /root in containers)
    home_result = exec_in_pod(first_pod, ["bash", "-c", 'echo "$HOME"'],
                              timeout=10, use_debug=False)
    ssh_home = (home_result.stdout.strip() or "/root")
    ssh_dir = f"{ssh_home}/.ssh"
    key_path = f"{ssh_dir}/nccl_test_key"
    _out(f"  Using SSH home: {ssh_home}")

    # Generate SSH key on first pod
    _out(f"\n  Generating SSH key pair on {first_pod} ...")
    exec_in_pod(first_pod, ["bash", "-c",
        f"mkdir -p {ssh_dir} && chmod 700 {ssh_dir} && "
        f"rm -f {key_path} {key_path}.pub && "
        f"ssh-keygen -t rsa -b 2048 -N '' -f {key_path} -q"
    ], use_debug=False)

    # Read the public key
    result = exec_in_pod(first_pod, ["cat", f"{key_path}.pub"],
                         timeout=10, use_debug=False)
    if result.returncode != 0 or not result.stdout.strip():
        _out(f"  Error: failed to generate SSH key at {key_path}.pub")
        return False
    pub_key = result.stdout.strip()

    # Distribute public key and start sshd on all pods
    for name, _ip in group_pods:
        _out(f"  Setting up SSH on {name} ...")

        # Create .ssh dir, add authorized key, configure ssh.
        # We write authorized_keys to BOTH the container user's home AND
        # /root/.ssh/, because dropbear/sshd look up the target user's home
        # (we SSH as root, so it checks /root/.ssh/authorized_keys).
        setup_script = (
            f"mkdir -p {ssh_dir} && chmod 700 {ssh_dir} && "
            f"echo '{pub_key}' >> {ssh_dir}/authorized_keys && "
            f"chmod 600 {ssh_dir}/authorized_keys && "
            f"echo 'StrictHostKeyChecking no' > {ssh_dir}/config && "
            f"echo 'UserKnownHostsFile /dev/null' >> {ssh_dir}/config && "
            f"chmod 600 {ssh_dir}/config && "
            # Also set up /root/.ssh if HOME is not /root
            f"if [ '{ssh_home}' != '/root' ]; then "
            f"  mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
            f"  echo '{pub_key}' >> /root/.ssh/authorized_keys && "
            f"  chmod 600 /root/.ssh/authorized_keys && "
            f"  cp {ssh_dir}/config /root/.ssh/config && "
            f"  chmod 600 /root/.ssh/config; "
            f"fi && "
            f"mkdir -p /etc/ssh"
        )
        exec_in_pod(name, ["bash", "-c", setup_script], use_debug=False)

        # Unlock root account (needed for dropbear and some sshd configs)
        exec_in_pod(name, ["bash", "-c", "passwd -u root 2>/dev/null; true"],
                    use_debug=False)

        # Kill any existing SSH server first
        exec_in_pod(name, ["bash", "-c",
            "pkill dropbear 2>/dev/null; pkill sshd 2>/dev/null; sleep 0.3; true"
        ], timeout=10, use_debug=False)

        # Start SSH server: prefer dropbear (no privsep/chroot needed)
        # dropbear forks to background by default (no -F), don't use -E
        # (logging to stderr keeps kubectl exec open)
        result = exec_in_pod(name, ["bash", "-c",
            "if command -v dropbear >/dev/null 2>&1; then "
            "  mkdir -p /etc/dropbear && "
            "  dropbear -R -p 22 2>/dev/null && sleep 0.3 && "
            "  pgrep -x dropbear >/dev/null && echo SSHD_OK || echo SSHD_FAIL; "
            "else "
            "  mkdir -p /run/sshd && "
            "  [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A 2>/dev/null; "
            "  (/usr/sbin/sshd -o PermitRootLogin=yes -o UsePrivilegeSeparation=no "
            f"   -o AuthorizedKeysFile={ssh_dir}/authorized_keys 2>/dev/null || "
            "   sshd -o PermitRootLogin=yes 2>/dev/null) "
            "  && echo SSHD_OK || echo SSHD_FAIL; "
            "fi"
        ], timeout=30, use_debug=False)
        if "SSHD_OK" in (result.stdout or ""):
            _out(f"    SSH server started on {name}.")
        else:
            _out(f"    Warning: SSH server may not have started on {name}: "
                 f"{result.stderr.strip()[:200]}")

    # Copy private key as id_rsa for mpirun convenience (both user home and /root)
    exec_in_pod(first_pod, ["bash", "-c",
        f"cp {key_path} {ssh_dir}/id_rsa && chmod 600 {ssh_dir}/id_rsa && "
        f"if [ '{ssh_home}' != '/root' ]; then "
        f"  cp {key_path} /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa && "
        f"  cp {key_path} /root/.ssh/nccl_test_key && chmod 600 /root/.ssh/nccl_test_key; "
        f"fi"
    ], use_debug=False)

    # Verify SSH connectivity from first pod to all others
    _out("\n  Verifying SSH connectivity ...")
    all_ok = True
    for name, ip in group_pods:
        result = exec_in_pod(first_pod, ["bash", "-c",
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ConnectTimeout=5 -i {key_path} root@{ip} echo ok"
        ], timeout=15, use_debug=False)
        if result.returncode == 0:
            _out(f"    {first_pod} -> {name} ({ip}): OK")
        else:
            _out(f"    {first_pod} -> {name} ({ip}): FAILED — "
                 f"{result.stderr.strip()[:200]}")
            all_ok = False

    # Return both ok status and key_path for use in mpirun
    return all_ok, key_path


def stop_sshd_all(pods, ssh_dir=None, out=None):
    """Stop SSH server (dropbear or sshd) on all pods and clean up keys."""
    _out = out or print
    _out("\n  Stopping SSH servers on all pods ...")
    clean_dir = ssh_dir or "$HOME/.ssh"
    for name, _ip in pods:
        exec_in_pod(name, ["bash", "-c",
            "pkill dropbear 2>/dev/null; pkill sshd 2>/dev/null; "
            f"rm -f {clean_dir}/nccl_test_key {clean_dir}/nccl_test_key.pub "
            f"{clean_dir}/id_rsa "
            f"/root/.ssh/nccl_test_key /root/.ssh/id_rsa "
            f"/root/.ssh/authorized_keys /root/.ssh/config; true"
        ], timeout=10, use_debug=False)
    _out("  SSH servers stopped and keys cleaned up.")


# ---------------------------------------------------------------------------
# all_reduce_perf execution and parsing
# ---------------------------------------------------------------------------
def parse_all_reduce_output(output):
    """Parse all_reduce_perf tabular output.

    Returns list of dicts with keys:
        size, count, data_type, redop, root, time_us, algbw_gbs, busbw_gbs
    """
    rows = []
    in_data = False
    for line in output.strip().split("\n"):
        stripped = line.strip()
        # Data lines start with a number (size column)
        if stripped and stripped[0].isdigit() and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) >= 8:
                try:
                    rows.append({
                        "size": int(parts[0]),
                        "count": int(parts[1]),
                        "data_type": parts[2],
                        "redop": parts[3],
                        "root": int(parts[4]) if parts[4].isdigit() else 0,
                        "time_us": float(parts[5]),
                        "algbw_gbs": float(parts[6]),
                        "busbw_gbs": float(parts[7]),
                    })
                except (ValueError, IndexError):
                    continue
            in_data = True
        elif in_data and not stripped:
            break  # end of data section
    return rows


def _human_bytes(n):
    """Format byte count for display: 8, 16, ..., 1K, ..., 128M."""
    if n < 1024:
        return str(n)
    elif n < 1024 * 1024:
        return f"{n // 1024}K"
    elif n < 1024 * 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    else:
        return f"{n // (1024 * 1024 * 1024)}G"


def run_all_reduce(group_pods, vendor, display_names, gpu_count,
                   mpirun_path="mpirun", key_path=None, nccl_lib_path=None):
    """Run all_reduce_perf across a group of same-vendor pods.

    Returns parsed results list or None on failure.
    """
    if not group_pods:
        return None

    first_pod = group_pods[0][0]
    n_pods = len(group_pods)
    total_gpus = n_pods * gpu_count

    # Build host list: ip:slots
    host_list = ",".join(f"{ip}:{gpu_count}" for _name, ip in group_pods)

    # Determine binary path
    binary = NCCL_BINARY if vendor == "nvidia" else RCCL_BINARY

    # Build LD_LIBRARY_PATH: prefer pip NCCL lib (matched to container CUDA)
    # over system NCCL (which may be compiled for a newer CUDA).
    ld_prefix = ""
    if nccl_lib_path:
        ld_prefix += f"{nccl_lib_path}:"
    ld_prefix += "/usr/lib64"

    mpi_cmd = (
        f"{mpirun_path} --allow-run-as-root "
        f"-np {total_gpus} -N {gpu_count} "
        f"-H {host_list} "
        f"-x LD_LIBRARY_PATH={ld_prefix}:$LD_LIBRARY_PATH "
        f"-x PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH "
    )

    # Add NCCL/RCCL-specific env vars.
    # NCCL_SOCKET_IFNAME=eth0 forces NCCL to use the pod network, not
    # service/secondary IPs that aren't routable between pods.
    # NCCL_IB_DISABLE=1 forces socket transport — IB/RoCE may have
    # connectivity issues in some clusters (vendor err 129 / status 12).
    # For IB testing, run with NCCL_IB_DISABLE=0 explicitly.
    if vendor == "nvidia":
        mpi_cmd += (
            "-x NCCL_DEBUG=WARN -x NCCL_SOCKET_IFNAME=eth0 "
            "-x NCCL_IB_DISABLE=1 "
        )
    else:
        mpi_cmd += (
            "-x RCCL_DEBUG=WARN -x NCCL_SOCKET_IFNAME=eth0 "
            "-x NCCL_IB_DISABLE=1 "
        )

    # Force MPI communication over the pod network (eth0) — without this,
    # MPI/UCX may try secondary interfaces (e.g. service IPs) that aren't
    # routable between pods.
    # --mca pml ob1: use OB1 PML (not UCX PML which picks wrong interfaces)
    # --mca btl tcp,self: restrict BTL to TCP sockets
    # --mca btl_tcp_if_include eth0: limit TCP BTL to pod network
    # --mca oob_tcp_if_include eth0: limit OOB (out-of-band) to pod network
    mpi_cmd += (
        f"--mca pml ob1 "
        f"--mca btl tcp,self "
        f"--mca btl_tcp_if_include eth0 "
        f"--mca oob_tcp_if_include eth0 "
        f"--mca plm_rsh_args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null{' -i ' + key_path if key_path else ''}' "
        f"{binary} -b {AR_MIN_BYTES} -e {AR_MAX_BYTES} -f {AR_FACTOR} -g {AR_GPUS_PER_PROC}"
    )

    vendor_label = "NCCL" if vendor == "nvidia" else "RCCL"
    pod_names = ", ".join(display_names.get(name, name) for name, _ in group_pods)
    print(f"\n  Running {vendor_label} all_reduce_perf across {n_pods} pods "
          f"({total_gpus} GPUs): {pod_names}")

    if _get_common().VERBOSE:
        print(f"  $ {mpi_cmd}")

    try:
        result = exec_in_pod(first_pod, ["bash", "-c", mpi_cmd], timeout=600, use_debug=False)
    except subprocess.TimeoutExpired:
        print(f"  all_reduce_perf TIMED OUT after 600s")
        print(f"  This may indicate a NCCL communication hang between pods.")
        print(f"  Try running with -v for the full mpirun command, or reduce message sizes.")
        return None

    if result.returncode != 0:
        print(f"  all_reduce_perf FAILED (exit {result.returncode})")
        combined = (result.stdout or "") + (result.stderr or "")
        if "CUDA driver version is insufficient" in combined:
            print(f"  Cause: CUDA driver/runtime version mismatch on one or more pods.")
            print(f"  The nccl-tests binary was compiled against a CUDA runtime newer than")
            print(f"  the GPU driver installed on the node. Check `nvidia-smi` on each pod.")
        if result.stderr.strip():
            print(f"  stderr (last 1000 chars): {result.stderr.strip()[-1000:]}")
        if result.stdout.strip():
            print(f"  stdout (last 1000 chars): {result.stdout.strip()[-1000:]}")
        return None

    # Print raw output if verbose
    if _get_common().VERBOSE:
        print(f"\n  Raw output:\n{result.stdout}")

    rows = parse_all_reduce_output(result.stdout)
    if not rows:
        print("  Warning: no data rows parsed from all_reduce_perf output.")
        if result.stdout.strip():
            print(f"  Output:\n{result.stdout.strip()[-1000:]}")
        return None

    return rows


def print_all_reduce_results(rows, vendor, n_pods, total_gpus):
    """Print all_reduce_perf results as a formatted table."""
    vendor_label = "NCCL" if vendor == "nvidia" else "RCCL"
    print(f"\n  {vendor_label} all_reduce_perf ({n_pods} {vendor.upper()} pods, "
          f"{total_gpus} GPUs total):")
    print(f"  {'Size':>10s}  {'Count':>12s}  {'Time(us)':>12s}  "
          f"{'AlgBW(GB/s)':>12s}  {'BusBW(GB/s)':>12s}")
    print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    for r in rows:
        size_str = _human_bytes(r["size"])
        print(f"  {size_str:>10s}  {r['count']:>12d}  {r['time_us']:>12.2f}  "
              f"{r['algbw_gbs']:>12.2f}  {r['busbw_gbs']:>12.2f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_nccl_rccl(pods, display_names):
    """Run NCCL/RCCL all_reduce_perf tests.

    1. Detect GPU vendor per pod (parallel)
    2. Group by vendor
    3. Ensure tests built (parallel per group)
    4. Setup SSH, run all_reduce, teardown SSH
    5. Return results dict
    """
    _c = _get_common()

    print(f"\n{'=' * 60}")
    print("  NCCL/RCCL COLLECTIVE TESTS (all_reduce_perf)")
    print(f"{'=' * 60}")

    # Step 1: Detect GPU vendors
    print("\nDetecting GPU vendors on all pods ...")
    vendors = detect_all_gpu_vendors(pods)

    # Group pods by vendor
    nvidia_pods = [(name, ip) for name, ip in pods if vendors.get(name) == "nvidia"]
    amd_pods = [(name, ip) for name, ip in pods if vendors.get(name) == "amd"]
    unknown_pods = [(name, ip) for name, ip in pods if vendors.get(name) not in ("nvidia", "amd")]

    if unknown_pods:
        print(f"\n  Skipping {len(unknown_pods)} pod(s) with unknown GPU vendor: "
              f"{', '.join(display_names.get(n, n) for n, _ in unknown_pods)}")

    if not nvidia_pods and not amd_pods:
        print("\n  No pods with recognized GPUs found. Skipping NCCL/RCCL tests.")
        return {}

    all_results = {}

    for vendor, group_pods in [("nvidia", nvidia_pods), ("amd", amd_pods)]:
        if not group_pods:
            continue

        vendor_label = "NCCL" if vendor == "nvidia" else "RCCL"
        print(f"\n{'─' * 50}")
        print(f"  {vendor_label} tests — {len(group_pods)} {vendor.upper()} pods")
        print(f"{'─' * 50}")

        # Detect GPU count
        print("\n  Detecting GPU count ...")
        gpu_counts = {}
        for name, _ip in group_pods:
            count = detect_gpu_count(name, vendor)
            gpu_counts[name] = count
            print(f"    {display_names.get(name, name)}: {count} GPUs")

        counts_set = set(gpu_counts.values())
        if 0 in counts_set:
            zero_pods = [display_names.get(n, n) for n, _ in group_pods if gpu_counts[n] == 0]
            print(f"  Warning: GPU count is 0 on: {', '.join(zero_pods)}. Skipping.")
            continue
        if len(counts_set) > 1:
            print(f"  Warning: GPU count varies across pods: {counts_set}. "
                  f"All pods must have the same GPU count. Skipping.")
            continue

        gpu_count = counts_set.pop()
        total_gpus = len(group_pods) * gpu_count
        print(f"  {gpu_count} GPUs per pod, {total_gpus} total")

        # Pre-flight: check CUDA driver/runtime compatibility on all pods
        if vendor == "nvidia":
            print("\n  Checking CUDA driver/runtime compatibility ...")
            cuda_ok = True
            for name, _ip in group_pods:
                result = exec_in_pod(name, ["bash", "-c",
                    "python3 -c 'import torch; torch.cuda.init(); "
                    "print(f\"driver={torch.version.cuda}, "
                    "runtime={torch.version.cuda}\")' 2>&1 || "
                    "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"
                ], timeout=30, use_debug=False)
                out_str = (result.stdout or "").strip()
                if "error" in out_str.lower() or "insufficient" in out_str.lower():
                    print(f"    {display_names.get(name, name)}: CUDA MISMATCH — {out_str[:200]}")
                    cuda_ok = False
                else:
                    print(f"    {display_names.get(name, name)}: {out_str[:100]}")
            if not cuda_ok:
                print(f"  Warning: CUDA driver/runtime mismatch detected. "
                      f"all_reduce_perf may fail on affected pods.")

        n = len(group_pods)

        # Step A: Install build deps (git, make, gcc, openmpi-devel) on all pods.
        # This must happen BEFORE building nccl/rccl-tests because `make MPI=1`
        # needs MPI headers at compile time.
        if _c.INSTALL_DEPS:
            print(f"\n  Installing build dependencies (git, make, gcc, OpenMPI) ...")
            dep_bufs = [_c._StreamingBuffer() for _ in range(n)]
            dep_done = [threading.Event() for _ in range(n)]

            def _dep_worker(idx, pod_name, buf, done_evt):
                try:
                    install_build_deps(pod_name, out=lambda msg: buf.write(msg + "\n"))
                except Exception as exc:
                    buf.write(f"  ERROR installing deps on {pod_name}: {exc}\n")
                finally:
                    done_evt.set()

            dep_threads = []
            for idx, (name, _ip) in enumerate(group_pods):
                t = threading.Thread(
                    target=_dep_worker,
                    args=(idx, name, dep_bufs[idx], dep_done[idx]),
                    daemon=True,
                )
                dep_threads.append(t)
                t.start()

            for idx in range(n):
                while not dep_done[idx].is_set():
                    dep_bufs[idx].flush_new()
                    dep_done[idx].wait(timeout=0.1)
                dep_bufs[idx].flush_all()

            for t in dep_threads:
                t.join(timeout=5)

        # Step B: Ensure nccl/rccl-tests are built (parallel)
        print(f"\n  Ensuring {vendor_label.lower()}-tests are available ...")
        bufs = [_c._StreamingBuffer() for _ in range(n)]
        done_events = [threading.Event() for _ in range(n)]
        build_ok = [False] * n

        def _build_worker(idx, pod_name, v, buf, done_evt):
            try:
                build_ok[idx] = ensure_collective_tests(
                    pod_name, v, out=lambda msg: buf.write(msg + "\n")
                )
            except Exception as exc:
                buf.write(f"  ERROR on {pod_name}: {exc}\n")
            finally:
                done_evt.set()

        threads = []
        for idx, (name, _ip) in enumerate(group_pods):
            t = threading.Thread(
                target=_build_worker,
                args=(idx, name, vendor, bufs[idx], done_events[idx]),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for idx in range(n):
            while not done_events[idx].is_set():
                bufs[idx].flush_new()
                done_events[idx].wait(timeout=0.1)
            bufs[idx].flush_all()

        for t in threads:
            t.join(timeout=5)

        if not all(build_ok):
            failed = [display_names.get(group_pods[i][0], group_pods[i][0])
                      for i in range(n) if not build_ok[i]]
            print(f"  Warning: {vendor_label.lower()}-tests not available on: "
                  f"{', '.join(failed)}. Skipping {vendor_label} tests.")
            continue

        # Detect pip NCCL lib path for runtime LD_LIBRARY_PATH.
        # The system NCCL at /lib64/ may be compiled for a newer CUDA
        # (e.g., 2.29.7+cuda13.2) than the driver supports. The pip
        # nvidia-nccl package is typically matched to the container's CUDA.
        _, nccl_lib_path = _find_nccl_paths(group_pods[0][0])
        if nccl_lib_path and nccl_lib_path != "/usr/lib64":
            print(f"  Using pip NCCL lib: {nccl_lib_path}")

        # Step C: Find mpirun path (already installed by build deps above)
        mpirun_path = find_mpirun(group_pods[0][0])
        if not mpirun_path:
            # Last resort: try install_mpi which searches harder
            mpirun_path = install_mpi(group_pods[0][0])
        if not mpirun_path:
            print(f"  Error: mpirun not available on {group_pods[0][0]}. "
                  f"Skipping {vendor_label} tests.")
            continue
        print(f"  Using mpirun: {mpirun_path}")

        # Setup SSH
        print(f"\n  Setting up SSH cluster for mpirun ...")
        ssh_result = setup_ssh_cluster(group_pods)
        if isinstance(ssh_result, tuple):
            ssh_ok, key_path = ssh_result
        else:
            ssh_ok, key_path = ssh_result, None
        ssh_dir = str(key_path).rsplit("/", 1)[0] if key_path else None
        if not ssh_ok:
            print(f"  Warning: SSH setup incomplete. Attempting to run anyway ...")

        # Run all_reduce_perf
        try:
            rows = run_all_reduce(group_pods, vendor, display_names, gpu_count,
                                  mpirun_path=mpirun_path, key_path=key_path,
                                  nccl_lib_path=nccl_lib_path)
        finally:
            # Always teardown SSH
            stop_sshd_all(group_pods, ssh_dir=ssh_dir)

        if rows:
            print_all_reduce_results(rows, vendor, len(group_pods), total_gpus)

            # Store max busbw as the headline metric
            max_busbw = max(r["busbw_gbs"] for r in rows)
            max_algbw = max(r["algbw_gbs"] for r in rows)
            print(f"\n  Peak Bus BW: {max_busbw:.2f} GB/s")
            print(f"  Peak Algorithm BW: {max_algbw:.2f} GB/s")

            all_results[f"{vendor_label} all_reduce BusBW (GB/s)"] = rows

    return all_results


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-nccl-rccl.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _USAGE = """\
Usage: uv run run-tests-nccl-rccl.py [options]

Run NCCL/RCCL all_reduce_perf collective tests between Kubernetes
inference pods.

Equivalent to: run-tests.sh -t nccl-rccl

Detects GPU vendor per pod (NVIDIA/AMD), builds nccl-tests or rccl-tests
from source if needed, sets up passwordless SSH for mpirun, runs
all_reduce_perf, and displays results.

Options:
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -h, --help            Show this help message.
  -i, --install-deps    Build nccl-tests/rccl-tests from source if missing.
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
    if "-h" in sys.argv or "--help" in sys.argv:
        print(_USAGE)
        sys.exit(0)

    _c = _get_common()
    _cfg = _c._parse_common_args()
    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    _results = run_nccl_rccl(_pods, _display_names)

    # Print summary if there are tabular results
    for _title, _rows in _results.items():
        if isinstance(_rows, list) and _rows:
            print(f"\n{_title} — summary:")
            max_busbw = max(r["busbw_gbs"] for r in _rows)
            max_algbw = max(r["algbw_gbs"] for r in _rows)
            print(f"  Peak Bus BW: {max_busbw:.2f} GB/s")
            print(f"  Peak Algorithm BW: {max_algbw:.2f} GB/s")
