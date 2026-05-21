#!/bin/bash
# install_nixl_nonroot.sh — Install NIXL and dependencies in non-root containers
#
# Designed for OpenShift/Kubernetes pods that run as non-root with:
#   - HOME=/ (not writable)
#   - /usr/local not writable
#   - No sudo / no apt-get / no dnf
#   - Only /tmp is writable
#
# Usage:
#   bash install_nixl_nonroot.sh [--force]
#   source /tmp/local/env.sh   # after install, to set PATH/LD_LIBRARY_PATH

set -euo pipefail

# --- Version Configuration (update these as needed) ---
GDRCOPY_VERSION="${GDRCOPY_VERSION:-2.5.2}"
UCX_VERSION="${UCX_VERSION:-1.20.1}"
NIXL_VERSION="${NIXL_VERSION:-1.1.0}"

FORCE=false
if [ "${1:-}" == "--force" ]; then
    FORCE=true
fi

export HOME=/tmp
ARCH=$(uname -m)

# All installs go under /tmp/local
ROOT_DIR="/tmp/local"
mkdir -p "$ROOT_DIR"

GDR_HOME="$ROOT_DIR/gdrcopy"
UCX_HOME="$ROOT_DIR/ucx"
NIXL_HOME="$ROOT_DIR/nixl"
PIP_BIN="/tmp/.local/bin"

export CUDA_HOME=/usr/local/cuda
export PATH="$PIP_BIN:$GDR_HOME/bin:$UCX_HOME/bin:$NIXL_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${GDR_HOME}/lib:${UCX_HOME}/lib:${NIXL_HOME}/lib:${NIXL_HOME}/lib64:${LD_LIBRARY_PATH:-}"

TEMP_DIR="/tmp/nixl_installer"
mkdir -p "$TEMP_DIR"

# --- Step 1: Python build tools via pip --user (HOME=/tmp) ---
echo "Installing Python build tools via pip..."
pip install --user meson ninja pybind11 cmake 2>/dev/null \
  || pip3 install --user meson ninja pybind11 cmake 2>/dev/null \
  || python3 -m pip install --user meson ninja pybind11 cmake
export PATH="$PIP_BIN:$PATH"
echo "  meson: $(meson --version 2>/dev/null || echo 'not found')"
echo "  cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"

# --- Step 2: git (download static binary if not available) ---
if ! command -v git &>/dev/null || [ "$FORCE" = true ]; then
    echo "Installing git static binary..."
    mkdir -p "$PIP_BIN"
    # Use git from the container distro's package, or download a portable version
    if command -v apt-get &>/dev/null && touch /var/lib/dpkg/lock-frontend 2>/dev/null; then
        apt-get update -qq && apt-get install -y -qq git
    else
        # Download git source tarball and extract just the git binary is complex;
        # instead use the git-core from conda-forge as static binary
        # Fallback: use curl to download tarballs directly (no git needed for our use case)
        echo "  git not available; will use curl for tarball downloads instead"
    fi
fi

# --- Step 3: gdrcopy (skip insmod if /dev/gdrdrv exists) ---
if [ ! -e "$GDR_HOME/lib/libgdrapi.so" ] || [ "$FORCE" = true ]; then
    if [ -e "/dev/gdrdrv" ]; then
        echo "Found /dev/gdrdrv (kernel module loaded on host)"
    fi
    echo "Installing gdrcopy v${GDRCOPY_VERSION} to $GDR_HOME..."
    cd "$TEMP_DIR"
    curl -sL "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz" -o "gdrcopy-v${GDRCOPY_VERSION}.tar.gz"
    tar xzf "gdrcopy-v${GDRCOPY_VERSION}.tar.gz" && rm "gdrcopy-v${GDRCOPY_VERSION}.tar.gz"
    cd "gdrcopy-${GDRCOPY_VERSION}"
    make prefix="$GDR_HOME" CUDA="$CUDA_HOME" all install 2>&1 | tail -5
    # Skip insmod.sh — /dev/gdrdrv is already provided by the host
    echo "  gdrcopy installed (skipped insmod — host kernel module already loaded)"
    cd "$TEMP_DIR"
else
    echo "gdrcopy already installed at $GDR_HOME"
fi

# --- Step 4: UCX ---
if ! command -v ucx_info &>/dev/null || [ "$FORCE" = true ]; then
    echo "Installing UCX v${UCX_VERSION} to $UCX_HOME..."
    cd "$TEMP_DIR"
    curl -sL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o "ucx-${UCX_VERSION}.tar.gz"
    tar xzf "ucx-${UCX_VERSION}.tar.gz" && rm "ucx-${UCX_VERSION}.tar.gz"
    cd "ucx-${UCX_VERSION}"

    # Check for Mellanox NICs
    MLX_OPTS=""
    if command -v lspci &>/dev/null && lspci | grep -qi mellanox; then
        echo "  Mellanox NIC detected"
        MLX_OPTS="--with-rdmacm --with-mlx5-dv --with-ib-hw-tm"
    elif command -v ibstat &>/dev/null; then
        echo "  IB tools detected"
        MLX_OPTS="--with-rdmacm --with-mlx5-dv --with-ib-hw-tm"
    fi

    ./configure --prefix="$UCX_HOME" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda="$CUDA_HOME" \
        --with-dm \
        --with-gdrcopy="$GDR_HOME" \
        --with-verbs \
        --enable-mt \
        $MLX_OPTS 2>&1 | tail -10

    make -j$(nproc) 2>&1 | tail -5
    make install-strip 2>&1 | tail -5
    # Skip ldconfig — we use LD_LIBRARY_PATH
    echo "  UCX installed (using LD_LIBRARY_PATH instead of ldconfig)"
    cd "$TEMP_DIR"
else
    echo "UCX already installed"
fi

# --- Step 5: NIXL ---
if ! command -v nixl_test &>/dev/null || [ "$FORCE" = true ]; then
    echo "Installing NIXL v${NIXL_VERSION} to $NIXL_HOME..."
    cd "$TEMP_DIR"
    curl -sL "https://github.com/ai-dynamo/nixl/archive/refs/tags/v${NIXL_VERSION}.tar.gz" -o "nixl-v${NIXL_VERSION}.tar.gz"
    tar xzf "nixl-v${NIXL_VERSION}.tar.gz" && rm "nixl-v${NIXL_VERSION}.tar.gz"
    cd "nixl-${NIXL_VERSION}"
    meson setup build --prefix="$NIXL_HOME" -Ducx_path="$UCX_HOME" 2>&1 | tail -10
    cd build
    ninja 2>&1 | tail -5
    ninja install 2>&1 | tail -5
    echo "  NIXL installed"
    cd "$TEMP_DIR"
else
    echo "NIXL already installed"
fi

# --- Step 6: Write env.sh for sourcing ---
ENV_FILE="$ROOT_DIR/env.sh"
cat > "$ENV_FILE" << EOF
#!/bin/bash
# Source this file to set up NIXL environment
export HOME=/tmp
export CUDA_HOME=/usr/local/cuda
export PATH="$PIP_BIN:$GDR_HOME/bin:$UCX_HOME/bin:$NIXL_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$GDR_HOME/lib:$UCX_HOME/lib:$NIXL_HOME/lib:$NIXL_HOME/lib64:$NIXL_HOME/lib/$ARCH-linux-gnu:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/.local/lib/python3/dist-packages:\${PYTHONPATH:-}"
EOF
chmod +x "$ENV_FILE"

echo ""
echo "=== Installation complete ==="
echo "To use: source $ENV_FILE"
echo ""
echo "Versions:"
echo "  gdrcopy: v${GDRCOPY_VERSION}"
echo "  UCX:     v${UCX_VERSION}"
echo "  NIXL:    v${NIXL_VERSION}"
echo ""
echo "Paths:"
echo "  GDR_HOME=$GDR_HOME"
echo "  UCX_HOME=$UCX_HOME"
echo "  NIXL_HOME=$NIXL_HOME"
echo "  PIP tools: $PIP_BIN"
