#!/bin/bash

ARCH=$(uname -m)

export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/$ARCH-linux-gnu:$LD_LIBRARY_PATH"

ROOT_DIR="$HOME/local"
mkdir -p "$ROOT_DIR"
GDR_HOME="$ROOT_DIR/gdrcopy"
UCX_HOME="$ROOT_DIR/ucx"
export PATH="$GDR_HOME/bin:$UCX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GDR_HOME/lib:$UCX_HOME/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$HOME/.local/lib/python3/dist-packages:$PYTHONPATH"