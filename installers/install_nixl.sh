#!/bin/bash

# Install NixL

# Check if Nix is installed
if ! command -v nix &> /dev/null
then
    echo "Nix is not installed. Please install Nix first."
    exit 1
fi

# Install NixL
nix-env -iA nixpkgs.nixl

# Install NixL dependencies
nix-env -iA nixpkgs.python3
nix-env -iA nixpkgs.python3Packages.pip

# Install NixL Python dependencies
pip install -r requirements.txt

# Install NixL shell
nix-shell --run "echo 'NixL installation complete.'"
