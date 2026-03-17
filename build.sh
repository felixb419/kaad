#!/usr/bin/env bash
set -euo pipefail

RELEASE=false
CLEAN=false
FAST=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [--release] [--clean] [--fast] [--help]

  --clean     Delete build/ directory before building
  --release   Build in release mode
  --help      Show this help message
EOF
}

# Parse long flags
while [[ $# -gt 0 ]]; do
    case "$1" in
    --release)
        RELEASE=true

        shift
        ;;
    --clean)
        CLEAN=true
        shift
        ;;
    --fast)
        FAST=true
        shift
        ;;
    --help)
        usage

        exit 0
        ;;
    --) # end of options
        shift
        break
        ;;
    -*)
        echo "Unknown option: $1" >&2
        usage

        exit 1
        ;;
    esac
done

BUILD_DIR=./build

if $CLEAN; then
    rm -rf $BUILD_DIR
fi

# Create build directory
mkdir -p $BUILD_DIR
cd build || exit 1 # exit if cd fails

# Configure CMake and set build type
if $RELEASE; then
    cmake -DCMAKE_BUILD_TYPE=Release ..
else
    cmake -DCMAKE_BUILD_TYPE=Debug ..
fi

# Turn off iwyu, clang-tidy and sanitizers
if $FAST; then
    cmake -DENABLE_SANITIZERS=OFF ..
    cmake -DUSE_CLANG_TIDY=OFF ..
    cmake -DUSE_IWYU=OFF ..
fi

# Build everything
cmake --build .

ctest --output-on-failure
