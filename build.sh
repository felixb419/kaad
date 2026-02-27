#!/usr/bin/env bash
set -e # exit on first error

# Create build directory
mkdir -p build
cd build || exit 1 # exit if cd fails

# Configure CMake and set build type
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build everything
cmake --build .

ctest --output-on-failure
