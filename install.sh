#!/usr/bin/env bash

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

cmake --install build --prefix ./install
