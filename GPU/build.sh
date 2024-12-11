#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Build code.
nvcc -O3 ray_tracing_cpu.cu -o ${CTR_BUILD_DIR}/ray_tracing_cpu

# Build CPU Raytracer
nvcc -O3 main.cu -o ${CTR_BUILD_DIR}/main
