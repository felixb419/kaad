add_library(kaad
    ./exceptions.cpp
    ./functions/batch_matmul.cpp
    ./functions/matmul.cpp
    ./graph/graph.cpp
    ./graph/node_handle.cpp
    ./graph/nodes/batch_matmul.cpp
    ./graph/nodes/dot.cpp
    ./graph/nodes/inode.cpp
    ./graph/nodes/matmul.cpp
    ./graph/nodes/mean.cpp
    ./graph/nodes/mean_dim.cpp
    ./graph/nodes/outer.cpp
    ./graph/nodes/slice.cpp
    ./graph/nodes/sum_dim.cpp
    ./graph/nodes/transp.cpp
    ./operators/binary_operators.cpp
    ./operators/dot.cpp
    ./operators/matmul.cpp
    ./operators/mean.cpp
    ./operators/outer.cpp
    ./operators/slice.cpp
    ./operators/sum.cpp
    ./operators/transpose.cpp
    ./operators/unary_operators.cpp
    ./tensor/print_tensor.cpp
    ./tensor/tensor.cpp
)

# Public headers location
target_include_directories(kaad
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler features attached to target
target_compile_features(kaad PUBLIC cxx_std_20)

# Warning policy
if (MSVC)
    target_compile_options(kaad PRIVATE /W4)
else()
    target_compile_options(kaad PRIVATE -Wall -Wextra -Wpedantic)
endif()
