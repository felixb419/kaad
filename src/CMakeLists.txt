add_library(kaad
    ./graph/nodes/inode.cpp
    ./graph/nodes/transp.cpp
    ./graph/nodes/dot.cpp
    ./graph/nodes/sum_dim.cpp
    ./graph/nodes/batch_matmul.cpp
    ./graph/nodes/mean.cpp
    ./graph/nodes/outer.cpp
    ./graph/nodes/matmul.cpp
    ./graph/nodes/slice.cpp
    ./graph/nodes/mean_dim.cpp
    ./graph/operators/transpose.cpp
    ./graph/operators/sum.cpp
    ./graph/operators/dot.cpp
    ./graph/operators/mean.cpp
    ./graph/operators/outer.cpp
    ./graph/operators/unary_operators.cpp
    ./graph/operators/matmul.cpp
    ./graph/operators/slice.cpp
    ./graph/operators/binary_operators.cpp
    ./graph/computation_graph.cpp
    ./tensor/tensor_view.cpp
    ./tensor/tensor.cpp
    ./exceptions.cpp
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
