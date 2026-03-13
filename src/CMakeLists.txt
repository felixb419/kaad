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
    ./operators/transpose.cpp
    ./operators/sum.cpp
    ./operators/dot.cpp
    ./operators/mean.cpp
    ./operators/outer.cpp
    ./operators/unary_operators.cpp
    ./operators/matmul.cpp
    ./operators/slice.cpp
    ./operators/binary_operators.cpp
    ./graph/graph.cpp
    ./tensor/tensor_view.cpp
    ./tensor/tensor.cpp
    ./tensor/common.cpp
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
