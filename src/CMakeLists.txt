add_library(kaad
    ./exceptions.cpp
    ./graph/graph.cpp
    ./graph/node_handle.cpp
    ./graph/operators/binary_operators.cpp
    ./graph/operators/dot.cpp
    ./graph/operators/input.cpp
    ./graph/operators/matmul.cpp
    ./graph/operators/mean.cpp
    ./graph/operators/outer.cpp
    ./graph/operators/slice.cpp
    ./graph/operators/sum.cpp
    ./graph/operators/transpose.cpp
    ./graph/operators/unary_operators.cpp
    ./operations/broadcast.cpp
    ./operations/dot_product.cpp
    ./operations/matmul.cpp
    ./operations/outer.cpp
    ./operations/reduce.cpp
    ./operations/slice.cpp
    ./operations/transpose.cpp
    ./tensor/print_tensor.cpp
    ./tensor/tensor.cpp
    ./tensor/tensor_view.cpp
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
