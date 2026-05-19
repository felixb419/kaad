add_library(kaad
    ./exceptions.cpp
    ./graph/graph.cpp
    ./graph/node_handle.cpp
    ./operations/broadcast.cpp
    ./operations/dot_product.cpp
    ./operations/matmul.cpp
    ./operations/outer.cpp
    ./operations/reduce.cpp
    ./operations/slice.cpp
    ./operations/transpose.cpp
    ./operators/binary_operators.cpp
    ./operators/dot.cpp
    ./operators/input.cpp
    ./operators/matmul.cpp
    ./operators/mean.cpp
    ./operators/outer.cpp
    ./operators/slice.cpp
    ./operators/sum.cpp
    ./operators/transpose.cpp
    ./operators/unary_operators.cpp
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
