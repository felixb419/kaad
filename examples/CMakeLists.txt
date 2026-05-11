set(EXAMPLE_TARGETS
    broadcast_add
    dot_product
    matmul
    transpose
)

foreach(EXAMPLE_TARGET ${EXAMPLE_TARGETS})
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_TARGET}.cpp)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE kaad)
    add_test(NAME ${EXAMPLE_TARGET} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES CXX_CLANG_TIDY "")
endforeach()
