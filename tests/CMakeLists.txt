set(TEST_TARGETS
    test_call_all_kernels
    test_binary_broadcasts
    test_unary_operations
    test_dot_product
    test_matmul
    test_mean
    test_transpose
)

foreach(TEST_TARGET ${TEST_TARGETS})
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE kaad)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
    set_target_properties(${TEST_TARGET} PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "")
endforeach()
