add_executable(test_binary_broadcasts test_binary_broadcasts.cpp)
target_link_libraries(test_binary_broadcasts PRIVATE kaad)
add_test(NAME test_binary_broadcasts COMMAND test_binary_broadcasts)

add_executable(test_dot_product test_dot_product.cpp)
target_link_libraries(test_dot_product PRIVATE kaad)
add_test(NAME test_dot_product COMMAND test_dot_product)

add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE kaad)
add_test(NAME test_matmul COMMAND test_matmul)

add_executable(test_mean test_mean.cpp)
target_link_libraries(test_mean PRIVATE kaad)
add_test(NAME test_mean COMMAND test_mean)

add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose PRIVATE kaad)
add_test(NAME test_transpose COMMAND test_transpose)
