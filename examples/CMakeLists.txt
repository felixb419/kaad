add_executable(broadcast_add broadcast_add.cpp)
target_link_libraries(broadcast_add PRIVATE kaad)

add_executable(dot_product dot_product.cpp)
target_link_libraries(dot_product PRIVATE kaad)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE kaad)

add_executable(transpose transpose.cpp)
target_link_libraries(transpose PRIVATE kaad)

