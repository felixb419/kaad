cmake_minimum_required(VERSION 3.21)

project(kaad
    VERSION 0.9
    DESCRIPTION "Kind Alright Auto Differentiation"
    LANGUAGES CXX
)

# ---- Global Settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXTENSIONS OFF)

# Option flags
option(ENABLE_SANITIZERS "Enable ASan + UBSan" ON)
option(KAAD_BUILD_TESTS "Build tests" ON)
option(KAAD_BUILD_EXAMPLES "Build examples" ON)

# Add subdirectories
add_subdirectory(src)

if (ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    add_link_options(
        -fsanitize=address,undefined
    )
endif()

if (KAAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (KAAD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
