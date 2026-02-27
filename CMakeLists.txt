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
option(KAAD_BUILD_TESTS "Build tests" ON)
option(KAAD_BUILD_EXAMPLES "Build examples" ON)

# Add subdirectories
add_subdirectory(src)

if (KAAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (KAAD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
