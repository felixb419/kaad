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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option flags
option(ENABLE_SANITIZERS "Enable ASan + UBSan" ON)
option(USE_CLANG_TIDY "Enable clang-tidy" ON)
option(USE_IWYU "Enable IWYU" ON)
option(KAAD_BUILD_TESTS "Build tests" ON)
option(KAAD_BUILD_EXAMPLES "Build examples" ON)

if (ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    add_link_options(
        -fsanitize=address,undefined
    )
endif()

if (USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)

    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-header-filter=.")
    endif()
endif()

if(USE_IWYU)
  find_program(IWYU_PATH NAMES include-what-you-use iwyu)

  if(IWYU_PATH)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_PATH}")
  endif()

endif()


# Add subdirectories
add_subdirectory(src)

if (KAAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (KAAD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
