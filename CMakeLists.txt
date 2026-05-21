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
option(USE_CLANG_TIDY_FIX "Enable clang-tidy --fix" OFF)
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

    set(CLANG_TIDY_ARGS
        -header-filter=.
    )

    if (USE_CLANG_TIDY_FIX)
        list(APPEND CLANG_TIDY_ARGS -fix)    
    endif()

    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};${CLANG_TIDY_ARGS}")
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


# Install logic
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS kaad

    EXPORT kaadTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT kaadTargets
    FILE kaadTargets.cmake
    NAMESPACE kaad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kaad
)

set(CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/kaad)

configure_package_config_file(
    cmake/kaadConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/kaadConfig.cmake
    INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/kaadConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)


install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/kaadConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/kaadConfigVersion.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
)
