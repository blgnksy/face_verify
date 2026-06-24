cmake_minimum_required(VERSION 3.16)
project(face_verify VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

configure_file(
    include/face_verify/version.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/face_verify/version.hpp
)

# --- Dependencies ---
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn objdetect calib3d)
find_package(onnxruntime QUIET)

if(NOT onnxruntime_FOUND)
    find_library(ONNXRUNTIME_LIB onnxruntime)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime/core/session
                      onnxruntime
    )
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE}/.."
        )
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install with: apt install libonnxruntime-dev")
    endif()
endif()

# --- Library ---
add_library(face_verify_lib STATIC
    src/capture.cpp
    src/config.cpp
    src/detector.cpp
    src/deep_recognizer.cpp
    src/classical_recognizer.cpp
    src/ensemble.cpp
    src/face_db.cpp
)

target_include_directories(face_verify_lib PUBLIC include)
target_link_libraries(face_verify_lib PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)
# Required for linking the static lib into the PAM shared module
set_target_properties(face_verify_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Main executable ---
add_executable(face_verify src/main.cpp)
target_link_libraries(face_verify PRIVATE face_verify_lib)

# --- PAM module ---
find_library(PAM_LIB pam)
find_path(PAM_INCLUDE security/pam_modules.h)

if(PAM_LIB AND PAM_INCLUDE)
    message(STATUS "Found PAM: ${PAM_LIB}")
    add_library(pam_face_verify MODULE src/pam_face_verify.cpp)
    target_include_directories(pam_face_verify PRIVATE ${PAM_INCLUDE})
    target_link_libraries(pam_face_verify PRIVATE face_verify_lib ${PAM_LIB})
    set_target_properties(pam_face_verify PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        POSITION_INDEPENDENT_CODE ON
    )
    install(TARGETS pam_face_verify LIBRARY DESTINATION /lib/security)
else()
    message(WARNING
        "libpam not found — pam_face_verify.so will not be built. "
        "Install with: apt install libpam-dev")
endif()

# --- Tests ---
enable_testing()

option(COVERAGE "Enable gcov coverage instrumentation" OFF)

if(COVERAGE)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Coverage is most useful with -DCMAKE_BUILD_TYPE=Debug")
    endif()
    find_program(GCOVR gcovr)
    if(NOT GCOVR)
        message(FATAL_ERROR "gcovr not found — install with: pip install gcovr")
    endif()
    target_compile_options(face_verify_lib PRIVATE --coverage -fprofile-abs-path)
    target_link_options(face_verify_lib INTERFACE --coverage)
endif()

foreach(test_name ensemble face_db config classical)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_link_libraries(test_${test_name} PRIVATE face_verify_lib)
    add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()

if(COVERAGE)
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ctest --test-dir ${CMAKE_BINARY_DIR} --output-on-failure -Q
        COMMAND ${GCOVR}
            --root ${CMAKE_SOURCE_DIR}/src
            --object-directory ${CMAKE_BINARY_DIR}
            --exclude-unreachable-branches
            --exclude-throw-branches
            --print-summary
            --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
            --xml ${CMAKE_BINARY_DIR}/coverage/coverage.xml
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests and generating coverage report"
    )
endif()

# --- Install ---
install(TARGETS face_verify RUNTIME DESTINATION bin)
