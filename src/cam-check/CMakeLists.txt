cmake_minimum_required(VERSION 3.10)

project(cam-check LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cam-check
    src/cam_check.c
)

target_include_directories(cam-check
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS cam-check
    RUNTIME DESTINATION bin
)