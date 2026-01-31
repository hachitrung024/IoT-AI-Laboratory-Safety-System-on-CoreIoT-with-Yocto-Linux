cmake_minimum_required(VERSION 3.10)

project(cam-check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cam-check
    src/cam_check.cpp
)

target_include_directories(cam-check
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cam-check
    PRIVATE
        ${OpenCV_LIBS}
)

install(TARGETS cam-check
    RUNTIME DESTINATION bin
)