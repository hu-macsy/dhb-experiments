cmake_minimum_required(VERSION 3.11.4)

project(graph_analysis)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(graph_analysis main.cpp)

find_package(OpenMP REQUIRED)
find_package(dhb REQUIRED)
find_package(gdsb REQUIRED)

# For address sanitizing link to "asan"
target_link_libraries(graph_analysis PRIVATE dhb::dhb gdsb::gdsb OpenMP::OpenMP_CXX stdc++fs)

install(TARGETS graph_analysis DESTINATION bin)
