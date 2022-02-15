cmake_minimum_required(VERSION 3.11.4)

project(dhb_exp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(dhb_exp main.cpp ${benchmark})

# OpenMP
find_package(OpenMP)
find_package(dhb REQUIRED)
find_package(gdsb REQUIRED)

# For address sanitizing link to "asan"
target_link_libraries(dhb_exp PRIVATE dhb::dhb gdsb::gdsb OpenMP::OpenMP_CXX stdc++fs)

install(TARGETS dhb_exp DESTINATION bin)
