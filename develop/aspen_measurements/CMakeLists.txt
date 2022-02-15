cmake_minimum_required(VERSION 3.11.4)

project(aspen_measurements)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcx16 -march=native -DUSEMALLOC -DEDGELONG -DHOMEGROWN")

add_executable(aspen_measurements main.cpp)

find_package(OpenMP REQUIRED)
find_package(dhb 1.0.0 CONFIG REQUIRED)
find_package(gdsb REQUIRED)

target_link_libraries(aspen_measurements PRIVATE dhb::dhb OpenMP::OpenMP_CXX gdsb::gdsb stdc++fs)

target_include_directories(aspen_measurements BEFORE PUBLIC ${ASPEN_COMPILE_DIR})
target_compile_options(aspen_measurements PRIVATE -mcx16)
target_compile_definitions(aspen_measurements PRIVATE EDGELONG USEMALLOC)

GET_DIRECTORY_PROPERTY(output INCLUDE_DIRECTORIES)
MESSAGE(STATUS ${output})


install(TARGETS aspen_measurements DESTINATION bin)
