cmake_minimum_required(VERSION 3.11.4)

project(stinger_measurements)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stinger_measurements main.cpp)

find_package(OpenMP REQUIRED)
find_package(dhb 1.0.0 CONFIG REQUIRED)
find_package(gdsb REQUIRED)

target_link_libraries(stinger_measurements PRIVATE dhb::dhb OpenMP::OpenMP_CXX gdsb::gdsb stdc++fs)

target_include_directories(stinger_measurements BEFORE PUBLIC ${STINGER_COMPILE_DIR}/include)

set(STINGER_LIB ${STINGER_COMPILE_DIR}/lib)

find_library(STINGER_UTILS stinger_utils PATH ${STINGER_LIB})
find_library(STINGER_CORE stinger_core PATH ${STINGER_LIB})


# target_link_libraries(stinger_measurements PRIVATE asan)
target_link_libraries(stinger_measurements PRIVATE ${STINGER_UTILS} ${STINGER_CORE})

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${STINGER_LIB}")

install(TARGETS stinger_measurements DESTINATION bin)
