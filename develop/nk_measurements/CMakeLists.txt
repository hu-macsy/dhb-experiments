cmake_minimum_required(VERSION 3.11.4)

project(nk_measurements)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nk_measurements main.cpp)

# OMP
find_package(OpenMP REQUIRED)
target_link_libraries(nk_measurements PRIVATE OpenMP::OpenMP_CXX)

find_package(gdsb REQUIRED)
target_link_libraries(nk_measurements PRIVATE gdsb::gdsb)

# C++ Filesystem
target_link_libraries(nk_measurements PRIVATE stdc++fs)

include(FindPkgConfig REQUIRED)

# networkit
pkg_search_module(NETWORKIT REQUIRED networkit)
target_include_directories(nk_measurements BEFORE PUBLIC ${NETWORKIT_INCLUDE_DIRS})
target_link_libraries(nk_measurements PRIVATE ${NETWORKIT_LDFLAGS})
target_link_libraries(nk_measurements PRIVATE ${NETWORKIT_LIBRARIES})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${NETWORKIT_LIBRARY_DIRS}")

find_package(dhb REQUIRED)
target_link_libraries(nk_measurements PRIVATE dhb::dhb)

install(TARGETS nk_measurements DESTINATION bin)
