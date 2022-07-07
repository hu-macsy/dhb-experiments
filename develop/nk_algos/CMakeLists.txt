cmake_minimum_required(VERSION 3.11.4)

project(nk_algos)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nk_algos main.cpp)

find_package(OpenMP REQUIRED)

target_link_libraries(nk_algos PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(nk_algos PRIVATE stdc++fs)

find_package(gdsb REQUIRED)
target_link_libraries(nk_algos PRIVATE gdsb::gdsb)

include(FindPkgConfig REQUIRED)

pkg_search_module(NETWORKIT REQUIRED networkit)
target_include_directories(nk_algos BEFORE PUBLIC ${NETWORKIT_INCLUDE_DIRS})
target_link_libraries(nk_algos PRIVATE ${NETWORKIT_LDFLAGS})
target_link_libraries(nk_algos PRIVATE ${NETWORKIT_LIBRARIES})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${NETWORKIT_LIBRARY_DIRS}")

find_package(dhb REQUIRED)
target_link_libraries(nk_algos PRIVATE dhb::dhb)

install(TARGETS nk_algos DESTINATION bin)
