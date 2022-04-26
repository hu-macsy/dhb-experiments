cmake_minimum_required (VERSION 3.11.4)

project (terrace_measurements)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT TERRACE_PREFIX_DIR)
  message(FATAL_ERROR "Please specify TERRACE_PREFIX_DIR")
endif()

find_package(OpenMP REQUIRED)
find_package(dhb 1.0.0 CONFIG REQUIRED)
find_package(gdsb REQUIRED)
find_library(TERRACE libterrace.so REQUIRED PATHS "${TERRACE_PREFIX_DIR}/lib")

add_executable(terrace_measurements main.cpp)

target_include_directories(terrace_measurements BEFORE PUBLIC "${TERRACE_PREFIX_DIR}/include")
target_compile_definitions(terrace_measurements PRIVATE OPENMP=1 ENABLE_LOCK=1 WEIGHTED=1 VERIFY=0)

target_link_libraries(terrace_measurements PRIVATE ${TERRACE})
target_link_libraries(terrace_measurements PRIVATE dhb::dhb OpenMP::OpenMP_CXX gdsb::gdsb stdc++fs)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${TERRACE_PREFIX_DIR}/lib")

install(TARGETS terrace_measurements DESTINATION bin)
