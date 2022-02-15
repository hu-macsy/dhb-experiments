#ifndef _PARALLEL_H
#define _PARALLEL_H

// intel cilk+
#if CILK == 1
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
#include <cilk/reducer_opxor.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
#define parallel_for cilk_for
#define parallel_main main
#define parallel_for_1 _Pragma("cilk grainsize = 1") cilk_for
#define parallel_for_256 _Pragma("cilk grainsize = 256") cilk_for

[[maybe_unused]] static int getWorkers() {
  return __cilkrts_get_nworkers();
}

[[maybe_unused]] static int getWorkerNum() {
  return __cilkrts_get_worker_number();
}

// openmp
#elif OPENMP == 1
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for _Pragma("omp parallel for") for
#define parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

[[maybe_unused]] static int getWorkers() { return omp_get_max_threads(); }
[[maybe_unused]] static void setWorkers(int n) { omp_set_num_threads(n); }
[[maybe_unused]] static int getWorkerNum() {
  return omp_get_thread_num();
}

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for for
#define parallel_for_1 for
#define parallel_for_256 for
#define cilk_for for

[[maybe_unused]] static int getWorkers() { return 1; }
[[maybe_unused]] static void setWorkers([[maybe_unused]] int n) { }
[[maybe_unused]] static int getWorkerNum() { return 1; }

#endif

#include <limits.h>

#if defined(LONG)
typedef long intT;
typedef unsigned long uintT;
#define INT_T_MAX LONG_MAX
#define UINT_T_MAX ULONG_MAX
#else
typedef int intT;
typedef unsigned int uintT;
#define INT_T_MAX INT_MAX
#define UINT_T_MAX UINT_MAX
#endif

#endif // _PARALLEL_H
