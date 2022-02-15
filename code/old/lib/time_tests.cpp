#include "utilities.h"
#include "get_time.h"
#include "time_operations.h"
#include "sequence_ops.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <iostream>
#include <ctype.h>
#include <math.h>
#include <limits>
#include <vector>
#include <algorithm>

using namespace std;

size_t str_to_int(char* str) {
    return strtol(str, NULL, 10);
}

void report_time(double t, string name) {
  cout << name << " : " << t << endl;
}

template<typename F>
vector<double> repeat(size_t n, size_t rounds, F test) {
  vector<double> R;
  for (size_t i=0; i < rounds; i++) R.push_back(test(n));
  return R;
}

template<typename F>
double reduce(vector<double> V, F f) {
  double x = V[0];
  for (size_t i=1; i < V.size(); i++) x = f(x,V[i]);
  return x;
}

double median(vector<double> V) {
  std::sort(V.begin(),V.end());
  if (V.size()%2 == 1)
    return V[V.size()/2];
  else
    return (V[V.size()/2] + V[V.size()/2 - 1])/2.0;
}

double sumf(double a, double b) {return a+ b;};
double minf(double a, double b) {return (a < b) ? a : b;};
double maxf(double a, double b) {return (a > b) ? a : b;};

template<typename F>
bool run_multiple(size_t n, size_t rounds, size_t bytes_per_elt,
		  string name, F test, bool half_length=1, string x="bw") {
  vector<double> t = repeat(n, rounds, test);
  
  double mint = reduce(t, minf);
  double maxt = reduce(t, maxf);
  double med = median(t);
  double rate = n/mint;
  double l=n;
  double tt;
  if (half_length)
    do {
      l = round(l * .8);
      tt = reduce(repeat(l, rounds, test),minf);
    } while (tt != 0.0 && l/tt > rate/2 && l > 1);

  double bandwidth = rate * bytes_per_elt / 1e9;

  cout << name << std::setprecision(3)
       << ": r=" << rounds
       << ", med=" << med
       << " (" << mint << "," << maxt << "), "
       << "hlen=" << round(l) << ", " 
       << x << " = " << bandwidth
       << endl;
  return 1;
}
	 
double pick_test(size_t id, size_t n, size_t rounds,
		 bool half_length) {
  switch (id) {
  case 0:  
    return run_multiple(n,rounds,24,"map long", t_map<long>, half_length);
  case 1:  
    return run_multiple(n,rounds,16,"tabulate long",t_tabulate<long>, half_length);
  case 2:  
    return run_multiple(n,rounds,8,"reduce add long", t_reduce_add<long>, half_length);
  case 3:  
    return run_multiple(n,rounds,32,"scan add long", t_scan_add<long>, half_length);
  case 4:  
    return run_multiple(n,rounds,18,"pack long", t_pack<long>, half_length);
  case 5:  
    return run_multiple(n,rounds,88,"gather long", t_gather<long>, half_length);
  case 6:  
    return run_multiple(n,rounds,136,"scatter long", t_scatter<long>, half_length);
  case 7:  
    return run_multiple(n,rounds,136,"write add long", t_write_add<long>, half_length);
  case 8:  
    return run_multiple(n,rounds,136,"write min long", t_write_min<long>, half_length);
  case 9: 
    return run_multiple(n,rounds,1,"count sort 8bit long", t_count_sort_8<long>, half_length, "Gelts/sec");
  case 10:  
    return run_multiple(n,rounds,1,"random shuffle long", t_shuffle<long>, half_length, "Gelts/sec");
  case 11: 
    return run_multiple(n,rounds,1,"histogram int", t_histogram<int>, half_length, "Gelts/sec");
  case 12: 
    return run_multiple(n,rounds,1,"histogram same int", t_histogram_same<int>, half_length, "Gelts/sec");
  case 13: 
    return run_multiple(n,rounds,1,"histogram few int", t_histogram_few<int>, half_length, "Gelts/sec");
  case 14: 
    return run_multiple(n,rounds,1,"integer sort<int,int>", t_integer_sort_pair<uint>, half_length, "Gelts/sec");
  case 15: 
    return run_multiple(n,rounds,1,"integer sort int", t_integer_sort<uint>, half_length, "Gelts/sec");
  case 16: 
    return run_multiple(n,rounds,1,"integer sort 128 bits", t_integer_sort_128, half_length, "Gelts/sec");
  case 17: 
    return run_multiple(n,rounds,1,"sort long", t_sort<long>, half_length, "Gelts/sec");
  case 18: 
    return run_multiple(n,rounds,1,"sort int", t_sort<int>, half_length, "Gelts/sec");
  case 19: 
    return run_multiple(n,rounds,1,"sort 128 bits", t_sort<__int128>, half_length, "Gelts/sec");
  case 20: 
    return run_multiple(n,rounds,24,"merge long", t_merge<long>, half_length);
  case 21: 
    return run_multiple(n,rounds,24 + 5*80,"mat vect mult", t_mat_vec_mult<size_t,double>, half_length);
  case 22:  
    return run_multiple(n,rounds,132,"scatter int", t_scatter<uint>, half_length);
  case 23:  
    return run_multiple(n,rounds,1,"merge sort long", t_merge_sort<long>, half_length, "Gelts/sec");
  case 24: 
    return run_multiple(n,rounds,1,"count sort 2bit long", t_count_sort_2<long>, half_length, "Gelts/sec");
  case 25: 
    return run_multiple(n,rounds,32,"split3 long", t_split3<long>, half_length);
  case 26:  
    return run_multiple(n,rounds,1,"quicksort long", t_quicksort<long>, half_length, "Gelts/sec");
    
  default: 
    assert(false);
    return 0.0 ;
  }
}

int main (int argc, char *argv[]) {
  if (argc > 5) {
    fprintf(stderr, "time_test <n> <rounds> [half_length] [<test_num>]\n");
    exit(1);
  }
  int num_tests = 27;
  int test_num = -1;
  bool half_length = 0;
  if (argc > 3) 
    half_length = str_to_int(argv[3]) != 0;
  if (argc > 4)
    test_num = str_to_int(argv[4]);
  size_t n       = str_to_int(argv[1]);
  size_t rounds  = str_to_int(argv[2]);

  cout << "n = " << n << endl;
  cout << "rounds = " << rounds << endl;
  cout << "num threads = " << num_workers() << endl;
  if (half_length) cout << "half length on" << endl;
  else cout << "half length off" << endl;

  if (test_num == -1)
    for (int i=0; i < num_tests; i++)
      pick_test(i,n,rounds,half_length);
  else pick_test(test_num,n,rounds,half_length);
}
  
  

