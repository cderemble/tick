%{
#include "performance_test.h"
%}

double test_sum_double_pointer(ulong size, ulong n_loops);
double test_sum_ArrayDouble(ulong size, ulong n_loops);
double test_sum_SArray_shared_ptr(ulong size, ulong n_loops);
double test_sum_VArray_shared_ptr(ulong size, ulong n_loops);
