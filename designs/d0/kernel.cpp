// numeric_range_dispatch.cpp
//
// Structured latency design: 3 discrete classes determined by |x| (a simple,
// interpretable predicate), each mapped to a qualitatively different code path.
//
//  |x| < 16         -> "small"  class: closed-form, ~O(1) cheap path
//  16 <= |x| < 256   -> "medium" class: fixed-iteration (8x) but per-iter-expensive path
//  |x| >= 256        -> "large"  class: data-dependent iterative path; still
//                        structured (larger |x| trends toward more iterations)
//                        but not a closed-form count, because integer
//                        div/mod truncation makes the exact iteration count
//                        depend on the bit pattern of x, not just its magnitude.
//                        This gives the "large" class internal spread on top
//                        of the coarse 3-way classification.

#include <ap_int.h>

#define N 256
typedef int data_t;

data_t small_path(data_t x) {
    return x * x + 1;
}

data_t medium_path(data_t x) {
    data_t acc = x;
MEDIUM_LOOP:
    for (int i = 0; i < 8; i++) {
        #pragma HLS PIPELINE II=1
        acc = (acc * 3 + x) >> 1;
    }
    return acc;
}

data_t large_path(data_t x, int &iters_out) {
    data_t val = x;
    int iters = 0;
LARGE_LOOP:
    while (val > 4 || val < -4) {
        val = val / 3 + (val % 5);
        iters++;
        if (iters > 200) break; // safety bound so HLS can still schedule/synth
    }
    iters_out = iters;
    return val + iters;
}

void numeric_range_dispatch_kernel(data_t in[N], data_t out[N], int class_log[N], int iters_log[N]) {
DISPATCH_LOOP:
    for (int i = 0; i < N; i++) {
        data_t x = in[i];
        data_t mag = x < 0 ? -x : x;

        if (mag < 16) {
            out[i] = small_path(x);
            class_log[i] = 0;
            iters_log[i] = 0;
        } else if (mag < 256) {
            out[i] = medium_path(x);
            class_log[i] = 1;
            iters_log[i] = 8;
        } else {
            int iters;
            out[i] = large_path(x, iters);
            class_log[i] = 2;
            iters_log[i] = iters;
        }
    }
}