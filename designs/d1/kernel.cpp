// filter_refine_two_knob.cpp
//
// Structured 2D latency surface from two orthogonal, interpretable knobs:
//
//  Knob A: fraction of elements that pass a data-DERIVED filter (threshold =
//          mean of the array, not a hard-coded constant) -> controls HOW MANY
//          elements take the expensive path. This depends on the shape of the
//          input distribution, not any single element.
//  Knob B: Newton-Raphson convergence difficulty for sqrt(x) on each element
//          that passes the filter -> controls HOW EXPENSIVE each passing
//          element is, based on how far the fixed initial guess is from the
//          true root.
//
// Total latency ~= (#elements passing filter) * (avg iterations among them).
// Two inputs can land at similar total latency through different mechanisms
// (many-but-quick vs few-but-slow) -- useful if you want to test whether an
// estimator is keying on the right structure rather than just total latency.

#include <ap_fixed.h>

#define N 512
typedef ap_fixed<32,16> fx_t;

fx_t newton_sqrt(fx_t x, int &iters_out) {
    fx_t y = x > fx_t(1) ? (x >> 1) : fx_t(1); // crude fixed initial guess
    int iters = 0;
NR_LOOP:
    for (int i = 0; i < 32; i++) {
        fx_t err = y * y - x;
        fx_t abs_err = err < fx_t(0) ? fx_t(-err) : err;
        if (abs_err < fx_t(0.001)) break; // data-dependent early exit
        y = (y + x / y) >> 1;
        iters++;
    }
    iters_out = iters;
    return y;
}

void filter_refine_two_knob_kernel(fx_t in[N], fx_t out[N], int pass_log[N], int iters_log[N]) {
    // Knob A: data-derived threshold (mean), computed from the input itself.
    fx_t sum = 0;
SUM_LOOP:
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        sum += in[i];
    }
    fx_t mean = sum / fx_t(N);

REFINE_LOOP:
    for (int i = 0; i < N; i++) {
        fx_t x = in[i];
        if (x > mean) {
            int iters;
            out[i] = newton_sqrt(x, iters);
            pass_log[i] = 1;
            iters_log[i] = iters;
        } else {
            out[i] = x * fx_t(0.5);
            pass_log[i] = 0;
            iters_log[i] = 0;
        }
    }
}