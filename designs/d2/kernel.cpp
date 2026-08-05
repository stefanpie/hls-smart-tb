// multi_root_basin.cpp
//
// A more non-trivial structured design: Newton-Raphson on a cubic with
// three real roots, f(y) = y^3 - y, roots at -1, 0, +1.
//
// Every input doubles as the initial guess. This gives you:
//
//   - A discrete latency CLASS: which root the iteration converges to
//     (3 classes total). This is a genuine structural property of the input,
//     not noise -- classic Newton fractal basins of attraction.
//   - A continuous sub-structure WITHIN each class: iteration count varies
//     smoothly as a function of distance from the basin boundary and from
//     the repelling fixed point near y=0 (slow near boundaries/repellers,
//     fast in the interior of a basin).
//   - Non-monotonicity: basin boundaries are not simple thresholds -- two
//     inputs that are numerically close can belong to different classes,
//     while two inputs that are far apart can belong to the same class.
//     This is "structured but non-trivial": every input's class and cost
//     are fully explainable post-hoc (you can look up which root it
//     converged to and how many iterations it took), but the input->class
//     mapping is not a simple interval partition the way it is in the
//     numeric-range-dispatch design.
//
// Use iters_log + out (final root value) together to label each input by
// class after the run, and to build input corpora that deliberately sample
// near basin boundaries (slow, high-variance) vs. deep inside a basin
// (fast, low-variance).

#include <ap_fixed.h>

#define N 512
typedef ap_fixed<32,16> fx_t;

int newton_cubic_class(fx_t y0, fx_t &root_out) {
    fx_t y = y0;
    int iters = 0;
CUBIC_LOOP:
    for (int i = 0; i < 40; i++) {
        fx_t f  = y*y*y - y;
        fx_t fp = fx_t(3)*y*y - fx_t(1);
        // guard near the critical points of f' (y = +-1/sqrt(3)) to avoid
        // divide-by-near-zero; this guard itself is a fixed, cheap branch,
        // not part of the data-dependent cost.
        if (fp > fx_t(-0.01) && fp < fx_t(0.01)) fp = fx_t(0.01);
        fx_t y_next = y - f / fp;
        fx_t delta = y_next - y;
        fx_t abs_delta = delta < fx_t(0) ? fx_t(-delta) : delta;
        y = y_next;
        iters++;
        if (abs_delta < fx_t(0.0005)) break; // data-dependent convergence
    }
    root_out = y;
    return iters;
}

void multi_root_basin_kernel(fx_t in[N], fx_t out[N], int iters_log[N]) {

BASIN_LOOP:
    for (int i = 0; i < N; i++) {
        fx_t root;
        int iters = newton_cubic_class(in[i], root);
        out[i] = root;
        iters_log[i] = iters;
    }
}