from pathlib import Path


def build_prompt_generate_tb(
    source_files: list[Path], top_fn: str, max_kernel_calls: int
) -> str:
    source_blocks = []
    for fp in source_files:
        source_blocks.append(f"### {fp.name}\n```cpp\n{fp.read_text()}\n```")
    source_section = "\n\n".join(source_blocks)

    return f"""You will write a C++ testbench for an already-implemented HLS kernel. The testbench is used to measure the kernel's latency distribution in C/RTL co-simulation.

## Goal

The testbench must call the top-level kernel `{top_fn}` on a small set of inputs whose co-simulated latencies (clock cycles per call) cover the full range of behavior the kernel can exhibit, from the fastest possible call to the slowest possible call, with several distinct values in between.

Each call to `{top_fn}` in `main()` is one data point: co-simulation reports the latency of every call separately. The aim is for those per-call latencies to be as spread out as possible. A set of inputs that all take about the same number of cycles is a failure, even if every input is valid.

Latency can depend on the inputs through data-dependent loop trip counts, early exits, branches that select paths of different cost, data-dependent iteration counts (e.g. convergence loops), and data-dependent stream or FIFO traffic.

## How to choose inputs

1. Read the kernel and work out which properties of the input change the number of cycles (loop bounds read from data, `break`/early-exit conditions, branch conditions, number of elements pushed or popped on streams, and so on).
2. Construct inputs that drive those properties to their extremes: the minimum-latency case (take every early exit, cheapest branches, empty or minimal work) and the maximum-latency case (no early exits, most expensive branches, full trip counts, worst-case convergence).
3. Add intermediate inputs so the latencies are spread across the range rather than clustered at the two ends. Where several mechanisms affect latency, vary them independently (e.g. many cheap iterations vs few expensive ones).
4. Build inputs deterministically from the kernel's logic. Do not rely on random values to hit the extremes.

## Hard requirements

- Use at most {max_kernel_calls} calls to `{top_fn}` in total. Co-simulation cost grows with total cycles, so every call must earn its place by exercising a different latency. Do not repeat inputs that give the same latency.
- Call `{top_fn}` once per input, directly from `main()`. Do not call it from other threads or from helper code that hides the call count.
- `main()` must return 0 in every case. Do not check outputs, do not `assert`, and do not return non-zero or call `exit` with an error. Functional correctness is not being tested; the testbench only needs to compile, run to completion, and exercise the kernel.
- Every input must be valid for the kernel: respect array sizes and value ranges the kernel assumes, so there is no out-of-bounds access or undefined behavior. Use `static` or heap storage for large arrays.
- If the kernel keeps state between calls (static or global variables), make each call's latency independent of the call order by re-initializing that state or ordering calls deliberately. If it takes streams, fill each stream completely before each call and drain it afterwards.
- The testbench must be self-contained and deterministic: no file I/O, no stdin, no command-line arguments, no time or `rand()` dependence.
- Include every header you need (for example `ap_int.h`, `ap_fixed.h`, `hls_stream.h`), and use the same types as the kernel.
- The testbench is compiled and linked as C++ together with the kernel sources. Declare the kernel with normal C++ linkage exactly as defined in the source (e.g. `void {top_fn}(...);`), or include the kernel's header if one is provided. Do NOT use `extern "C"` unless the kernel source itself does. Do not define `{top_fn}` yourself and do not modify the kernel.

## Kernel sources

Top-level function: `{top_fn}`

{source_section}

## Output format

Respond with exactly one markdown code block, opened with ```cpp and closed with ```, containing the complete contents of `tb.cpp`. Put short explanations of why each input was chosen in `//` comments inside the code. Write no text before or after the code block, and do not use triple backticks anywhere else in your response.
"""
