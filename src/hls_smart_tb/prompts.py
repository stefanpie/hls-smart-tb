from pathlib import Path


def build_prompt_generate_tb(source_files: list[Path]) -> str:
    source_blocks = []
    for fp in source_files:
        source_blocks.append(f"### {fp.name}\n```cpp\n{fp.read_text()}\n```")
    source_section = "\n\n".join(source_blocks)

    return f"""You will generate a C++ testbench for an already-implemented HLS design.

The testbench must explicitly call the HLS top-level kernel on a set of inputs. Functional correctness of the kernel outputs does not need to be checked. The testbench only needs to compile and exercise the kernel.

The testbench is compiled and linked as C++ together with the kernel sources. Declare the top-level kernel with normal C++ linkage that matches the source (e.g. `void top(...);`). Do NOT use `extern "C"` on the kernel prototype or definition unless the provided kernel source itself uses `extern "C"`.

Choose inputs that produce a wide distribution of kernel execution latency, including extreme cases. This matters when the kernel has data-dependent control flow or input-dependent latency. The testbench will later be run in C/RTL co-simulation to measure the exact latency of each input.

Keep the input set small. A large number of kernel calls makes co-simulation too expensive. Prefer a compact set of inputs that still covers a wide latency range.

The current HLS source files are:

{source_section}

Output a single markdown code block containing the testbench C++. It will be written to `tb.cpp` and run in a co-simulation flow together with the source files.
"""
