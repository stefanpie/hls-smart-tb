# simple test that loads benchmakr cases in ../../designs/
# and synthesizes them with vitis HLS

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hls_smart_tb.data import BenchmarkCase
from hls_smart_tb.tools import VitisHLSSynthTool, auto_find_vitis_hls_dir

N_JOBS = 32

DIR_CURRENT = Path(__file__).parent

DIR_DESIGNS = DIR_CURRENT.parent.parent / "designs"

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(parents=True)

vhls_synth = VitisHLSSynthTool(auto_find_vitis_hls_dir())



def synth_design(design_dir):
    design_case_dir = DIR_DATA / design_dir.name
    if design_case_dir.exists():
        shutil.rmtree(design_case_dir)
    shutil.copytree(design_dir, design_case_dir)

    design_case = BenchmarkCase(design_case_dir)

    vhls_synth.run(
        build_dir=design_case_dir / "build",
        source_files=design_case.source_files,
        build_name=design_case.name,
        hls_fpga_part="xczu9eg-ffvb1156-2-e",
        hls_clock_period_ns=5,
        hls_top_function=design_case.top_fn,
        hls_flow_target="vivado",
        hls_unsafe_math=True,
        hls_disable_auto_optimizations=True,
        timeout=60.0 * 8,
    )
    print(f"Design {design_case.name} done.")

design_dirs = [d for d in DIR_DESIGNS.iterdir()]
with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
    futures = {executor.submit(synth_design, design_dir): design_dir for design_dir in design_dirs}
    for future in as_completed(futures):
        # This will re-raise any exceptions from the threads
        try:
            future.result()
        except Exception as e:  # noqa: BLE001
            print(f"Error in design {futures[future].name}: {e}")