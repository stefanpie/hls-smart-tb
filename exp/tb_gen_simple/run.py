import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv

from hls_smart_tb.data import BenchmarkCase
from hls_smart_tb.models import build_model_remote_openrouter, normalize_model_name
from hls_smart_tb.runner import build_tb_with_model
from hls_smart_tb.tools import auto_find_vitis_hls_dir
from hls_smart_tb.utils import unwrap

dotenv.load_dotenv()
OPENROUTER_API_KEY = unwrap(os.getenv("OPENROUTER_API_KEY"))

N_JOBS = 64

DIR_CURRENT = Path(__file__).parent

DIR_DESIGNS = DIR_CURRENT.parent.parent / "designs"

DIR_DATA = DIR_CURRENT / "data"
if not DIR_DATA.exists():
    DIR_DATA.mkdir(parents=True)

VITIS_HLS_PATH = unwrap(auto_find_vitis_hls_dir())

MODELS = ["deepseek/deepseek-v4.1-flash"]

K = 5
SAMPLE_INDEXES = list(range(K))

MAX_KERNEL_CALLS = 8

designs = sorted(
    [d for d in DIR_DESIGNS.iterdir() if d.is_dir()],
    key=lambda x: x.name,
)


combs = []
for design in designs:
    for model in MODELS:
        for sample_index in SAMPLE_INDEXES:
            combs.append((design, model, sample_index))


def run_single_comb(comb):
    design, model_name, sample_index = comb
    design_case = BenchmarkCase(design)
    model = build_model_remote_openrouter(model_name, api_key=OPENROUTER_API_KEY)
    run_id = f"{design.name}__{normalize_model_name(model.name)}__{sample_index}"
    work_dir = DIR_DATA / run_id

    build_tb_with_model(
        run_id, design_case, model, work_dir, VITIS_HLS_PATH, MAX_KERNEL_CALLS
    )
    return run_id


with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
    futures = {executor.submit(run_single_comb, comb): comb for comb in combs}
    for future in as_completed(futures):
        design, model_name, sample_index = futures[future]
        try:
            future.result()
        except Exception as e:  # noqa: BLE001
            print(
                f"Error in run_single_comb ({design.name}, {model_name}, {sample_index}): {e}"
            )
