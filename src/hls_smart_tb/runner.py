import json
import shutil
from pathlib import Path
from typing import Any

from hls_smart_tb.data import SOURCE_FILE_EXTENSIONS, BenchmarkCase
from hls_smart_tb.models import Model
from hls_smart_tb.prompt_processing import extract_code_markdown_from_llm_outout
from hls_smart_tb.prompts import build_prompt_generate_tb
from hls_smart_tb.tools import (
    VitisHLSCSimTool,
)


def dump_to_run_data(run_data_fp: Path, data: dict) -> None:
    run_data_fp.write_text(json.dumps(data, indent=4))


def build_tb_with_model(
    run_id: str,
    design_case: BenchmarkCase,
    model: Model,
    work_dir: Path,
    vitis_hls_path: Path,
) -> None:
    print(f"[{run_id}] Starting run...")

    run_data_fp = work_dir / "run_data.json"
    data: dict[str, Any] = {}

    print(f"[{run_id}] Setting up work dir...")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    dir_source = work_dir / "sources"
    dir_source.mkdir()
    shutil.copytree(design_case.design_dir, dir_source, dirs_exist_ok=True)

    source_files = [
        fp
        for fp in dir_source.rglob("*")
        if fp.is_file() and fp.suffix in SOURCE_FILE_EXTENSIONS
    ]

    top_file = work_dir / "top.txt"
    top_file.write_text(design_case.top_fn)

    print(f"[{run_id}] Building LLM prompt...")
    prompt = build_prompt_generate_tb(source_files)

    fp_prompt = work_dir / "llm_prompt.txt"
    fp_prompt.write_text(prompt)

    data["llm_prompt"] = prompt

    print(f"[{run_id}] Running LLM...")
    try:
        r = model.llm.prompt(prompt)
        r._force()
        r_txt = r.text()
        r_json = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[{run_id}] Error running LLM: {e}")
        data["can_run_llm"] = False
        dump_to_run_data(run_data_fp, data)
        return
    data["can_run_llm"] = True
    data["llm_response"] = r_txt
    data["llm_response_json"] = r_json

    fp_llm_output = work_dir / "llm_response.txt"
    fp_llm_output.write_text(r_txt)

    print(f"[{run_id}] Extracting code from LLM response...")
    try:
        code_from_prompt = extract_code_markdown_from_llm_outout(r_txt)
        if len(code_from_prompt) != 1:
            raise ValueError(
                f"Expected 1 code block in the response, got {len(code_from_prompt)}"
            )
        extracted_tb_cpp = next(iter(code_from_prompt.values()))
        tb_cpp_path = work_dir / "tb.cpp"
        tb_cpp_path.write_text(extracted_tb_cpp)
    except Exception as e:  # noqa: BLE001
        print(f"[{run_id}] Error extracting code from LLM response: {e}")
        data["can_extract_code_from_prompt"] = False
        dump_to_run_data(run_data_fp, data)
        return
    data["can_extract_code_from_prompt"] = True
    data["tb_cpp"] = extracted_tb_cpp

    print(f"[{run_id}] Running C simulation...")
    csim_tool = VitisHLSCSimTool(vitis_hls_path)
    r_compile, r_run = csim_tool.run(
        work_dir,
        source_files + [tb_cpp_path],
        hls_top_function=design_case.top_fn,
    )
    data["csim_compile_output"] = r_compile.as_dict()
    data["csim_run_output"] = r_run.as_dict() if r_run is not None else None

    dump_to_run_data(run_data_fp, data)
    print(f"[{run_id}] Finished run!")
