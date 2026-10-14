#!/usr/bin/env bash
# Print the status of every run in ./data. Read-only: it never modifies anything.
# Usage: ./watch_exp.sh            (print once)
#        watch -n 30 ./watch_exp.sh (refresh every 30 s)

cd "$(dirname "$0")/data" || exit 1

for d in */; do
    d=${d%/}
    if [ -f "$d/run_data.json" ]; then
        # run_data.json is written when a run ends (success or failure)
        jq -r --arg d "$d" '
            [ $d,
              (if .can_run_llm == false then "LLM_FAIL"
               elif .can_extract_code_from_prompt == false then "EXTRACT_FAIL"
               elif .csim_ok != true then "CSIM_FAIL"
               elif .cosim_ok == true then "DONE"
               else "COSIM_FAIL" end),
              ((.data_cosim.transaction_latencies // [])
               | "calls=\(length) min=\(min // "-") max=\(max // "-")")
            ] | @tsv' "$d/run_data.json"
    else
        # still running (or died before writing run_data.json): guess stage from files
        s=llm
        [ -f "$d/llm_response.txt" ] && s=extracting
        [ -f "$d/tb.cpp" ] && s=csim
        [ -n "$(find "$d" -maxdepth 1 -name 'vitis_hls_cosim_tool__*' -print -quit)" ] && s=csynth+cosim
        printf '%s\t%s\t-\n' "$d" "RUNNING($s)"
    fi
done | column -t -s$'\t'
