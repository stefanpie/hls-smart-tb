import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from hls_smart_tb.utils import unwrap


def _find_text(elem: ET.Element | None, *path: str) -> str:
    """Walk child tags in `path` from `elem`, unwrapping each step, and return the text."""
    for tag in path:
        elem = unwrap(unwrap(elem).find(tag))
    return unwrap(unwrap(elem).text)


@dataclass
class DesignHLSSynthData:
    clock_period: float

    latency_best_cycles: int | None
    latency_best_seconds: float | None
    latency_average_cycles: int | None
    latency_average_seconds: float | None
    latency_worst_cycles: int | None
    latency_worst_seconds: float | None

    resources_lut_used: int
    resources_ff_used: int
    resources_dsp_used: int
    resources_bram_used: int
    resources_uram_used: int

    resources_lut_total: int
    resources_ff_total: int
    resources_dsp_total: int
    resources_bram_total: int
    resources_uram_total: int

    resources_lut_fraction_used: float
    resources_ff_fraction_used: float
    resources_dsp_fraction_used: float
    resources_bram_fraction_used: float
    resources_uram_fraction_used: float

    @classmethod
    def parse_from_synth_report_file(cls, fp: Path) -> "DesignHLSSynthData":
        tree = ET.parse(fp)
        root = tree.getroot()

        # Gather latency data
        performance_estimates = unwrap(root.find("PerformanceEstimates"))

        summary_of_timing_analysis = unwrap(
            performance_estimates.find("SummaryOfTimingAnalysis")
        )

        clock_units = _find_text(summary_of_timing_analysis, "unit")
        clock_period = float(
            _find_text(summary_of_timing_analysis, "EstimatedClockPeriod")
        )

        unit_scaler = 1.0
        if clock_units == "ns":
            unit_scaler = 1e-9
        elif clock_units == "us":
            unit_scaler = 1e-6
        elif clock_units == "ms":
            unit_scaler = 1e-3
        else:
            raise NotImplementedError(f"Unknown clock unit: {clock_units}")

        clock_period_t = clock_period * unit_scaler

        summary_of_overall_latency = unwrap(
            performance_estimates.find("SummaryOfOverallLatency")
        )
        latency_data = {}
        # fmt: off
        best_case_latency = _find_text(summary_of_overall_latency, "Best-caseLatency")
        try:
            latency_data["best_case_latency"] = int(best_case_latency)
        except ValueError:
            latency_data["best_case_latency"] = None
        average_case_latency = _find_text(summary_of_overall_latency, "Average-caseLatency")
        try:
            latency_data["average_case_latency"] = int(average_case_latency)
        except ValueError:
            latency_data["average_case_latency"] = None
        worst_case_latency = _find_text(summary_of_overall_latency, "Worst-caseLatency")
        try:
            latency_data["worst_case_latency"] = int(worst_case_latency)
        except ValueError:
            latency_data["worst_case_latency"] = None

        if latency_data["best_case_latency"] is not None:
            latency_data["best_case_latency_t"] = latency_data["best_case_latency"] * clock_period_t
        else:
            latency_data["best_case_latency_t"] = None
        if latency_data["average_case_latency"] is not None:
            latency_data["average_case_latency_t"] = latency_data["average_case_latency"] * clock_period_t
        else:
            latency_data["average_case_latency_t"] = None
        if latency_data["worst_case_latency"] is not None:
            latency_data["worst_case_latency_t"] = latency_data["worst_case_latency"] * clock_period_t
        else:
            latency_data["worst_case_latency_t"] = None
        # fmt: on

        # Gather resource data
        area_estimates = unwrap(root.find("AreaEstimates"))
        resource_data = {}
        # fmt: off
        resource_data["used_abs"] = {}
        resource_data["used_abs"]["BRAM_18K"] = int(_find_text(area_estimates, "Resources", "BRAM_18K"))
        resource_data["used_abs"]["DSP"] = int(_find_text(area_estimates, "Resources", "DSP"))
        resource_data["used_abs"]["FF"] = int(_find_text(area_estimates, "Resources", "FF"))
        resource_data["used_abs"]["LUT"] = int(_find_text(area_estimates, "Resources", "LUT"))
        resource_data["used_abs"]["URAM"] = int(_find_text(area_estimates, "Resources", "URAM"))

        resource_data["available_abs"] = {}
        resource_data["available_abs"]["BRAM_18K"] = int(_find_text(area_estimates, "AvailableResources", "BRAM_18K"))
        resource_data["available_abs"]["DSP"] = int(_find_text(area_estimates, "AvailableResources", "DSP"))
        resource_data["available_abs"]["FF"] = int(_find_text(area_estimates, "AvailableResources", "FF"))
        resource_data["available_abs"]["LUT"] = int(_find_text(area_estimates, "AvailableResources", "LUT"))
        resource_data["available_abs"]["URAM"] = int(_find_text(area_estimates, "AvailableResources", "URAM"))

        resource_data["used_percent"] = {}
        # Checks for any cases were the available resources are 0 on a device
        # Only observed for URAM but we check all resources just in case

        if resource_data["available_abs"]["BRAM_18K"] == 0:
            assert resource_data["used_abs"]["BRAM_18K"] == 0
            resource_data["used_percent"]["BRAM_18K"] = 0.0
        else:
            resource_data["used_percent"]["BRAM_18K"] = float(resource_data["used_abs"]["BRAM_18K"] / resource_data["available_abs"]["BRAM_18K"])

        if resource_data["available_abs"]["DSP"] == 0:
            assert resource_data["used_abs"]["DSP"] == 0
            resource_data["used_percent"]["DSP"] = 0.0
        else:
            resource_data["used_percent"]["DSP"] = float(resource_data["used_abs"]["DSP"] / resource_data["available_abs"]["DSP"])

        if resource_data["available_abs"]["FF"] == 0:
            assert resource_data["used_abs"]["FF"] == 0
            resource_data["used_percent"]["FF"] = 0.0
        else:
            resource_data["used_percent"]["FF"] = float(resource_data["used_abs"]["FF"] / resource_data["available_abs"]["FF"])

        if resource_data["available_abs"]["LUT"] == 0:
            assert resource_data["used_abs"]["LUT"] == 0
            resource_data["used_percent"]["LUT"] = 0.0
        else:
            resource_data["used_percent"]["LUT"] = float(resource_data["used_abs"]["LUT"] / resource_data["available_abs"]["LUT"])

        if resource_data["available_abs"]["URAM"] == 0:
            assert resource_data["used_abs"]["URAM"] == 0
            resource_data["used_percent"]["URAM"] = 0.0
        else:
            resource_data["used_percent"]["URAM"] = float(resource_data["used_abs"]["URAM"] / resource_data["available_abs"]["URAM"])
        # fmt: on

        return cls(
            clock_period=clock_period_t,
            latency_best_cycles=latency_data["best_case_latency"],
            latency_best_seconds=latency_data["best_case_latency_t"],
            latency_average_cycles=latency_data["average_case_latency"],
            latency_average_seconds=latency_data["average_case_latency_t"],
            latency_worst_cycles=latency_data["worst_case_latency"],
            latency_worst_seconds=latency_data["worst_case_latency_t"],
            resources_lut_used=resource_data["used_abs"]["LUT"],
            resources_ff_used=resource_data["used_abs"]["FF"],
            resources_dsp_used=resource_data["used_abs"]["DSP"],
            resources_bram_used=resource_data["used_abs"]["BRAM_18K"],
            resources_uram_used=resource_data["used_abs"]["URAM"],
            resources_lut_total=resource_data["available_abs"]["LUT"],
            resources_ff_total=resource_data["available_abs"]["FF"],
            resources_dsp_total=resource_data["available_abs"]["DSP"],
            resources_bram_total=resource_data["available_abs"]["BRAM_18K"],
            resources_uram_total=resource_data["available_abs"]["URAM"],
            resources_lut_fraction_used=resource_data["used_percent"]["LUT"],
            resources_ff_fraction_used=resource_data["used_percent"]["FF"],
            resources_dsp_fraction_used=resource_data["used_percent"]["DSP"],
            resources_bram_fraction_used=resource_data["used_percent"]["BRAM_18K"],
            resources_uram_fraction_used=resource_data["used_percent"]["URAM"],
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "DesignHLSSynthData":
        return cls(**d)


@dataclass
class DesignHLSCoSimData:
    # $MAX_LATENCY = "100866"
    # $MIN_LATENCY = "100866"
    # $AVER_LATENCY = "100866"
    # $MAX_THROUGHPUT = "0"
    # $MIN_THROUGHPUT = "0"
    # $AVER_THROUGHPUT = "0"
    # $TOTAL_EXECUTE_TIME = "100866"

    max_latency: int
    min_latency: int
    average_latency: int

    max_throughput: int
    min_throughput: int
    average_throughput: int

    total_execute_time: int

    # Per kernel invocation (one entry per call in the testbench), from
    # result.transaction.rpt. The interval of the last transaction is None.
    transaction_latencies: list[int] = field(default_factory=list)
    transaction_intervals: list[int | None] = field(default_factory=list)

    @staticmethod
    def parse_transaction_report_file(
        fp: Path,
    ) -> tuple[list[int], list[int | None]]:
        # transaction       0:            2051            2052
        # transaction       1:          124675               x
        latencies: list[int] = []
        intervals: list[int | None] = []
        pattern = re.compile(r"^transaction\s+(\d+):\s+(\d+)\s+(\d+|x)\s*$")
        for line in fp.read_text().splitlines():
            m = pattern.match(line.strip())
            if m is None:
                continue
            assert int(m.group(1)) == len(latencies), f"Out of order row: {line}"
            latencies.append(int(m.group(2)))
            intervals.append(None if m.group(3) == "x" else int(m.group(3)))
        return latencies, intervals

    @classmethod
    def parse_from_synth_report_file(cls, fp: Path) -> "DesignHLSCoSimData":
        txt = fp.read_text()
        data = {}
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("$"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("$").lower()
            value = value.strip().strip('"')
            value_int = int(value)
            data[key] = value_int

        max_latency = unwrap(data.get("max_latency"))
        min_latency = unwrap(data.get("min_latency"))
        average_latency = unwrap(data.get("aver_latency"))

        max_throughput = unwrap(data.get("max_throughput"))
        min_throughput = unwrap(data.get("min_throughput"))
        average_throughput = unwrap(data.get("aver_throughput"))

        total_execute_time = unwrap(data.get("total_execute_time"))

        transaction_fp = fp.parent / "result.transaction.rpt"
        if transaction_fp.exists():
            latencies, intervals = cls.parse_transaction_report_file(transaction_fp)
        else:
            latencies, intervals = [], []

        return cls(
            max_latency=max_latency,
            min_latency=min_latency,
            average_latency=average_latency,
            max_throughput=max_throughput,
            min_throughput=min_throughput,
            average_throughput=average_throughput,
            total_execute_time=total_execute_time,
            transaction_latencies=latencies,
            transaction_intervals=intervals,
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "DesignHLSCoSimData":
        return cls(**d)
