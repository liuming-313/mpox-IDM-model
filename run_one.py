"""Single-run wrapper for calibration subprocesses.

The model entrypoint in main.py is currently a top-level script. This module
therefore runs it in a subprocess after temporarily patching only allowed
calibration/runtime values, then restores the original main.py text.
"""

from __future__ import annotations

import ast
import csv
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from calibration_config import (
    DIAGNOSTIC_PLOT_DIR,
    MAIN_PATH,
    PROJECT_DIR,
    SEARCH_SPACE_PRIMARY,
    SEARCH_SPACE_SECONDARY,
    TARGETS,
)
from objective import calculate_objective


ALLOWED_OVERRIDE_KEYS = set(SEARCH_SPACE_PRIMARY) | set(SEARCH_SPACE_SECONDARY)
ALLOWED_OVERRIDE_KEYS |= {
    "party_touch_frac",
    "close_touch",
    "sexual_risk_scale",
    "p_stable_sex_general_per_day",
    "p_stable_sex_one_mcp_per_day",
    "p_stable_sex_both_mcp_per_day",
    "monitor_sexual_contact_mult",
    "monitor_social_contact_mult",
    "MONITORING_DAYS",
    "party_mix_homo_spill_frac",
    "p_bi_f_to_wsw",
}


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _label_for_pop_size(pop_size: int) -> str:
    return f"{int(int(pop_size) / 1e6)}M_homo_male_mcp"


def _diagnostics_path(pop_size: int) -> Path:
    return PROJECT_DIR / "outputs" / _label_for_pop_size(pop_size) / "calibration_diagnostics.csv"


def _figures_dir(pop_size: int) -> Path:
    return PROJECT_DIR / "outputs" / _label_for_pop_size(pop_size) / "figures"


def _patch_policy_testing(text: str, policy_adjustments: dict[str, Any] | None) -> str:
    if not policy_adjustments:
        return text

    if "testing_prob_mult" in policy_adjustments:
        mult = float(policy_adjustments["testing_prob_mult"])

        def repl_prob(match: re.Match[str]) -> str:
            value = min(1.0, max(0.0, float(match.group(2)) * mult))
            return f"{match.group(1)}{value:g}"

        text = re.sub(r"(test_pathway_prob\s*=\s*)(" + _float_pattern() + r")", repl_prob, text)

    if "testing_delay_mean_days_by_phase" in policy_adjustments:
        values_by_phase = policy_adjustments["testing_delay_mean_days_by_phase"]
        for phase, phase_value in values_by_phase.items():
            pattern = (
                r"(dict\(name=['\"]"
                + re.escape(str(phase))
                + r"['\"].*?test_seek_delay_mean_days\s*=\s*)"
                + _float_pattern()
            )
            text, n = re.subn(
                pattern,
                rf"\g<1>{float(phase_value):g}",
                text,
                count=1,
                flags=re.DOTALL,
            )
            if n != 1:
                raise ValueError(f"Could not safely patch test delay for policy phase {phase}")
    elif "testing_delay_shift_days" in policy_adjustments:
        shift = float(policy_adjustments["testing_delay_shift_days"])

        def repl_delay(match: re.Match[str]) -> str:
            value = max(0.0, float(match.group(2)) + shift)
            return f"{match.group(1)}{value:g}"

        text = re.sub(r"(test_seek_delay_mean_days\s*=\s*)(" + _float_pattern() + r")", repl_delay, text)

    return text


def patch_main_text(
    original: str,
    params_override: dict[str, float],
    seed: int,
    pop_size: int,
    make_plots: bool,
    policy_adjustments: dict[str, Any] | None = None,
) -> str:
    text = original
    for key, value in params_override.items():
        if key not in ALLOWED_OVERRIDE_KEYS:
            raise ValueError(f"Refusing to patch non-calibration parameter: {key}")
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_pattern()})'
        text, n = re.subn(pattern, rf"\g<1>{float(value):g}", text, count=1, flags=re.MULTILINE)
        if n == 0:
            insert_pattern = r'(^\s*"calibration_targets"\s*:\s*CALIBRATION_TARGETS\s*,)'
            text, n = re.subn(
                insert_pattern,
                rf'    "{key}": {float(value):g},\n\g<1>',
                text,
                count=1,
                flags=re.MULTILINE,
            )
        if n != 1:
            raise ValueError(f"Could not safely patch or insert {key} in main.py")
    replacements = [
        (rf"(^\s*base_run_seed\s*=\s*)({_float_pattern()})", rf"\g<1>{int(seed)}", "base_run_seed"),
        (rf"(^\s*POPULATION_SIZE\s*=\s*)(.+)$", rf"\g<1>int({int(pop_size)})", "POPULATION_SIZE"),
        (r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)", rf"\g<1>{str(bool(make_plots))}", "MAKE_PLOTS"),
    ]
    for pattern, replacement, name in replacements:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch {name} in main.py")
    return _patch_policy_testing(text, policy_adjustments)


def _read_diagnostics(pop_size: int, retries: int = 10) -> dict[str, float]:
    path = _diagnostics_path(pop_size)
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            values: dict[str, float] = {}
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    metric = (row.get("metric") or "").strip()
                    if metric in TARGETS:
                        values[metric] = float(row["current"])
                    elif metric == "mean_absolute_error_pp":
                        values["diagnostic_mae_pp"] = float(row["diff_percentage_points"])
            missing = sorted(set(TARGETS) - set(values))
            if missing:
                raise ValueError(f"Missing diagnostics: {missing}")
            return values
        except (PermissionError, FileNotFoundError, ValueError) as exc:
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(f"Could not read diagnostics for pop_size={pop_size}: {last_exc}")


def _parse_final_state(stdout: str) -> dict[str, Any]:
    match = re.search(r"Final state:\s*(\{.*?\})", stdout, flags=re.DOTALL)
    if not match:
        return {}
    raw = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", match.group(1))
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def _total_cases(summary: dict[str, Any]) -> float | None:
    count_keys = [
        "episode_female_hetero",
        "episode_male_hetero",
        "episode_male_homo",
        "episode_female_homo",
        "episode_male_bi",
        "episode_female_bi",
    ]
    if all(isinstance(summary.get(k), (int, float)) for k in count_keys):
        return float(sum(summary[k] for k in count_keys))
    for key in ["episode_total_cases", "total_episode_cases", "diagnosed_total", "total_cases"]:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def copy_diagnostic_plots(run_id: str, pop_size: int, destination_root: Path = DIAGNOSTIC_PLOT_DIR) -> str:
    src = _figures_dir(pop_size)
    if not src.exists():
        return ""
    dst = destination_root / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for png in src.glob("*.png"):
        shutil.copy2(png, dst / png.name)
    return str(dst)


def run_model(
    params_override: dict[str, float],
    seed: int,
    pop_size: int,
    run_id: str | None = None,
    make_plots: bool = False,
    log_dir: str | Path | None = None,
    copy_plots: bool = False,
    diagnostic_plot_dir: str | Path = DIAGNOSTIC_PLOT_DIR,
    policy_adjustments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"run_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_root = Path(log_dir) if log_dir else PROJECT_DIR / "outputs" / "calibration_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_log = log_root / f"{run_id}.stdout.log"
    stderr_log = log_root / f"{run_id}.stderr.log"

    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    result: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "pop_size": pop_size,
        "status": "failed",
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "diagnostic_plot_dir": "",
    }
    try:
        MAIN_PATH.write_text(
            patch_main_text(original, params_override, seed, pop_size, make_plots, policy_adjustments),
            encoding="utf-8",
        )
        proc = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=PROJECT_DIR, text=True, capture_output=True)
        stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            result["error"] = f"main.py exited with return code {proc.returncode}"
            return result
        final_state = _parse_final_state(proc.stdout)
        diagnostics = _read_diagnostics(pop_size)
        summary = {
            "total_cases": _total_cases(final_state),
            "hetero": diagnostics["hetero"],
            "bi": diagnostics["bi"],
            "homo": diagnostics["homo"],
            "female_known": diagnostics["female_known"],
            "homo_male": diagnostics["homo_male"],
            "bi_male": diagnostics["bi_male"],
        }
        objective = calculate_objective(summary)
        result.update(summary)
        result.update({f"objective_{k}": v for k, v in objective.items()})
        result["score"] = objective["score"]
        result["status"] = "ok"
        if copy_plots:
            result["diagnostic_plot_dir"] = copy_diagnostic_plots(run_id, pop_size, Path(diagnostic_plot_dir))
    except Exception:
        result["error"] = traceback.format_exc(limit=6)
    finally:
        MAIN_PATH.write_text(original, encoding="utf-8")
        result["elapsed_s"] = (datetime.now() - started).total_seconds()
    return result
