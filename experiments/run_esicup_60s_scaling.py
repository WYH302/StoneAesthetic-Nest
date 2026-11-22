"""Run Official Rust Sparrow v0.3.0 at 60 s/instance across all 14 EURO/ESICUP-14 benchmarks.

Compares interactive 2 s/instance vs 60 s/instance time-density scaling using the
exact unmodified Rust binary (`/root/native_solvers/sparrow`) in OCCAM-Ubuntu WSL.
Runs 2 instances concurrently with 4 workers each (8 total threads on 8C/16T CPU).
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MASTER_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MASTER_DIR.parent
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from src.solver_core.native_sota_runners import run_native_sparrow_on_esicup_file

ES_DIR = WORKSPACE_ROOT / "third_party" / "sparrow" / "data" / "input"
ES_NAMES = [
    "albano",
    "blaz1",
    "dagli",
    "fu",
    "gardeyn0",
    "jakobs1",
    "jakobs2",
    "mao",
    "marques",
    "shapes0",
    "shapes1",
    "shirts",
    "swim",
    "trousers",
]
OUT_JSON = MASTER_DIR / "experiments" / "results" / "esicup_14_sparrow_60s.json"


def _solve_one(name: str) -> dict[str, Any]:
    jp = ES_DIR / f"{name}.json"
    t0 = time.perf_counter()
    res = run_native_sparrow_on_esicup_file(jp, time_limit_sec=60, seed=42)
    wall = round(time.perf_counter() - t0, 2)
    res["wall_sec"] = wall
    print(
        f"[Sparrow 60s] {name:10s} | density={res['density']*100:6.2f}% | "
        f"width={res['strip_width']:10.3f} | items={res['placed_count']:3d} | wall={wall:5.1f}s",
        flush=True,
    )
    return res


def main() -> None:
    t_all = time.perf_counter()
    results_map: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_solve_one, nm): nm for nm in ES_NAMES}
        for fut in concurrent.futures.as_completed(futs):
            nm = futs[fut]
            results_map[nm] = fut.result()

    rows = [results_map[nm] for nm in ES_NAMES]
    dens = [float(r["density"]) for r in rows]
    summary = {
        "time_limit_sec": 60,
        "seed": 42,
        "workers_per_instance": 4,
        "num_instances": len(rows),
        "total_wall_sec": round(time.perf_counter() - t_all, 2),
        "sparrow_60s_mean_density": round(float(np.mean(dens)), 4),
        "sparrow_60s_std_density": round(float(np.std(dens, ddof=1)), 4),
        "per_instance": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\nDONE! Mean 60s density = {summary['sparrow_60s_mean_density']*100:.2f}% "
        f"+/- {summary['sparrow_60s_std_density']*100:.2f}% in {summary['total_wall_sec']}s -> {OUT_JSON}",
        flush=True,
    )


if __name__ == "__main__":
    main()
