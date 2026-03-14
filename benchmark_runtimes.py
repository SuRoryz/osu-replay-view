"""Compare benchmark results across multiple Python interpreters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="pythons", action="append", required=True, help="Python executable to benchmark. Repeat for multiple runtimes.")
    parser.add_argument("--mode", default="gameplay", choices=("menu", "gameplay"))
    parser.add_argument("--output-dir", default="bench-results")
    parser.add_argument("--menu-seconds", type=float, default=8.0)
    parser.add_argument("--gameplay-seconds", type=float, default=12.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=4)
    return parser.parse_args()


def _runtime_tag(python_exe: str) -> str:
    exe_path = Path(python_exe)
    parent = exe_path.parent.name.replace(" ", "_")
    stem = exe_path.stem.replace(" ", "_")
    return f"{parent}_{stem}"


def _run_benchmark(python_exe: str, args: argparse.Namespace, output_path: Path) -> dict:
    env = os.environ.copy()
    runtime_tag = _runtime_tag(python_exe)
    env.update(
        {
            "OSU_PROFILE": "1",
            "OSU_PROFILE_INTERVAL": "9999",
            "OSU_BENCH_MODE": args.mode,
            "OSU_BENCH_WIDTH": str(args.width),
            "OSU_BENCH_HEIGHT": str(args.height),
            "OSU_BENCH_SAMPLES": str(args.samples),
            "OSU_BENCH_MENU_SECONDS": str(args.menu_seconds),
            "OSU_BENCH_GAMEPLAY_SECONDS": str(args.gameplay_seconds),
            "OSU_BENCH_WARMUP_SECONDS": str(args.warmup_seconds),
            "OSU_BENCH_OUTPUT": str(output_path),
            "OSU_BENCH_LABEL": runtime_tag,
        }
    )
    subprocess.run([python_exe, "profile_runtime.py"], check=True, env=env)
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for python_exe in args.pythons:
        runtime_tag = _runtime_tag(python_exe)
        output_path = output_dir / f"{runtime_tag}_{args.mode}.json"
        result = _run_benchmark(python_exe, args, output_path)
        results.append(result)

    results.sort(key=lambda item: float(item.get("avg_fps", 0.0)), reverse=True)
    for result in results:
        print(
            f"{result.get('label') or result.get('mode')}: "
            f"{float(result.get('avg_fps', 0.0)):.2f} FPS "
            f"(avg {float(result.get('frame_stats', {}).get('avg_ms', 0.0)):.3f} ms)"
        )
    summary_path = output_dir / f"summary_{args.mode}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
