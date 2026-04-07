from __future__ import annotations

import argparse
import json
from pathlib import Path

from crowsnest.poc.eval.runner import ScenarioConfig, run_centroiding_iteration
from crowsnest.poc.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate centroiding pipeline"
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--num-stars", type=int, default=50)
    parser.add_argument("--max-flux", type=float, default=5000.0)
    parser.add_argument("--base-noise", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma-mul", type=float, default=5.0)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=100)

    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        logger_name="crowsnest.benchmark",
    )

    config = ScenarioConfig(
        width=args.width,
        height=args.height,
        num_stars=args.num_stars,
        max_flux=args.max_flux,
        base_noise=args.base_noise,
        seed=args.seed,
    )

    timing, metrics = run_centroiding_iteration(
        config=config,
        sigma_mul=args.sigma_mul,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )

    result = {
        "scenario": {
            "width": config.width,
            "height": config.height,
            "num_stars": config.num_stars,
            "max_flux": config.max_flux,
            "base_noise": config.base_noise,
            "seed": config.seed,
        },
        "timing": {
            "runs": timing.runs,
            "warmup_runs": timing.warmup_runs,
            "mean_ms": timing.mean_ms,
            "median_ms": timing.median_ms,
            "p95_ms": timing.p95_ms,
            "min_ms": timing.min_ms,
            "max_ms": timing.max_ms,
        },
        "metrics": {
            "generated_stars": metrics.generated_stars,
            "detected_stars": metrics.detected_stars,
            "matched_stars": metrics.matched_stars,
            "precision": metrics.precision,
            "recall": metrics.recall,
        },
    }

    logger.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
