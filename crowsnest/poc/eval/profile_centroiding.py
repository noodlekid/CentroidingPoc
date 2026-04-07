from __future__ import annotations

import argparse
import cProfile
from io import StringIO
from pathlib import Path
import pstats

from crowsnest.poc.eval.runner import ScenarioConfig, generate_scenario
from crowsnest.poc.logging_config import setup_logging
from crowsnest.poc.pipeline.centroiding import extract_centroids_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated centroiding iterations for profiler/flamegraph tooling"
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--num-stars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma-mul", type=float, default=5.0)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=100)

    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="How many times to run centroiding on one deterministic frame",
    )
    parser.add_argument(
        "--cprofile-output",
        type=Path,
        default=None,
        help="Optional cProfile output path (.prof)",
    )
    parser.add_argument(
        "--top-functions",
        type=int,
        default=20,
        help="How many top cumulative functions to print in terminal",
    )
    parser.add_argument(
        "--profile-detail",
        action="store_true",
        help="Include top cProfile function table in INFO logs",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def run_loop(
    iterations: int,
    cfg: ScenarioConfig,
    sigma_mul: float,
    min_pixels: int,
    max_pixels: int,
) -> int:
    frame, _ = generate_scenario(cfg)
    total_detected = 0

    for _ in range(iterations):
        stars = extract_centroids_pipeline(
            frame,
            sigma_mul=sigma_mul,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        total_detected += len(stars)

    return total_detected


def main() -> None:
    args = parse_args()
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        logger_name="crowsnest.profile",
    )

    cfg = ScenarioConfig(
        width=args.width,
        height=args.height,
        num_stars=args.num_stars,
        seed=args.seed,
    )

    if args.cprofile_output is not None:
        args.cprofile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        total_detected = run_loop(
            iterations=args.iterations,
            cfg=cfg,
            sigma_mul=args.sigma_mul,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        profiler.disable()
        profiler.dump_stats(str(args.cprofile_output))

        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(args.top_functions)
        if args.profile_detail:
            logger.info("cProfile top functions (cumtime):\n%s", stream.getvalue())
        else:
            logger.debug(
                "cProfile top functions (cumtime):\n%s",
                stream.getvalue(),
            )
        logger.info("Saved cProfile stats to: %s", args.cprofile_output)
    else:
        total_detected = run_loop(
            iterations=args.iterations,
            cfg=cfg,
            sigma_mul=args.sigma_mul,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    avg_detected = total_detected / args.iterations if args.iterations else 0.0
    logger.info(
        "Completed %d iterations. Average detected stars per frame: %.2f",
        args.iterations,
        avg_detected,
    )


if __name__ == "__main__":
    main()
