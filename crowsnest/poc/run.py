import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from crowsnest.poc.eval.runner import (
    ScenarioConfig,
    generate_scenario,
    run_centroiding_iteration,
)
from crowsnest.poc.logging_config import setup_logging
from crowsnest.poc.pipeline.centroiding import (
    StarCandidate,
    apply_threshold,
    calculate_subpixel_centroids,
    estimate_background,
    extract_blobs,
    extract_centroids_pipeline,
)


def run_pipeline(frame: npt.NDArray[np.uint8], w: np.uint16, h: np.uint16):
    if frame.shape != (int(h), int(w)):
        raise ValueError(
            f"Frame shape {frame.shape} does not match declared dimensions ({h}, {w})"
        )
    return extract_centroids_pipeline(frame)


def build_before_after_visualization(
    frame: npt.NDArray[np.uint8],
    sigma_mul: float = 5.0,
    min_pixels: int = 2,
    max_pixels: int = 100,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], list[StarCandidate]]:
    median_bg, noise_std = estimate_background(frame)
    binary_mask = apply_threshold(frame, median_bg, noise_std, sigma_mul)
    num_labels, labels, stats = extract_blobs(binary_mask)
    stars = calculate_subpixel_centroids(
        image_array=frame,
        labels=labels,
        stats=stats,
        num_labels=num_labels,
        median_bg=median_bg,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    before_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    after_bgr = before_bgr.copy()

    for star in stars:
        x_start = int(stats[star.id, cv2.CC_STAT_LEFT])
        y_start = int(stats[star.id, cv2.CC_STAT_TOP])
        width = int(stats[star.id, cv2.CC_STAT_WIDTH])
        height = int(stats[star.id, cv2.CC_STAT_HEIGHT])

        cv2.rectangle(
            after_bgr,
            (x_start, y_start),
            (x_start + width, y_start + height),
            color=(0, 255, 0),
            thickness=1,
        )
        cv2.drawMarker(
            after_bgr,
            (int(round(star.x)), int(round(star.y))),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
        )

    return before_bgr, after_bgr, stars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrowsNest iteration runner")
    parser.add_argument("--mode", choices=["run", "bench"], default="run")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--num-stars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--sigma-mul", type=float, default=5.0)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=100)
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("artifacts/centroiding.png"),
    )
    parser.add_argument("--show-image", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        logger_name="crowsnest.run",
    )

    cfg = ScenarioConfig(
        width=args.width,
        height=args.height,
        num_stars=args.num_stars,
        seed=args.seed,
    )

    if args.mode == "bench":
        timing, metrics = run_centroiding_iteration(
            config=cfg,
            sigma_mul=args.sigma_mul,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
        )
        logger.info(
            "Benchmark: mean=%.3f ms, p95=%.3f ms, precision=%.3f, recall=%.3f, matched=%d",
            timing.mean_ms,
            timing.p95_ms,
            metrics.precision,
            metrics.recall,
            metrics.matched_stars,
        )
    else:
        frame, _ = generate_scenario(cfg)
        before_image, after_image, stars = build_before_after_visualization(
            frame,
            sigma_mul=args.sigma_mul,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        before_path = args.output_image.with_name(
            f"{args.output_image.stem}_before{args.output_image.suffix}"
        )
        after_path = args.output_image.with_name(
            f"{args.output_image.stem}_after{args.output_image.suffix}"
        )
        cv2.imwrite(str(before_path), before_image)
        cv2.imwrite(str(after_path), after_image)

        logger.info("Detected %d stars", len(stars))
        logger.info("Before image saved to: %s", before_path)
        logger.info("After image saved to: %s", after_path)
        logger.info("Top 5 by flux:")
        for star in stars[:5]:
            logger.info(
                "id=%d, x=%.3f, y=%.3f, flux=%.2f, pixels=%d",
                star.id,
                star.x,
                star.y,
                star.flux,
                star.pixels,
            )

        if args.show_image:
            cv2.imshow("CrowsNest Before", before_image)
            cv2.imshow("CrowsNest After", after_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
