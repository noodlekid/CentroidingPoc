from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from crowsnest.poc.pipeline.centroiding import StarCandidate, extract_centroids_pipeline
from crowsnest.poc.sim.synthetic_star_field import generate_synthetic_starfield


@dataclass(frozen=True)
class ScenarioConfig:
    width: int = 800
    height: int = 600
    num_stars: int = 50
    max_flux: float = 5000.0
    base_noise: float = 15.0
    seed: int = 42


@dataclass(frozen=True)
class TimingStats:
    runs: int
    warmup_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class DetectionMetrics:
    generated_stars: int
    detected_stars: int
    matched_stars: int
    precision: float
    recall: float


def generate_scenario(
    config: ScenarioConfig,
) -> tuple[npt.NDArray[np.uint8], list[dict[str, float]]]:
    rng = np.random.default_rng(config.seed)
    return generate_synthetic_starfield(
        width=config.width,
        height=config.height,
        num_stars=config.num_stars,
        max_flux=config.max_flux,
        base_noise=config.base_noise,
        rng=rng,
    )


def benchmark_function(
    fn: Callable[[], object],
    runs: int = 30,
    warmup_runs: int = 5,
) -> TimingStats:
    for _ in range(warmup_runs):
        fn()

    run_times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        run_times_ms.append(elapsed_ms)

    values = np.asarray(run_times_ms, dtype=np.float64)
    return TimingStats(
        runs=runs,
        warmup_runs=warmup_runs,
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
    )


def score_detections(
    detected: Sequence[StarCandidate],
    ground_truth: Iterable[dict[str, float]],
    max_distance_px: float = 2.0,
) -> DetectionMetrics:
    truth = list(ground_truth)
    unmatched_truth = set(range(len(truth)))
    matched = 0

    for star in detected:
        best_idx = None
        best_distance = float("inf")

        for idx in unmatched_truth:
            gt = truth[idx]
            dx = star.x - gt["true_x"]
            dy = star.y - gt["true_y"]
            dist = float(np.hypot(dx, dy))
            if dist < best_distance:
                best_distance = dist
                best_idx = idx

        if best_idx is not None and best_distance <= max_distance_px:
            unmatched_truth.remove(best_idx)
            matched += 1

    detected_count = len(detected)
    truth_count = len(truth)
    precision = matched / detected_count if detected_count else 0.0
    recall = matched / truth_count if truth_count else 0.0

    return DetectionMetrics(
        generated_stars=truth_count,
        detected_stars=detected_count,
        matched_stars=matched,
        precision=precision,
        recall=recall,
    )


def run_centroiding_iteration(
    config: ScenarioConfig,
    sigma_mul: float = 5.0,
    min_pixels: int = 2,
    max_pixels: int = 100,
    runs: int = 30,
    warmup_runs: int = 5,
) -> tuple[TimingStats, DetectionMetrics]:
    frame, truth = generate_scenario(config)

    def pipeline_call() -> list[StarCandidate]:
        return extract_centroids_pipeline(
            frame,
            sigma_mul=sigma_mul,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    timing = benchmark_function(pipeline_call, runs=runs, warmup_runs=warmup_runs)
    metrics = score_detections(pipeline_call(), truth)
    return timing, metrics
