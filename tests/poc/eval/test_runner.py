from crowsnest.poc.eval.runner import (
    ScenarioConfig,
    benchmark_function,
    generate_scenario,
    run_centroiding_iteration,
)


def test_generate_scenario_is_seed_deterministic() -> None:
    config = ScenarioConfig(width=128, height=96, num_stars=8, seed=7)

    frame_a, truth_a = generate_scenario(config)
    frame_b, truth_b = generate_scenario(config)

    assert frame_a.shape == frame_b.shape
    assert (frame_a == frame_b).all()
    assert truth_a == truth_b


def test_benchmark_function_returns_valid_stats() -> None:
    counter = {"count": 0}

    def fn() -> None:
        counter["count"] += 1

    stats = benchmark_function(fn, runs=5, warmup_runs=2)

    assert stats.runs == 5
    assert stats.warmup_runs == 2
    assert stats.mean_ms >= 0.0
    assert stats.p95_ms >= stats.min_ms
    assert counter["count"] == 7


def test_run_centroiding_iteration_returns_timing_and_metrics() -> None:
    config = ScenarioConfig(width=256, height=192, num_stars=20, seed=13)
    timing, metrics = run_centroiding_iteration(config=config, runs=4, warmup_runs=1)

    assert timing.runs == 4
    assert metrics.generated_stars == 20
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
