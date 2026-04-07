import numpy as np

from crowsnest.poc.pipeline.centroiding import extract_centroids_pipeline
from crowsnest.poc.sim.synthetic_star_field import generate_synthetic_starfield


def test_extract_centroids_sorted_by_flux_descending() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10, 10] = 240
    image[10, 11] = 180

    image[30, 30] = 220
    image[30, 31] = 210

    stars = extract_centroids_pipeline(
        image, sigma_mul=1.0, min_pixels=1, max_pixels=10
    )

    assert len(stars) >= 2
    assert all(stars[i].flux >= stars[i + 1].flux for i in range(len(stars) - 1))


def test_extract_centroids_recovers_synthetic_field_reasonably() -> None:
    rng = np.random.default_rng(123)
    image, ground_truth = generate_synthetic_starfield(
        width=256,
        height=192,
        num_stars=20,
        max_flux=4000,
        base_noise=10,
        rng=rng,
    )

    stars = extract_centroids_pipeline(
        image, sigma_mul=5.0, min_pixels=2, max_pixels=100
    )

    assert len(stars) > 0
    assert len(stars) >= int(len(ground_truth) * 0.5)
