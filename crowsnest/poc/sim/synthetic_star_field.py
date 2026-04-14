import numpy as np
import numpy.typing as npt
import cv2
from typing import TypedDict


class GroundTruth(TypedDict):
    id: int
    true_x: float
    true_y: float
    flux: float

def generate_synthetic_starfield(
    width: int=800,
    height: int=600,
    num_stars: int=50,
    max_flux: float=5000,
    base_noise: float =15,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.uint8], list[GroundTruth]]:
    """
    generates synthetic star field with subpixel rendering and some noise

    returns the image and a list of the true coordinates for validation
    """
    rng = rng if rng is not None else np.random.default_rng()

    # blank field
    image = np.zeros((height, width), dtype=np.float32)
    ground_truth: list[GroundTruth] = []

    for i in range(num_stars):
        # random sub pixels generation
        true_x = rng.uniform(5, width - 5)
        true_y = rng.uniform(5, height - 5)
        flux = rng.uniform(max_flux * 0.1, max_flux)

        # blur radius (sigma)
        sigma = rng.uniform(0.8, 1.5)

        ground_truth.append({"id": i, "true_x": true_x, "true_y": true_y, "flux": flux})

        # render the 2D Gaussian PSF over a small bounding box
        box_radius = int(sigma * 4) + 1
        x_min = max(0, int(true_x) - box_radius)
        x_max = min(width, int(true_x) + box_radius + 1)
        y_min = max(0, int(true_y) - box_radius)
        y_max = min(height, int(true_y) + box_radius + 1)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # distance to center
                dx = x - true_x
                dy = y - true_y

                # guass intensity
                intensity = flux * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                image[y, x] += intensity


    # inject bg noise (simulates dark current)
    image += base_noise

    # add some Poisson noise based on accumulated signal
    noisy_image = rng.poisson(image).astype(np.float32)

    # add guassian read noise
    read_noise = rng.normal(loc=0.0, scale=3.0, size=(height, width))
    noisy_image += read_noise

    # clip to 8 bit
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image, ground_truth
