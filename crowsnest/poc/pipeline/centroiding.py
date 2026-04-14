import numpy as np
import numpy.typing as npt
import cv2
from dataclasses import dataclass


@dataclass
class StarCandidate:
    id: int
    x: float
    y: float
    flux: float
    pixels: int


def estimate_background(image_array: npt.NDArray[np.uint8]) -> tuple[float, float]:
    """
    statistical background estimation

    pretty memory bound, reads the whole array
    """
    median_bg = float(np.median(image_array))
    noise_std = float(np.std(image_array))
    return median_bg, noise_std


def apply_threshold(
    image_array: npt.NDArray[np.uint8],
    median_bg: float,
    noise_std: float,
    sigma_mul: float,
) -> npt.NDArray[np.uint8]:
    """
    binarization

    memory bound : we can vectorize this
    """
    threshold_val = median_bg + (sigma_mul * noise_std)
    _, binary_mask = cv2.threshold(image_array, threshold_val, 255, cv2.THRESH_BINARY)
    return binary_mask.astype(np.uint8)


def extract_blobs(
    binary_mask: npt.NDArray[np.uint8],
) -> tuple[int, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    connected components

    heavy branching and spatial locality dependent
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # TODO: cast to int, array, array
    return num_labels, labels, stats


def calculate_subpixel_centroids(
    image_array: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.int32],
    stats: npt.NDArray[np.int32],
    num_labels: int,
    median_bg: float,
    min_pixels: int = 2,
    max_pixels: int = 100,
) -> list[StarCandidate]:
    """
    center of mass calculations
    compute bound might be bad with cache misses
    """
    candidate_stars: list[StarCandidate] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_pixels or area > max_pixels:
            continue

        x_start = stats[i, cv2.CC_STAT_LEFT]
        y_start = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # extract ROI
        roi = image_array[y_start : y_start + height, x_start : x_start + width].astype(
            np.float32
        )

        intensity_sum = 0.0
        weighted_x_sum = 0.0
        weighted_y_sum = 0.0

        for y in range(height):
            for x in range(width):
                if labels[y_start + y, x_start + x] == i:
                    intensity = roi[y, x] - median_bg
                    if intensity > 0:
                        intensity_sum += intensity
                        weighted_x_sum += (x_start + x) * intensity
                        weighted_y_sum += (y_start + y) * intensity

        if intensity_sum > 0:
            sub_pixel_x = weighted_x_sum / intensity_sum
            sub_pixel_y = weighted_y_sum / intensity_sum

            candidate_stars.append(
                StarCandidate(
                    id=i,
                    x=sub_pixel_x,
                    y=sub_pixel_y,
                    flux=intensity_sum,
                    pixels=int(area),
                )
            )

    # sort by brightness
    candidate_stars.sort(key=lambda s: s.flux, reverse=True)
    return candidate_stars


def extract_centroids_pipeline(
    image_array: npt.NDArray[np.uint8],
    sigma_mul: float = 5.0,
    min_pixels: int = 2,
    max_pixels: int = 100,
) -> list[StarCandidate]:
    """
    run full pipeline
    """
    median_bg, noise_std = estimate_background(image_array)
    binary_mask = apply_threshold(image_array, median_bg, noise_std, sigma_mul)
    num_labels, labels, stats = extract_blobs(binary_mask)
    stars = calculate_subpixel_centroids(
        image_array=image_array,
        labels=labels,
        stats=stats,
        num_labels=num_labels,
        median_bg=median_bg,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return stars
