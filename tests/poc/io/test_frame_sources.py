from pathlib import Path

import cv2
import numpy as np

from crowsnest.poc.io.frame_sources import FileFrameSource, SyntheticFrameSource


def test_synthetic_frame_source_is_seed_deterministic() -> None:
    source_a = SyntheticFrameSource(width=96, height=72, num_stars=12, seed=123)
    source_b = SyntheticFrameSource(width=96, height=72, num_stars=12, seed=123)

    sample_a = source_a.load()
    sample_b = source_b.load()

    assert sample_a.source.startswith("synthetic:")
    assert sample_a.frame.shape == (72, 96)
    assert (sample_a.frame == sample_b.frame).all()
    assert sample_a.ground_truth == sample_b.ground_truth


def test_file_frame_source_loads_color_image_as_grayscale_uint8(tmp_path: Path) -> None:
    color = np.zeros((10, 16, 3), dtype=np.uint8)
    color[:, :, 1] = 200
    image_path = tmp_path / "sample_color.png"
    cv2.imwrite(str(image_path), color)

    sample = FileFrameSource(image_path=image_path).load()

    assert sample.source.endswith(str(image_path))
    assert sample.frame.dtype == np.uint8
    assert sample.frame.shape == (10, 16)


def test_file_frame_source_normalizes_and_resizes_16bit_image(tmp_path: Path) -> None:
    image_16 = np.linspace(0, 4095, num=20 * 30, dtype=np.uint16).reshape((20, 30))
    image_path = tmp_path / "sample_16bit.png"
    cv2.imwrite(str(image_path), image_16)

    sample = FileFrameSource(
        image_path=image_path,
        target_width=15,
        target_height=10,
    ).load()

    assert sample.frame.dtype == np.uint8
    assert sample.frame.shape == (10, 15)
