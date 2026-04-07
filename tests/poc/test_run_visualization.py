import numpy as np

from crowsnest.poc.run import build_before_after_visualization


def test_build_before_after_visualization_layout_and_type() -> None:
    frame = np.zeros((64, 80), dtype=np.uint8)
    frame[20, 20] = 255
    frame[40, 60] = 230

    before_image, after_image, stars = build_before_after_visualization(
        frame,
        sigma_mul=1.0,
        min_pixels=1,
        max_pixels=20,
    )

    assert before_image.dtype == np.uint8
    assert after_image.dtype == np.uint8
    assert before_image.shape == (64, 80, 3)
    assert after_image.shape == (64, 80, 3)
    assert len(stars) >= 1
