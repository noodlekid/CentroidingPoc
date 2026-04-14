from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from crowsnest.poc.sim.synthetic_star_field import (
    GroundTruth,
    generate_synthetic_starfield,
)


@dataclass(frozen=True)
class FrameSample:
    frame: npt.NDArray[np.uint8]
    source: str
    ground_truth: list[GroundTruth] | None = None


class FrameSource(Protocol):
    def load(self) -> FrameSample: ...


@dataclass(frozen=True)
class SyntheticFrameSource:
    width: int
    height: int
    num_stars: int
    seed: int
    max_flux: float = 5000.0
    base_noise: float = 15.0

    def load(self) -> FrameSample:
        rng = np.random.default_rng(self.seed)
        frame, ground_truth = generate_synthetic_starfield(
            width=self.width,
            height=self.height,
            num_stars=self.num_stars,
            max_flux=self.max_flux,
            base_noise=self.base_noise,
            rng=rng,
        )
        return FrameSample(
            frame=frame,
            source=f"synthetic:{self.width}x{self.height}",
            ground_truth=ground_truth,
        )


@dataclass(frozen=True)
class FileFrameSource:
    image_path: Path
    target_width: int | None = None
    target_height: int | None = None

    def load(self) -> FrameSample:
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image file: {self.image_path}")

        gray = self._to_grayscale_uint8(image)

        if self.target_width is not None and self.target_height is not None:
            if gray.shape != (self.target_height, self.target_width):
                gray = cv2.resize(
                    gray,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )

        return FrameSample(frame=gray, source=f"file:{self.image_path}")

    @staticmethod
    def _to_grayscale_uint8(image: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unsupported channel count: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if gray.dtype == np.uint8:
            return gray

        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
