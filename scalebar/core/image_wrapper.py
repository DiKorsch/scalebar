import cv2
import numpy as np
import typing as T

from dataclasses import dataclass

from scalebar import utils

@dataclass
class StructureSizes:
    size: int

    @property
    def kernel_size(self) -> int:
        return self.size * 2 + 1

    @property
    def kernel_shape(self) -> tuple[int, int]:
        return (self.kernel_size, self.kernel_size)

    @property
    def kernel(self) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_shape, (self.size, self.size))

    @property
    def template_size(self) -> int:
        return self.size * 5

    @classmethod
    def new(cls, im: np.ndarray, fraction: float = 0.002) -> 'StructureSizes':
        H, W, *_ = im.shape
        size = max(int(min(H, W) * fraction), 1)
        return cls(size)

@dataclass
class Images:
    original: np.ndarray

    gray: T.Optional[np.ndarray] = None
    equalized: T.Optional[np.ndarray] = None
    binary: T.Optional[np.ndarray] = None
    masked: T.Optional[np.ndarray] = None
    template: T.Optional[np.ndarray] = None
    matched: T.Optional[np.ndarray] = None

    roi_fraction: float = 0.2
    structure_fraction: float = 0.002
    structure_sizes: StructureSizes = None

    def __post_init__(self):
        assert self.original is not None, "Original image is required"

        self.gray = gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)

        self.equalized = equalized = utils.equalize(gray)
        self.binary = binary = utils.threshold(equalized,
                                               threshold=64, mode=cv2.THRESH_BINARY)

        self.structure_sizes = StructureSizes.new(binary)
        kernel = self.structure_sizes.kernel
        self.binary = binary = cv2.dilate(cv2.erode(binary, kernel=kernel, iterations=2), kernel=kernel, iterations=2)

        self.masked = utils.hide_non_roi(binary, self.roi_fraction, 255)
