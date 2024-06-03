import numpy as np
import typing as T

from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y+self.height, self.x:self.x+self.width]

    def __post_init__(self):
        if self.width < 0:
            raise ValueError("Width must be positive")
        if self.height < 0:
            raise ValueError("Height must be positive")

    @property
    def left(self) -> int:
        return self.x

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def top(self) -> int:
        return self.y

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> T.Tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def __contains__(self, point: T.Tuple[int, int]) -> bool:
        x, y = point
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def __and__(self, other: 'BoundingBox') -> 'BoundingBox':
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        return BoundingBox(x, y, right - x, bottom - y)

    def __or__(self, other: 'BoundingBox') -> 'BoundingBox':
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return BoundingBox(x, y, right - x, bottom - y)

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __str__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
