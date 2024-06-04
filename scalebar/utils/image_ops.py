import cv2
import typing as T
import numpy as np

from pathlib import Path
from PIL import Image

from scalebar.core.bounding_box import BoundingBox

WHITE = 255
GRAY = 127
BLACK = 0

def read_image(image_path: T.Optional[T.Union[str, Path]],
               mode: str = "RGB") -> np.ndarray:
    """ Reads an image located at the given path """
    with Image.open(image_path) as img:
        return np.array(img.convert(mode))

def threshold(image: np.ndarray,
              threshold: int = GRAY,
              max_value: int = WHITE,
              mode: int = cv2.THRESH_BINARY) -> np.ndarray:
    """ Applies a threshold to the given image """
    _, binary = cv2.threshold(image, threshold, max_value, mode)
    return binary

def equalize(gray: np.ndarray, *,
             clipLimit=2.0, tileGridSize=(8,8)) -> np.ndarray:
    """ applies the CLAHE algorithm to the given grayscale image """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def hide_non_roi(im, fraction: float, value,
                 location: "Position" = None, min_size: int = 200):
    """ mask the non-ROI region of the image """
    res = im.copy()
    H, W, *_ = im.shape
    if min(H, W) < min_size * 2:
        return res

    y0, x0 = max(int(H*fraction), min_size), max(int(W*fraction), min_size)
    res[y0:H-y0, x0:W-x0] = value
    if location is not None:
        mask = np.zeros_like(res, dtype=bool)
        location.crop(mask, fraction=2*fraction, square=False)[:] = True
        res[~mask] = value
    return res

def match_scalebar(bin_im, template_size: int):
    """ Match the template with the binary image
        and return the result of the template matching and the template itself.

        The template is a small checkerboard pattern that is used to match the
        scale bar in the image.

        The template is created by creating a 2D array of size (2*template_size, 2*template_size)
        and filling the top-right and bottom-left quadrants with white pixels.

        The template is then matched with the binary image using the cv2.matchTemplate function
        with the TM_CCOEFF_NORMED method. The result is then normalized and thresholded to
        create a binary image highlighting the matched regions.

        Finally, the matched image and the template are returned.
    """

    template = np.full((2*template_size, 2*template_size), BLACK, dtype=bin_im.dtype)
    template[:template_size, template_size:] = WHITE
    template[template_size:, :template_size] = WHITE

    padding = (template_size, template_size-1)
    match = np.abs(cv2.matchTemplate(bin_im, template, method=cv2.TM_CCOEFF_NORMED))
    match = np.pad(match, (padding, padding))
    assert bin_im.shape == match.shape

    match -= match.min()
    match /=  (match.max() or 1)
    match = (match * 255).astype(np.uint8)

    _, match = cv2.threshold(match, GRAY, WHITE, cv2.THRESH_BINARY)

    return match, template


def detect_scalebar(match, enlarge: int = 10) -> BoundingBox:
    """
        Find the bounding box of the scale bar in the image.
        We assume that 'match' is the result of the template matching: a binary image
        highlighting the locations matched with a small checkerboard.

        Next, we find the contours of the matched regions and filter them based on their
        mean and standard deviation. This is done to remove outliers and keep only the
        regions that are close to the center of the scale bar.

        Finally, we compute the bounding box of the scale bar by enlarging the region
        around the matched points.
    """
    # Finding contours for the thresholded image
    contours, _ = cv2.findContours(match, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_im = np.full_like(match, BLACK)
    cont_im = cv2.drawContours(cont_im, contours, -1, WHITE, 2)

    coords = np.zeros((len(contours), 2), dtype=np.int32)
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        coords[i, 0] = x + w//2
        coords[i, 1] = y + h//2

    mean, std = coords.mean(axis=0), coords.std(axis=0)
    low, high = mean - 2*std, mean + 2*std
    xs, ys = coords.T

    mask = np.logical_and(
        (low[0] <= xs) & (xs <= high[0]),
        (low[1] <= ys) & (ys <= high[1])
    )

    coords = coords[mask]
    H, W = match.shape[:2]
    x0, y0 = np.maximum(coords.min(axis=0) - enlarge, 0)
    x1, y1 = np.minimum(coords.max(axis=0) + 2*enlarge, (W, H))
    return BoundingBox(x0, y0, x1-x0, y1-y0)
