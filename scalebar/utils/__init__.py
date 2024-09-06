import contextlib

from scalebar.utils.image_ops import read_image
from scalebar.utils.image_ops import equalize
from scalebar.utils.image_ops import threshold
from scalebar.utils.image_ops import hide_non_roi
from scalebar.utils.image_ops import match_scalebar
from scalebar.utils.image_ops import detect_scalebar
from scalebar.utils.image_ops import detect_scalebar_multi
from scalebar.utils.pattern import create as create_pattern
from scalebar.utils.pattern import match as match_pattern
from scalebar.utils.corner_ops import rectify
from scalebar.utils.corner_ops import filter as filter_corners


@contextlib.contextmanager
def try_import(package_name):
    try:
        yield
    except ImportError:
        msg = f"{package_name} not found! " + \
            f"Install it with 'pip install {package_name}'!"
        print(msg)
        raise


__all__ = [
    "try_import",
    "read_image",
    "rectify",
    "equalize",
    "threshold",
    "hide_non_roi",
    "match_scalebar",
    "detect_scalebar",
    "detect_scalebar_multi",
    "create_pattern",
    "match_pattern",
    "filter_corners",
]
