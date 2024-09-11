import cv2
import numpy as np
import typing as T

from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from PIL import Image
from dataclasses import dataclass

from scalebar import utils
from scalebar.core.bounding_box import BoundingBox
from scalebar.core.image_wrapper import Images
from scalebar.core.position import Position


@dataclass
class Result:
    image_path: str
    scalebar_location: T.Optional[Position] = None # apriori knowledge of the scale bar location
    image_size: T.Tuple[int, int] = (0, 0)


    images: T.Optional[Images] = None
    position: T.Optional[BoundingBox] = None # estimated position of the scale bar
    scalebar: T.Optional[np.ndarray] = None
    px_per_square: T.Optional[float] = None # [px/square]
    distances: T.Optional['Distances'] = None

    match: T.Optional[np.ndarray] = None
    template: T.Optional[np.ndarray] = None
    template_path: T.Optional[str] = None
    template_scale: T.Optional[float] = None # [px/square]

    roi_fraction: float = 0.15 # fraction of the image's border that will be used for the scale estimation
    size_per_square: float = 1.0 # [mm/square] how many mm is a single square

    def __post_init__(self):
        with Image.open(self.image_path) as img:
            self.image_size = img.size
        im = utils.read_image(self.image_path)
        self.images = Images(im, roi_fraction=self.roi_fraction,
                             location=self.scalebar_location)

    @classmethod
    def new(cls, file_name: str, *, max_corners: int = 50, **kwargs) -> 'Result':
        res = cls(file_name, **kwargs)
        res.locate()
        res.estimate(max_corners=max_corners)
        return res

    def locate(self, *, multi_scale: bool = True) -> BoundingBox:
        """ this function will locate the scale bar in the image, store the bounding box, and finally return it """

        if multi_scale:
            self.position, temp_size = utils.detect_scalebar_multi(
                self.images, self.template_path, template_scale=self.template_scale)

            if temp_size is not None:
                """ if the template size is estimated, store it directly """
                self.px_per_square = temp_size

        else:
            temp_size = self.images.structure_sizes.template_size
            self.match, self.template = utils.match_scalebar(self.images.masked, template_size=temp_size)
            self.position = utils.detect_scalebar(self.match, enlarge=temp_size)

        self.scalebar = self.position.crop(self.images.equalized)
        self.scalebar = utils.threshold(self.scalebar, mode=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return self.position

    def estimate(self, max_corners: int = 50) -> float:
        """ this function will estimate the scale of the image and return it """
        assert self.scalebar is not None, "Located and cropped scalebar is required"
        if self.px_per_square is not None:
            """ if the scale is already estimated, return it """
            return self.scale

        mask = None
        if self.match is not None:
            mask = self.position.crop(self.match)
        min_distance = self.images.structure_sizes.size

        corners = cv2.goodFeaturesToTrack(self.scalebar,
                                          maxCorners=0 if np.isinf(max_corners) else max_corners,
                                          qualityLevel=0.1,
                                          minDistance=min_distance,
                                          mask=mask)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # shape = (2*min_distance+1, 2*min_distance+1)
        # corners = cv2.cornerSubPix(self.scalebar, corners, shape, (-1,-1), criteria)
        # switch x and y coordinates for convenience (needed only for OpenCV)
        corners = corners[:, 0, ::-1].astype(int)

        self.distances = Distances(corners)
        self.px_per_square = self.distances.optimal_distance()

        return self.scale

    @property
    def scale(self) -> float:
        return self.px_per_square / self.size_per_square

    def best_corners(self, *, distance: T.Optional[float] = None) -> np.ndarray:
        return self.distances.best_corners(distance or self.px_per_square)


class Distances:
    def __init__(self, corners: np.ndarray, metric: str = "cityblock"):
        self.corners = corners
        self.distances = pdist(corners, metric=metric)

        self.unique_dists = None

    def reset(self):

        self.errors = [[], []]
        self.n_bins = []

        self.unique_dists = sorted(np.unique(np.maximum(self.distances, 10)))
        self.unique_dists = self.unique_dists[:len(self.unique_dists)//5]

    def optimal_distance(self, use_bic: bool = False) -> float:
        self.reset()
        smallest_err = np.inf
        best_distance = None

        for d in self.unique_dists:
            norm_dist = self.distances / d
            grid = np.arange(0, np.max(self.distances) + d, d) / d

            if len(grid) <= 2:
                print(f"FOO: {d}")
                continue


            prototypes = grid[1:-1]
            self.n_bins.append(len(prototypes))

            # compute quantization error
            bins = (grid[:-1] + grid[1:]) / 2.0
            # bin assignment for each distance
            bin_idxs = np.digitize(norm_dist, bins)
            bin_idxs -= 1
            bin_idxs[bin_idxs == -1] = 0
            bin_idxs[bin_idxs == len(prototypes)] = len(prototypes) - 1

            # theoretically derived criterion
            # err = n * np.log(2 * np.pi) + \
            #     np.linalg.norm(distances - prototypes[bin_idxs])**2 + \
            #     len(prototypes) * np.log(n)
            # quantization error with BIC model selection
            # adhoc version
            n = len(norm_dist)
            err = np.linalg.norm((norm_dist - prototypes[bin_idxs]))
            self.errors[0].append(err)

            if use_bic:
                err = err + len(prototypes) * np.log(n)
            self.errors[1].append(err)


            if err < smallest_err:
                smallest_err, best_distance = err, d

        return best_distance

    def best_corners(self, distance: float) -> np.ndarray:
        """ select corners that match the given distance """
        if self.unique_dists is None:
            self.reset()

        norm_dist = self.distances / distance
        grid = np.arange(0, np.max(self.distances) + distance, distance) / distance
        prototypes = grid[1:-1]

        # compute quantization error
        bins = (grid[:-1] + grid[1:]) / 2.0
        # bin assignment for each distance
        bin_idxs = np.digitize(norm_dist, bins)
        bin_idxs -= 1
        bin_idxs[bin_idxs == -1] = 0
        bin_idxs[bin_idxs == len(prototypes)] = len(prototypes) - 1

        errs = (norm_dist - prototypes[bin_idxs])**2

        mask = np.zeros_like(self.distances, dtype=bool)
        mask[errs == errs.min()] = True

        mask_matrix0 = squareform(mask)
        mask_matrix = np.logical_and(mask_matrix0, np.tri(len(self.corners)))

        corner_pairs = np.where(mask_matrix)

        corner_idxs = sorted(set(corner_pairs[0]) | set(corner_pairs[1]))

        return corner_idxs, corner_pairs


    def plot_errors(self, **kwargs)-> T.Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(**kwargs)

        ax.plot(self.unique_dists, self.errors[0], label="err0")
        ax.plot(self.unique_dists, self.errors[1], label="err1")
        ax = ax.twinx()
        ax.plot(self.unique_dists, np.log(self.n_bins), linestyle="dashed",
                label="# bins")

        return fig, ax
