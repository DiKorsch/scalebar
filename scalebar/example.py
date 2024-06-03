#!/usr/bin/env python
if __name__ != '__main__':
    raise Exception("Do not import me!")

import scalebar
import cv2
import numpy as np
import logging
import skimage

from cvargparse import Arg
from cvargparse import BaseParser
from matplotlib import pyplot as plt

from scalebar import utils
from pathlib import Path

BLACK = 0
GRAY = 127
WHITE = 255

def main(args):
    impath = Path(args.image_path)
    im = utils.read_image(args.image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


    thresh, bin_im = cv2.threshold(gray, GRAY, WHITE,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    H, W, C = im.shape
    y0, y1 = int(H*args.fraction), int(H*(1-args.fraction))
    x0, x1 = int(W*args.fraction), int(W*(1-args.fraction))
    bin_im[y0:y1, x0:x1] = WHITE

    size = max(int(min(H, W) * 2e-3), 1)
    kernel_size = size * 2 + 1
    kernel_shape = (kernel_size, kernel_size)

    logging.info(f"Image size: {im.shape}")
    logging.info(f"Using kernel size: {kernel_shape}")
    # min_dist = size + 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape, (size, size))
    bin_im = cv2.dilate(cv2.erode(bin_im, kernel=kernel), kernel=kernel)


    # contours, hier = cv2.findContours(match, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # corners = []
    # # print('found', len(contours), 'contours')
    # # draw a bounding box around each contour
    # for contour in contours:
    #     x,y,w,h = cv2.boundingRect(contour)
    #     corners.append([[x+w//2, y+h//2]])
    #     cv2.rectangle(match, (x,y), (x+w,y+h), GRAY, 2)

    # corners = np.array(corners, dtype=np.float32)
    # print(corners.shape)
    # # OpenCV's corner/feature detector
    corners = cv2.goodFeaturesToTrack(bin_im, 0, 0.1,
                                      minDistance=size,
                                      mask=None,
                                      blockSize=kernel_size,
                                      gradientSize=kernel_size,
                                      useHarrisDetector=True)
    # breakpoint()
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(bin_im, corners, kernel_shape, (-1,-1), criteria)
    # switch x and y coordinates for convenience (needed only for OpenCV)
    corners = corners[:, 0, ::-1].astype(int)

    temp_size = size*5
    template = np.full((2*temp_size, 2*temp_size), BLACK, dtype=bin_im.dtype)
    template[:temp_size, temp_size:] = WHITE
    template[temp_size:, :temp_size] = WHITE

    match = np.abs(cv2.matchTemplate(bin_im, template, method=cv2.TM_CCOEFF_NORMED))
    match = np.pad(match, ((temp_size, temp_size-1), (temp_size, temp_size-1)))
    match -= match.min()
    match /=  (match.max() or 1)
    match = (match * 255).astype(np.uint8)

    print(match.shape, bin_im.shape, template.shape)

    _, match = cv2.threshold(match, GRAY, WHITE,
                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    match = cv2.erode(match, None, iterations=2)
    peaks = skimage.feature.peak_local_max(match, min_distance=temp_size)
    for pt in peaks:
        cv2.circle(match, pt[::-1], radius=temp_size, color=GRAY)
        cv2.circle(bin_im, pt[::-1], radius=2*size, color=GRAY, thickness=-1)

    print(len(corners))
    # for pt in corners:
    #     cv2.circle(bin_im, pt[::-1], 3*size, 127)

    fig, axs = plt.subplots(ncols=3)
    [ax.axis("off") for ax in axs.ravel()]

    axs[0].imshow(im)
    axs[1].imshow(gray, cmap=plt.cm.gray)
    axs[2].imshow(bin_im, cmap=plt.cm.gray)

    plt.tight_layout()
    fig.savefig(f"vis/{impath.stem}.vis.png")
    cv2.imwrite(f"vis/{impath.stem}.bin_im.png", bin_im)
    cv2.imwrite(f"matches/{impath.stem}.match.png", match)


parser = BaseParser([
    Arg("image_path"),

    Arg("--position", "-pos", default="top_right",
        choices=[pos.name.lower() for pos in scalebar.Position]),

    Arg("--unit", "-u", type=float, default=1.0,
        help="Size of a single square in the scale bar (in mm). Default: 1"),
    Arg.float("--fraction", "-frac", default=0.1),
    # Arg.float("--crop_size", nargs=2, default=(0.2, 0.2)),
    Arg.flag("--crop_square"),
    Arg("--output", "-o"),
])
main(parser.parse_args())
