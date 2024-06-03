#!/usr/bin/env python
if __name__ != '__main__':
    raise Exception("Do not import me!")

import numpy as np
import logging
import scalebar

from scalebar import utils

with utils.try_import("cvargparse"):
    from cvargparse import Arg
    from cvargparse import BaseParser

with utils.try_import("matplotlib, pyqt5"):
    from matplotlib import pyplot as plt

def imshow(ims):

    if len(ims) <= 3:
        nrows, ncols = 1, len(ims)
    else:
        nrows = int(np.ceil(np.sqrt(len(ims))))
        ncols = int(np.ceil( len(ims) / nrows))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(16,9), squeeze=False)
    for i, (title, im, cmap) in enumerate(ims):
        ax = axs[np.unravel_index(i, axs.shape)]

        if isinstance(im, (list, tuple)):
            alpha = 1 / len(im)
            for _im, _cm in zip(im, cmap):
                ax.imshow(_im, cmap=_cm, alpha=alpha)
        else:
            ax.imshow(im, cmap=cmap)
        ax.set_title(title)

    for _ in range(i+1, nrows*ncols):
        ax = axs[np.unravel_index(_, axs.shape)]
        ax.axis("off")

    return fig, axs

def main(args) -> None:
    logging.info(f"Processing {args.image_path}")
    res = scalebar.Result.new(args.image_path,
                              roi_fraction=args.fraction,
                              size_per_square=args.unit)

    images = res.images
    ROI = utils.hide_non_roi(images.binary, res.roi_fraction, 127)
    scalebar_crop = res.position.crop(images.equalized)
    px_per_mm = res.scale

    logging.info(f"Estimated Pixel per square: {res.px_per_square:.2f}")
    logging.info(f"Used size per square: {res.size_per_square:.2f}")
    logging.info(f"Result: {px_per_mm:.2f} px/mm | Image size: {res.image_size} | ")

    logging.info("Plotting results")
    fig, axs = imshow([
        ("Original", images.original, None),
        ("B/W image", images.gray, plt.cm.gray),
        ("B/W image equalized", images.equalized, plt.cm.gray),

        ("Binarized", images.binary, plt.cm.gray),
        ("ROI to be masked", ROI, plt.cm.gray),
        ("Masked", images.masked, plt.cm.gray),

        ("Template Matches", (
            images.binary,
            res.match), (plt.cm.gray, plt.cm.viridis)),
        (f"Scalebar | {res.scale} px/mm", scalebar_crop, plt.cm.gray),
    ])

    ax = axs[np.unravel_index(7, axs.shape)]
    ys, xs = res.distances.corners.transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red")

    W, H = res.image_size
    if px_per_mm is None:
        fig.suptitle("Estimation Failed!")
    else:
        size = W / px_per_mm, H / px_per_mm
        fig.suptitle(" | ".join(
            [
                f"{px_per_mm:.2f} px/mm",
                f"Image size: {size[0]:.2f} x {size[1]:.2f}mm"
            ]))

    plt.tight_layout()
    if args.output is not None:
        logging.info(f"Saving results to {args.output}")
        plt.savefig(args.output)
    else:
        plt.show()
    plt.close()


parser = BaseParser([
    Arg("image_path"),
    Arg("--unit", "-u", type=float, default=1.0,
        help="Size of a single square in the scale bar (in mm). Default: 1"),
    Arg.float("--fraction", default=0.1),
    Arg("--output", "-o"),
])
main(parser.parse_args())
