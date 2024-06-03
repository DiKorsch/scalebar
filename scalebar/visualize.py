#!/usr/bin/env python
if __name__ != '__main__':
    raise Exception("Do not import me!")

import cv2
import logging
import scalebar

from scalebar import utils

with utils.try_import("cvargparse"):
    from cvargparse import Arg
    from cvargparse import BaseParser

with utils.try_import("matplotlib, pyqt5"):
    from matplotlib import pyplot as plt


def main(args) -> None:
    fig = plt.figure(figsize=(16,9))

    im = utils.read_image(args.image_path)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    H, W, *C = im.shape

    grid = plt.GridSpec(2, 2)

    ax = plt.subplot(grid[0, 0])
    ax.axis("off")
    ax.imshow(hsv[..., 2], cmap=plt.cm.gray)
    ax.set_title("Input image")

    # positions = {pos.name.lower(): pos for pos in scalebar.Position}
    # pos = positions[args.position]
    # crop = pos.crop(im, x=args.crop_size[0], y=args.crop_size[1], square=args.crop_square)
    result = scalebar.get_scale(im,
                                # pos=pos,
                                fraction=args.fraction,
                                square_unit=args.unit,
                                max_corners=100,
                                smooth=False,
                                return_intermediate=True)

    if result is None:
        logging.info("Cannot estimate the scale in the given image!")

    px_per_mm, interm = result
    crop = interm["crop"]
    size = interm["size"]
    init_corners = interm["detected_corners"]
    mask = interm["filter_mask"]
    angle = interm["rectification_angle"]
    corners = interm["final_corners"]

    print(interm["distances"].shape)
    print(interm["unit_distance"], px_per_mm)
    ax = plt.subplot(grid[0, 1])
    ax.hist(interm["distances"], bins=30)

    ax = plt.subplot(grid[1, 0])
    ax.axis("off")

    # for pt in init_corners[mask]:
    #     cv2.circle(crop, pt[::-1], 3*size, (255, 0, 0), -1)

    # for pt in init_corners[~mask]:
    #     cv2.circle(crop, pt[::-1], 3*size, (255, 0, 0), -1)
    ax.imshow(crop, cmap=plt.cm.grey)
    ax.set_title("Original crop")

    ys, xs = init_corners[mask].transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red", label="used")

    ys, xs = init_corners[~mask].transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="blue", alpha=0.7, label="rejected")
    # ax.legend(loc="upper right")

    ax = plt.subplot(grid[1, 1])

    rot_mat = cv2.getRotationMatrix2D([0, 0], angle, 1.0)
    new_crop = cv2.warpAffine(crop, rot_mat, crop.shape[:2][::-1])

    # for pt in corners:
    #     cv2.circle(new_crop, pt[::-1], 3*size, 255, -1)
    ax.imshow(new_crop, cmap=plt.cm.grey)
    ax.axis("off")
    ax.set_title(f"Rectified crop ({angle=})")
    ys, xs = corners.transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red")

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
        plt.savefig(args.output)
    else:
        plt.show()
    plt.close()


parser = BaseParser([
    Arg("image_path"),

    # Arg("--position", "-pos", default="top_right",
    #     choices=[pos.name.lower() for pos in scalebar.Position]),

    Arg("--unit", "-u", type=float, default=1.0,
        help="Size of a single square in the scale bar (in mm). Default: 1"),
    Arg.float("--fraction", default=0.1),
    Arg.flag("--crop_square"),
    Arg("--output", "-o"),
])
main(parser.parse_args())
