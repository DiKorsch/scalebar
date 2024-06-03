#!/usr/bin/env python
if __name__ != '__main__':
    raise Exception("Do not import me!")

import os
import scalebar

from scalebar import utils

with utils.try_import("cvargparse"):
    from cvargparse import Arg
    from cvargparse import BaseParser


def main(args):

    res = scalebar.Result.new(args.image_path,
                              roi_fraction=args.fraction,
                              size_per_square=args.unit)

    px_per_mm = res.scale

    if args.output:
        assert not os.path.exists(args.output), \
            f"Output file ({args.output}) already exists!"
        with open(args.output, "w") as f:
            f.write(f"{px_per_mm:f}\n")
    else:
        print(px_per_mm)


parser = BaseParser([
    Arg("image_path"),

    Arg.float("--unit", "-u", default=1.0,
              help="Size of a single square in the scale bar (in mm). Default: 1"),

    Arg.float("--fraction", default=0.1,
              help="Fraction of the image's border that will be used for the scale estimation. Default: 0.1"),

    Arg("--output", "-o")
])

main(parser.parse_args())
