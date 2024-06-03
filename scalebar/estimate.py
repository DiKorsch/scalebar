#!/usr/bin/env python
if __name__ != '__main__':
    raise Exception("Do not import me!")

import os
import scalebar

from scalebar import utils
from scalebar.core.size import Size

with utils.try_import("cvargparse"):
    from cvargparse import Arg
    from cvargparse import BaseParser


def main(args):
    res = scalebar.Result.new(args.image_path,
                              scalebar_size=Size.get(args.size),
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

    Arg("--size", default="MEDIUM", choices=["SMALL", "MEDIUM", "LARGE"],
              help="Rough apriori estimate of the scale bar size - SMALL: 10%, MEDIUM: 30%, and LARGE: 50%. Default: MEDIUM"),

    Arg("--output", "-o")
])

main(parser.parse_args())
