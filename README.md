# scalebar - Processing tools for scale bars in images
*Author: Dimitri Korsch*

This package is meant to estimate the scale of a standardized scale bar in images captured by automatic light traps.
An example of the visualization output can be seen below.
The main function `scalebar.get_scale` outputs for a given image a float number representing the scale of the image in pixels per mm.
This scale can later be used to estimate the size of the objects, insects, or animals in the picture.

<img src="example.png">

## Installation

*python >= 3.7 required. Tested on Ubuntu 18.04 and Python v3.9 so far*

### With `pip`

```bash
pip install scalebar
```

### From Source

```bash
git clone https://github.com/cvjena/scalebar
cd scalebar
# create a new environment, if you want (e.g. with conda create -n scalebar python~=3.9.0)
pip install -r requirements.txt
```

then either

```bash
make install
```

or

```bash
pip install .
```

or
```bash
python setup.py install
```

## Usage

### In your code

```python
import cv2
import scalebar

res = scalebar.Result.new(args.image_path)
size_per_square = res.size_per_square # mm in each square (default: 1.0). Can be set in Result.new(image_path, size_per_square=size)
px_per_square = res.px_per_square #
px_per_mm = res.scale # computed as px_per_square / size_per_square
# process the numbers as you want
```

### From the console

```bash
python -m scalebar.estimate path/to/image.jpg
# outputs the scale to stdout
42.0
```

```bash
python -m scalebar.estimate path/to/image.jpg -o output.txt
# writes the scale to output.txt
```


```bash
# requires cvargparse, matplotlib, pyqt5
python -m scalebar.visualize path/to/image.jpg
# visualizes the estimation process

python -m scalebar.visualize path/to/image.jpg -o vis.png
# stores the visualization in a file
```

## Licence
This work is licensed under a [GNU Affero General Public License][agplv3].

[![AGPLv3][agplv3-image]][agplv3]

[agplv3]: https://www.gnu.org/licenses/agpl-3.0.html
[agplv3-image]: https://www.gnu.org/graphics/agplv3-88x31.png
