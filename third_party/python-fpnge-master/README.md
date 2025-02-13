This simple module provides Python bindings for the fast PNG encoder, [fpnge](https://github.com/animetosho/fpnge).

# Installation

This module has only been tested with Python3. Distutils is required (`apt install python3-distutils` under Debian based distros should grab it).

fpnge currently only runs on x86 processors, and requires SSE4.1 support at minimum.
Note: there is no dynamic CPU detection implemented, so this module should be installed/compiled on the machine it is going to run on.

This module can then be installed by running `python3 setup.py install` in the folder with the code.

# Example Usage

Exporting a PIL.Image to a PNG file.

```python
from PIL import Image
import fpnge

with Image.open('input.png') as im:
	png = fpnge.fromPIL(im)
	with open('output.png', 'wb') as f:
		f.write(png)
```

API
===

Note: fpnge does not support indexed color or bit depths below 8 bits/pixel.
Note: the *comp_level* parameter in the following functions is the fpnge compression level, which ranges from 1 to 5, defaulting at 4 (which is also aliased to 0).

fpnge.fromPIL(image [, comp_level])
------------------------------------------------------------------

Converts a PIL image specified in *image* to a PNG, returning it as a bytes object. If the image uses a palette, it will be converted to RGBA before saving.

fpnge.frombytes(bytes, width, height, channels, bits_per_channel [, comp_level] [, stride])
--------------------------------------------------------------------------------

Converts a raw image specified as a bytes object in *bytes* to a PNG, returning it as a bytes object.

The pixel dimensions of the supplied image must be specified in *width* and *height,* with *stride* being the number of bytes per row (if unspecified, assumes *stride* = *width* \* *channels* \* *bits_per_channel*/8).
*bits_per_channel* signifies the bit depth for each colour channel. Only 8 and 16 bits per channel are supported. Note that if 16 bits per channel is selected, samples must be big-endian.
*channels* refers to the number of colour channels the image has, with the following assumption:

* 1 channel: this is a greyscale image
* 2 channels: this is a greyscale + alpha image. Channels are interleaved with greyscale being first, followed by alpha.
* 3 channels: RGB colour. Channel interleaving order is: red, green, blue.
* 4 channels: RGBA colour. Channel interleaving order is: red, green, blue, alpha.

No other values are allowed for *channels*

fpnge.fromNP(ndarray [, comp_level])
--------------------------------------------------------------------------------

Converts a raw image stored in a 3-dimensional NumPy *ndarray* to a PNG, returning it as a bytes object.

The dimensions must be height, width and colour channels (see `frombytes` function above for how the channel count is interpreted and channel components are ordered).  
The element type should be a `uint8` or `'>u2'`.

**Note:** This differs from the Pillow definition of `shape` which is `(width, height, channels)`

## fpnge.fromview(view [, width] [, height] [, channels] [, bits_per_channel] [, comp_level] [, stride])

The `memoryview` version of `frombytes`, where *view* is a C-contiguous `memoryview`.

If other arguments are not supplied, assumes *view* is a 3-dimensional structure (width x height x channels) and derives the values from there.

# See Also

[Python bindings](https://github.com/qrmt/fpng-python) for the other speed-optimized PNG writer, [fpng](https://github.com/richgel999/fpng).

License
=======

This module is Public Domain or [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) (or equivalent) if PD isn’t recognised.

fpnge itself is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)