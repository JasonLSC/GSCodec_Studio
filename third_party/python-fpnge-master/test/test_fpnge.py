from PIL import Image
import cv2
import numpy as np
from numpy.typing import NDArray
import fpnge
import io


def helper_read_png_bytes_to_ndarray_cv2(image_bytes: bytes) -> NDArray:
    image: cv2.Mat = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    # OpenCV automatically reads as BGR, but this flips PNG which stores as RGB, so we have to read it back
    nparray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # OpenCV uses little-endian, but PNG uses big-endian and so if >8bit dtype, it will falsely be reported as little endian
    if nparray.dtype == np.uint16:  # This will be mislabelled!
        # Change to big-endian, openCV uses little endian by default
        nparray = nparray.astype(">u2")
    return nparray  # Leave as cv2.Mat, which is a superset of NDArray


def test_8bit_pillow(pillow_8bit_png: Image.Image):
    image_bytes: bytes = fpnge.fromPIL(pillow_8bit_png)
    image_check: cv2.Mat = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    assert image_check.dtype == np.uint8
    assert image_check.shape[0] == pillow_8bit_png.height
    assert image_check.shape[1] == pillow_8bit_png.width
    assert image_check.shape[2] == 3


def test_uint8_pillow_cv2_readback(pillow_8bit_png: Image.Image):
    # Ensure reading back from pillow and cv2 matches
    np_image = np.asarray(pillow_8bit_png, np.uint8)
    image_bytes: bytes = fpnge.fromPIL(pillow_8bit_png)

    # Open with pillow
    image_bytes_io = io.BytesIO(image_bytes)
    test = Image.open(image_bytes_io)
    image_check = np.asarray(test, np.uint8)
    assert image_check.dtype == np_image.dtype
    assert image_check.shape == np_image.shape
    assert (image_check == np_image).all()
    del image_check

    image_check: cv2.Mat = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    # Need to corrct for RGB/BGR
    image_check = cv2.cvtColor(image_check, cv2.COLOR_BGR2RGB)
    assert image_check.dtype == np_image.dtype
    assert image_check.shape == np_image.shape
    assert (image_check == np_image).all()


def test_8bit_cv2(cv2_8bit_png: cv2.Mat):
    # image_bytes = fpnge.fromMat(cv2_8bit_png)
    image_bytes = fpnge.fromMat(cv2_8bit_png)
    image_check: cv2.Mat = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    assert image_check.dtype == cv2_8bit_png.dtype
    assert image_check.shape == cv2_8bit_png.shape
    assert (image_check == cv2_8bit_png).all()


def test_12bit_cv2(cv2_12bit_png: cv2.Mat):
    image_bytes = fpnge.fromMat(cv2_12bit_png)
    image_check: cv2.Mat = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    assert image_check.dtype == cv2_12bit_png.dtype
    assert image_check.shape == cv2_12bit_png.shape
    assert (image_check == cv2_12bit_png).all()


def test_uint16_numpy_zeros():
    # Zeros
    blank_image = np.zeros((1, 2, 3), dtype=">u2")
    image_bytes = fpnge.fromNP(blank_image)
    image_bytes_io = io.BytesIO(image_bytes)
    test = Image.open(image_bytes_io)
    image_check = np.asarray(test, dtype=">u2")
    # image_check: cv2.Mat = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert image_check.dtype == blank_image.dtype
    assert image_check.shape == blank_image.shape
    assert (image_check == blank_image).all()


def test_uint16_numpy_max():
    # Max
    max_image = (np.ones((1, 2, 3), dtype=">u2") * np.iinfo(">u2").max).astype(">u2")
    image_bytes = fpnge.fromNP(max_image)
    image_check: cv2.Mat = helper_read_png_bytes_to_ndarray_cv2(image_bytes)
    assert image_check.dtype == max_image.dtype
    assert image_check.shape == max_image.shape
    assert (image_check == max_image).all()


def test_uint16_numpy_random():
    # Random
    # randint doesn't support non-native byteorder. But random number are still random, so just override order
    rand_image = np.random.randint(
        0, high=np.iinfo(">u2").max, size=(1, 2, 3), dtype=np.uint16
    ).astype(">u2")
    image_bytes = fpnge.fromNP(rand_image)
    # image_check: cv2.Mat = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image_check: cv2.Mat = helper_read_png_bytes_to_ndarray_cv2(image_bytes)
    assert image_check.dtype == rand_image.dtype
    assert image_check.shape == rand_image.shape
    assert (image_check == rand_image).all()
