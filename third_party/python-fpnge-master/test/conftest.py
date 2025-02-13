from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pytest
import os

print(f"PID TO DEBUG: {os.getpid()}")

_BASE_PATH = Path(__file__).parent / "testfiles"
_PNG_8BIT_PATH = _BASE_PATH / "8b.png"
_PNG_12BIT_PATH = _BASE_PATH / "12b.png"


@pytest.fixture
def cv2_8bit_png():
    image = cv2.imread(str(_PNG_8BIT_PATH), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    yield image


@pytest.fixture
def cv2_12bit_png():
    image = cv2.imread(str(_PNG_12BIT_PATH), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert image.shape[2] == 3
    assert image.dtype == np.uint16
    yield image


@pytest.fixture
def pillow_8bit_png():
    image = Image.open(str(_PNG_8BIT_PATH))
    image.load()
    yield image


# note: Pillow CANNOT load 16b PNG
