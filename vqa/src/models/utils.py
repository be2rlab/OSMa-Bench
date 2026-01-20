import io
import base64

from pathlib import Path
from PIL import Image
import os


def load_image(image_path):
    """Reads and encodes an image as base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def encode_pil_image(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format = "PNG")
    img_byte_arr = img_byte_arr.getvalue()
    enc_data = base64.b64encode(img_byte_arr).decode("utf8")
    return enc_data


def encode_image(img_input):
    if isinstance(img_input, Image.Image):
        return encode_pil_image(img_input)

    if isinstance(img_input, (str, Path)):
        image_path = Path(img_input)
        if image_path.exists() and image_path.is_file():
            return load_image(image_path)
        else:
            raise FileNotFoundError(f"Invalid image path: {img_input}")

    raise TypeError(f"Unsupported input type: {type(img_input).__name__}")