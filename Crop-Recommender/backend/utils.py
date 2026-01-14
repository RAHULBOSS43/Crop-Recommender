import cv2
import numpy as np
import requests

def load_image_from_anywhere(image_url=None, image_file=None):
    if image_url:
        response = requests.get(image_url)
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    elif image_file:
        img = cv2.imdecode(
            np.frombuffer(image_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
    else:
        img = None

    if img is None:
        raise ValueError("Image could not be loaded")

    return img
