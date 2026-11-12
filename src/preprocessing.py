"""
Brain MRI Preprocessing Pipeline
Contains skull-stripping (morphological boundary extraction) and tensor normalization.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional


def crop_brain_tumor(image: np.ndarray) -> np.ndarray:
    """
    Isolates brain parenchyma and strips cranial margins using
    morphological thresholding, erosion/dilation, and contour bounding.
    
    Args:
        image: BGR or RGB image as numpy array.
        
    Returns:
        Cropped numpy array containing the brain region of interest.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image provided to crop_brain_tumor.")

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # 5x5 Gaussian blur to attenuate noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Empirical binarization (tau = 45)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    # 2x erosion followed by 2x dilation to clean noise and restore margins
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contour extraction (Suzuki-Abe algorithm)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Largest contour corresponds to intracranial parenchyma
    c = max(contours, key=cv2.contourArea)

    # Find extreme cardinal points
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bottom = tuple(c[c[:, :, 1].argmax()][0])

    # Crop with safety checks
    ymin, ymax = ext_top[1], ext_bottom[1]
    xmin, xmax = ext_left[0], ext_right[0]

    # Ensure valid bounding box with minimum dimensions
    if (ymax - ymin < 20) or (xmax - xmin < 20):
        return image

    cropped = image[ymin:ymax, xmin:xmax]
    if cropped.size == 0:
        return image

    return cropped


def preprocess_image(
    image_input: Union[str, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (120, 120)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full clinical preprocessing pipeline:
    1. Ingestion from path, numpy array, or PIL image.
    2. Morphological skull-stripping (crop_brain_tumor).
    3. Resizing to target_size via bicubic interpolation.
    4. Min-max normalization [0.0, 1.0].
    5. Batch dimension expansion.

    Returns:
        tensor: np.ndarray of shape (1, H, W, 3) normalized to [0, 1].
        cropped_rgb: np.ndarray of cropped image in RGB for UI display.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Unable to read image file: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Apply morphological skull stripping
    # Convert RGB to BGR for cv2 operations inside crop_brain_tumor
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cropped_bgr = crop_brain_tumor(img_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Bicubic resizing for feature fidelity
    resized = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_CUBIC)

    # Min-max intensity scaling
    normalized = resized.astype(np.float32) / 255.0

    # Expand batch dimension
    tensor = np.expand_dims(normalized, axis=0)

    return tensor, cropped_rgb

