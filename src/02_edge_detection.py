"""Generate candidate edge masks with six classical operators."""

import os

import cv2
import numpy as np


PROCESSED_DIR = "data/processed"
MASKS_DIR = "results/masks"
OPERATORS = ("canny", "laplacian", "sobel", "prewitt", "roberts", "fft")


def detect_canny(image):
    """Apply Canny edge detection."""
    smoothed = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.Canny(smoothed, 50, 150)


def detect_laplacian(image):
    """Apply Laplacian edge detection."""
    laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
    magnitude = np.abs(laplacian)
    threshold = np.percentile(magnitude, 98.5)
    return np.where(magnitude >= max(threshold, 1.0), 255, 0).astype(np.uint8)


def detect_sobel(image):
    """Apply Sobel edge detection in the x and y directions."""
    sobel_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold = np.percentile(magnitude, 97.0)
    return np.where(magnitude >= max(threshold, 1.0), 255, 0).astype(np.uint8)


def detect_prewitt(image):
    """Apply Prewitt edge detection with custom kernels."""
    kernel_x = np.array(
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32
    )
    kernel_y = np.array(
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32
    )
    gradient_x = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_x)
    gradient_y = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    threshold = np.percentile(magnitude, 97.0)
    return np.where(magnitude >= max(threshold, 1.0), 255, 0).astype(np.uint8)


def detect_roberts(image):
    """Apply Roberts cross edge detection."""
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gradient_x = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_x)
    gradient_y = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    threshold = np.percentile(magnitude, 97.0)
    return np.where(magnitude >= max(threshold, 1.0), 255, 0).astype(np.uint8)


def detect_fft(image):
    """Apply a high-pass filter in the frequency domain."""
    spectrum = np.fft.fft2(image.astype(np.float64))
    shifted_spectrum = np.fft.fftshift(spectrum)

    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) * 0.12

    y_coordinates, x_coordinates = np.ogrid[:height, :width]
    distance = np.sqrt(
        (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2
    )
    high_pass_mask = 1.0 - np.exp(-(distance**2) / (2 * radius**2))

    filtered_spectrum = shifted_spectrum * high_pass_mask
    inverse_shifted = np.fft.ifftshift(filtered_spectrum)
    reconstructed = np.abs(np.fft.ifft2(inverse_shifted))

    threshold = np.percentile(reconstructed, 99.0)
    return np.where(reconstructed >= max(threshold, 1e-8), 255, 0).astype(
        np.uint8
    )


EDGE_DETECTORS = {
    "canny": detect_canny,
    "laplacian": detect_laplacian,
    "sobel": detect_sobel,
    "prewitt": detect_prewitt,
    "roberts": detect_roberts,
    "fft": detect_fft,
}


def clean_mask(binary_mask):
    """Connect small gaps without removing one-pixel cracks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def run():
    """Apply all edge operators to every processed image."""
    if not os.path.exists(PROCESSED_DIR):
        print(
            f"  error: processed image directory {PROCESSED_DIR} does not exist; "
            "run 01_preprocessing.py first"
        )
        return

    image_names = [
        name
        for name in os.listdir(PROCESSED_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_names:
        print(f"  error: no processed images found in {PROCESSED_DIR}")
        return

    print(
        f"  detecting edges in {len(image_names)} images "
        f"with {len(OPERATORS)} operators..."
    )
    for operator in OPERATORS:
        os.makedirs(os.path.join(MASKS_DIR, operator), exist_ok=True)

    processed_count = 0
    for image_name in image_names:
        image_path = os.path.join(PROCESSED_DIR, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [!] could not read {image_name}")
            continue

        try:
            for operator, detector in EDGE_DETECTORS.items():
                mask = clean_mask(detector(image))
                output_path = os.path.join(MASKS_DIR, operator, image_name)
                cv2.imwrite(output_path, mask)
            processed_count += 1
        except Exception as error:
            print(f"  [!] edge detection failed for {image_name}: {error}")

    print(
        f"  [OK] saved masks for {processed_count}/{len(image_names)} "
        f"images to {MASKS_DIR}/<operator>/"
    )


if __name__ == "__main__":
    run()
