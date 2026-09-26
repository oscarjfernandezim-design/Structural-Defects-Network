"""Convert input images to grayscale, resize them, and reduce noise."""

import os

import cv2


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGE_SIZE = (256, 256)


def median_filter(image, kernel_size=3):
    """Apply an OpenCV median filter."""
    return cv2.medianBlur(image, kernel_size)


def process_image(input_path, output_path):
    """Read and preprocess one image."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"  warning: could not read {input_path}")
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, IMAGE_SIZE)
    filtered = median_filter(resized, kernel_size=3)

    cv2.imwrite(output_path, filtered)
    return filtered


def run():
    """Process all images in the raw input directory."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_DIR):
        print(f"  error: input directory {RAW_DIR} does not exist")
        return False

    image_names = [
        name
        for name in os.listdir(RAW_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_names:
        print(f"  error: no images found in {RAW_DIR}")
        return False

    print(f"  processing {len(image_names)} images...")

    processed_count = 0
    for image_name in image_names:
        input_path = os.path.join(RAW_DIR, image_name)
        output_path = os.path.join(PROCESSED_DIR, image_name)
        try:
            result = process_image(input_path, output_path)
            if result is not None:
                processed_count += 1
        except Exception as error:
            print(f"  [!] failed to process {image_name}: {error}")

    print(
        f"  [OK] saved {processed_count}/{len(image_names)} "
        f"processed images to {PROCESSED_DIR}/"
    )
    return processed_count > 0


if __name__ == "__main__":
    run()
