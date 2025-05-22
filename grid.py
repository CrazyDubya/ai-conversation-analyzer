import math
import os

import cv2
import numpy as np


def load_images(image_dir, size=(200, 300)):
    images = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, size)
            images.append(resized_image)
    return images


def create_image_grid(images, image_size=(200, 300)):
    img_width, img_height = image_size
    num_images = len(images)

    # Calculate the grid size
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    # Create a blank canvas
    stitched_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)

    # Place images on the canvas
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        stitched_image[y_start:y_end, x_start:x_end] = image

    return stitched_image


def main():
    image_dir = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/bookcovers'  # Replace with your image directory
    image_size = (200, 300)  # Define the size to which each image will be resized

    images = load_images(image_dir, size=image_size)

    # Check if we have enough images for the grid
    if not images:
        print("No images found in the directory.")
        return

    stitched_image = create_image_grid(images, image_size=image_size)

    # Save the stitched image
    output_path = 'stitched_bookshelf.jpg'
    cv2.imwrite(output_path, stitched_image)
    print(f'Stitched image saved as {output_path}')

    # Display the stitched image
    cv2.imshow('Stitched Image', stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
