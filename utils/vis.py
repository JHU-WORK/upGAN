#!/usr/bin/env python3

import random
import matplotlib.pyplot as plt

def calculate_point(image_dim, randomize=True):
    """
    Calculates a random or fixed point within an image.

    Parameters:
    image_dim (tuple): A tuple (height, width) of the image dimensions.
    randomize (bool): Flag to randomize the point.

    Returns:
    Tuple of (point_x, point_y) as coordinates within the image.
    """
    height, width = image_dim
    if randomize:
        point_x = random.randint(0, width - 1)
        point_y = random.randint(0, height - 1)
    else:
        point_x, point_y = width // 2, height // 2  # Fixed central point

    return point_x, point_y

# Functions display images
def visualize_image(img):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 1, 1)
    plt.title('Resolution')
    plt.imshow(img.permute(1, 2, 0))

    plt.show()

def visualize_data(input_img, label_img):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title('Low Resolution')
    plt.imshow(input_img.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.title('High Resolution')
    plt.imshow(label_img.permute(1, 2, 0))

    plt.show()

def visualize_images(input_img, output_img, label_img):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('Low Resolution')
    plt.imshow(input_img.permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.title('Output Resolution')
    plt.imshow(output_img.permute(1, 2, 0))

    plt.subplot(1, 3, 3)
    plt.title('High Resolution')
    plt.imshow(label_img.permute(1, 2, 0))

    plt.show()

# Functions to zoom and display images
def zoom_and_display_infer(image, num_zooms, randomize_point=True):
    img_height, img_width = image.shape[-2], image.shape[-1]
    
    # Calculate random or fixed point for zooming
    zoom_point_x, zoom_point_y = calculate_point((img_height, img_width), randomize_point)

    # Ensure zoom window maintains aspect ratio
    aspect_ratio = img_width / img_height

    # Iterate through zoom levels
    for zoom_level in range(num_zooms):
        # Calculate zoom window size, maintaining aspect ratio
        zoom_window_width = int(img_width // (2 ** zoom_level))
        zoom_window_height = int(zoom_window_width / aspect_ratio)

        # Calculate start and end coordinates for the zoom window
        start_x = max(zoom_point_x - zoom_window_width // 2, 0)
        start_y = max(zoom_point_y - zoom_window_height // 2, 0)
        end_x = min(start_x + zoom_window_width, img_width)
        end_y = min(start_y + zoom_window_height, img_height)

        # Plotting the zoomed image
        plt.figure(figsize=(8, 8))
        plt.title(f'Zoom Level {zoom_level + 1}')
        plt.imshow(image.permute(1, 2, 0)[start_y:end_y, start_x:end_x, :])
        plt.show()

def zoom_and_display_data(low_res, high_res, scale_factor, num_zooms, randomize_point=True):
    lr_height, lr_width = low_res.shape[-2], low_res.shape[-1]
    
    # Calculate random or fixed point for zooming
    zoom_point_x, zoom_point_y = calculate_point((lr_height, lr_width), randomize_point)

    # Ensure zoom window maintains aspect ratio
    aspect_ratio = lr_width / lr_height

    # Iterate through zoom levels
    for zoom_level in range(num_zooms):
        # Calculate zoom window size, maintaining aspect ratio
        zoom_window_width = int(lr_width // (2 ** zoom_level))
        zoom_window_height = int(zoom_window_width / aspect_ratio)

        # Calculate start and end coordinates for low-resolution image
        lr_start_x = max(zoom_point_x - zoom_window_width // 2, 0)
        lr_start_y = max(zoom_point_y - zoom_window_height // 2, 0)
        lr_end_x = min(lr_start_x + zoom_window_width, lr_width)
        lr_end_y = min(lr_start_y + zoom_window_height, lr_height)

        # Calculate corresponding coordinates for high-resolution image
        hr_start_x = int(lr_start_x * scale_factor)
        hr_start_y = int(lr_start_y * scale_factor)
        hr_end_x = int(lr_end_x * scale_factor)
        hr_end_y = int(lr_end_y * scale_factor)

        # Plotting the images
        plt.figure(figsize=(15, 15))

        # Low-resolution image
        plt.subplot(1, 2, 1)
        plt.title(f'Low Resolution (Zoom Level {zoom_level + 1})')
        plt.imshow(low_res.permute(1, 2, 0)[lr_start_y:lr_end_y, lr_start_x:lr_end_x, :])

        # High-resolution image
        plt.subplot(1, 2, 2)
        plt.title(f'High Resolution (Zoom Level {zoom_level + 1})')
        plt.imshow(high_res.permute(1, 2, 0)[hr_start_y:hr_end_y, hr_start_x:hr_end_x, :])

        plt.show()

def zoom_and_display_results(low_res, up_res, high_res, scale_factor, num_zooms, randomize_point=True):
    lr_height, lr_width = low_res.shape[-2], low_res.shape[-1]
    hr_height, hr_width = high_res.shape[-2], high_res.shape[-1]
    
    # Calculate random or fixed point for zooming
    zoom_point_x, zoom_point_y = calculate_point((lr_height, lr_width), randomize_point)

    # Ensure zoom window maintains aspect ratio
    lr_aspect_ratio = lr_width / lr_height
    hr_aspect_ratio = hr_width / hr_height

    # Iterate through zoom levels
    for zoom_level in range(num_zooms):
        # Calculate zoom window size, maintaining aspect ratio
        lr_zoom_window_width = int(lr_width // (2 ** zoom_level))
        lr_zoom_window_height = int(lr_zoom_window_width / lr_aspect_ratio)

        hr_zoom_window_width = int(hr_width // (2 ** zoom_level))
        hr_zoom_window_height = int(hr_zoom_window_width / hr_aspect_ratio)

        # Calculate start and end coordinates for low-resolution image
        lr_start_x = max(zoom_point_x - lr_zoom_window_width // 2, 0)
        lr_start_y = max(zoom_point_y - lr_zoom_window_height // 2, 0)
        lr_end_x = min(lr_start_x + lr_zoom_window_width, lr_width)
        lr_end_y = min(lr_start_y + lr_zoom_window_height, lr_height)

        # Calculate corresponding coordinates for high-resolution and upscaled images
        hr_start_x = int(lr_start_x * scale_factor)
        hr_start_y = int(lr_start_y * scale_factor)
        hr_end_x = min(hr_start_x + hr_zoom_window_width, hr_width)
        hr_end_y = min(hr_start_y + hr_zoom_window_height, hr_height)

        # Plotting the images
        plt.figure(figsize=(15, 15))

        # Low-resolution image
        plt.subplot(1, 3, 1)
        plt.title(f'Low Resolution (Zoom Level {zoom_level + 1})')
        plt.imshow(low_res.permute(1, 2, 0)[lr_start_y:lr_end_y, lr_start_x:lr_end_x, :])

        # Upscaled image
        plt.subplot(1, 3, 2)
        plt.title(f'Upscaled Resolution (Zoom Level {zoom_level + 1})')
        plt.imshow(up_res.permute(1, 2, 0)[hr_start_y:hr_end_y, hr_start_x:hr_end_x, :])

        # High-resolution image
        plt.subplot(1, 3, 3)
        plt.title(f'High Resolution (Zoom Level {zoom_level + 1})')
        plt.imshow(high_res.permute(1, 2, 0)[hr_start_y:hr_end_y, hr_start_x:hr_end_x, :])

        plt.show()
