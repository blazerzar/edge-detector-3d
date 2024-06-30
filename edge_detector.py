from os import makedirs, path
from sys import argv

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def main():
    if len(argv) < 3:
        print(
            'Assignment 2: Edge detection\n'
            'Program detects edges in the given sequence of CT images. It\n'
            'saves binarised output images into .png files in the directory\n'
            'results. At the same time it displays the sequence of original\n'
            'images, images after edges linking and images after linking\n'
            'between consecutive images.\n\n'
            'Usage: python edge_detector.py <directory> <image_1> [<image_2> ...]\n'
            'Examples:\n'
            '\tpython edge_detector.py . 0001.png 0002.png\n'
            '\tpython edge_detector.py ../databases/CT/Patient-01/ 0001.png 0002.png'
        )
        exit(1)

    directory = argv[1]
    files = argv[2:]

    images = [read_image(path.join(directory, file)) for file in files]

    edges = [
        canny(image, t_low=0.15, t_high=0.3, method='analytic') for image in images
    ]
    linked = link_edges(edges)

    # Save results
    if not path.exists('results'):
        makedirs('results')
    for file, edges_ in zip(files, linked):
        plt.imsave(path.join('results', file), edges_, cmap='gray')

    linked = [l - e for l, e in zip(linked, edges)]

    # Display images
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    # Display the first set of images
    ax[0].set_title('Original')
    ax[0].axis('off')
    im1 = ax[0].imshow(images[0], cmap='gray')
    ax[1].set_title('Canny edges')
    ax[1].axis('off')
    im2 = ax[1].imshow(edges[0], cmap='gray')
    ax[2].set_title('3D Linked edges')
    ax[2].axis('off')
    im3 = ax[2].imshow(edges[0], cmap='gray')
    im4 = ax[2].imshow(linked[0], cmap=mcolors.ListedColormap(['none', 'yellow']))
    im4.set_clim(0, 1)

    # Create a slider
    ax_slider = plt.axes([0.15, 0.1, 0.3, 0.03])
    slider = Slider(ax_slider, 'Slice Index ', 0, len(images) - 1, valinit=0, valstep=1)

    ax_prev = plt.axes([0.73, 0.1, 0.1, 0.04])
    ax_next = plt.axes([0.85, 0.1, 0.1, 0.04])
    button_prev = Button(ax_prev, 'Previous')
    button_next = Button(ax_next, 'Next')

    # Update function for the slider
    def update(val):
        index = int(slider.val)
        im1.set_data(images[index])
        im2.set_data(edges[index])
        im3.set_data(edges[index])
        im4.set_data(linked[index])

        fig.canvas.draw_idle()

    # Button on click functions
    def prev(event):
        current_val = slider.val
        if current_val > 0:
            slider.set_val(current_val - 1)

    def next(event):
        current_val = slider.val
        if current_val < len(images) - 1:
            slider.set_val(current_val + 1)

    slider.on_changed(update)
    button_prev.on_clicked(prev)
    button_next.on_clicked(next)

    plt.show()


def read_image(file):
    image = plt.imread(file)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype == np.uint8:
        image = image / 255
    return image


def gaussian(image):
    """Smooth image using a Gaussian filter."""
    sigma = np.maximum(np.min(image.shape) * 0.005, 0.5)
    N = np.round(3 * sigma).astype(int)
    kernel = cv2.getGaussianKernel(2 * N + 1, sigma)

    return cv2.filter2D(cv2.filter2D(image, -1, kernel), -1, kernel.T)


def prewitt(image):
    """Compute derivatives in x and y directions using Prewitt operator on
    top of the smoothed image."""
    kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return (
        cv2.filter2D(image, -1, kernel),
        cv2.filter2D(image, -1, kernel.T),
    )


def sobel(image):
    """Compute derivatives in x and y directions using Sobel operator on
    top of the smoothed image."""
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return (
        cv2.filter2D(image, -1, kernel),
        cv2.filter2D(image, -1, kernel.T),
    )


def analytic(image):
    """Compute derivates in x and y directions using analytic derivative
    of a Gaussian."""
    sigma = np.maximum(np.min(image.shape) * 0.005, 0.5)
    N = np.round(3 * sigma).astype(int)

    # Get 2D analytically derived Gaussian kernel
    x = np.linspace(-N, N, 2 * N + 1)
    y, x = np.meshgrid(x, x)
    kernel = -x * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.abs(kernel).sum()

    return (
        cv2.filter2D(image, -1, kernel),
        cv2.filter2D(image, -1, kernel.T),
    )


def nonmaxima(magnitude, angle):
    """Suppress magnitude values that are not local maxima in the direction
    of the gradient."""
    magnitude = np.pad(magnitude, 1, mode='reflect')
    angle = (np.mod(angle - np.pi / 8, np.pi) / np.pi * 4).astype(int)

    suppressed = np.zeros_like(magnitude)
    for (i, j), angle in np.ndenumerate(angle):
        mag = magnitude[i + 1, j + 1]

        # Get values to compare depending on angle:
        # - top left and bottom right: (i, j) and (i + 2, j + 2)
        # - top and bottom: (i, j + 1) and (i + 2, j + 1)
        # - bottom left and top right: (i + 2, j) and (i, j + 2)
        # - left and right: (i + 1, j) and (i + 1, j + 2)
        left = magnitude[[i, i, i + 2, i + 1], [j, j + 1, j, j]][angle]
        right = magnitude[[i + 2, i + 2, i, i + 1], [j + 2, j + 1, j + 2, j + 2]][angle]

        # keep = mag > left and mag >= right or mag >= left and mag > right
        keep = mag >= left and mag >= right
        suppressed[i, j] = mag if keep else 0

    return suppressed[1:-1, 1:-1]


def hysteresis(magnitude, t_low, t_high):
    """Apply hysteresis thresholding to non-maxima suppressed image.

    Parameters:
        - magnitude: np.array of shape (height, width) representing
                     gradient magnitude after non-maxima suppression
        - t_low: float representing low threshold
        - t_high: float representing high threshold

    Returns:
        - binary np.array of shape (height, width)
    """
    # Remove non-edge pixels and find strong edge pixels
    magnitude[magnitude < t_low] = 0
    high_pixels = magnitude > t_high

    # Work with integer values [0, 255] because of connected components
    magnitude = (magnitude * 255).round().astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(magnitude, 8, cv2.CV_32S)

    # Find labels of strong edge pixels
    high_labels = set((high_pixels * labels).flatten()) - {0}

    # Get weak edge pixels that are connected to strong edge pixels
    to_keep = np.isin(labels, list(high_labels))

    return to_keep.astype(np.uint8)


def canny(image, method='analytic', t_low=0.05, t_high=0.15):
    """Detect edges in an image using the Canny algorithm.

    Parameters:
        - image: np.array of shape (height, width) representing grayscale image

    Returns:
        - binary np.array
    """
    if method == 'prewitt':
        image_dy, image_dx = prewitt(gaussian(image))
    elif method == 'sobel':
        image_dy, image_dx = sobel(gaussian(image))
    elif method == 'analytic':
        image_dy, image_dx = analytic(image)
    else:
        raise ValueError('Invalid method')

    magnitude = np.sqrt(image_dx**2 + image_dy**2)
    angle = np.arctan2(image_dy, image_dx)

    # Normalise gradient magnitude to [0, 1] regardless of the method
    magnitude /= magnitude.max()

    s = nonmaxima(magnitude, angle)
    h = hysteresis(s, t_low, t_high)

    return h


def link_consecutive_edges(current, next):
    """Link edges between consecutive images."""
    next = np.pad(next, 2, mode='constant')

    for i, j in zip(*current.nonzero()):
        # (i, j) -> (i + 2, j + 2)
        # Check pixel above
        if next[i + 2, j + 2] == 1:
            continue

        # Check 3x3 neighbourhood above
        if next[i + 1 : i + 4, j + 1 : j + 4].sum() >= 1:
            continue

        # Check 5x5 neighbourhood above
        if next[i : i + 5, j : j + 5].sum() == 0:
            continue

        # Link pixels
        ys, xs = next[i : i + 5, j : j + 5].nonzero()
        ys += i
        xs += j

        for y, x in zip(ys, xs):
            steps = np.maximum(np.abs(i + 2 - y), np.abs(j + 2 - x))
            dy = np.sign(i + 2 - y)
            dx = np.sign(j + 2 - x)

            add_x = np.array([j + 2] * steps)
            add_y = np.array([i + 2] * steps)
            if x != j + 2:
                add_x[: np.abs(j + 2 - x)] = np.arange(x, j + 2 + dx, dx)[1:]
            if y != i + 2:
                add_y[: np.abs(i + 2 - y)] = np.arange(y, i + 2 + dy, dy)[1:]

            next[add_y, add_x] = 1

    return next[2:-2, 2:-2]


def link_edges(edge_images):
    """Link edges from multiple edge images."""
    return [edge_images[0]] + [
        link_consecutive_edges(current, next)
        for current, next in zip(edge_images[:-1], edge_images[1:])
    ]


if __name__ == '__main__':
    main()
