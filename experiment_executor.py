import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from image_processor import ImageProcessor

class ExperimentExecutor:
    def __init__(self, image_path, result_dir):
        """Initialize ExperimentExecutor with image path and result directory."""
        self.image_path = image_path
        self.result_dir = result_dir

    def run_experiment(self, test_name):
        """Run experiment for the specified test."""
        # Check if the image is in HEIC format and convert it to JPEG if necessary
        if self.image_path.lower().endswith('.heic'):
            color_image = self.convert_heic_to_jpeg(self.image_path)
        elif self.image_path.lower().endswith('.webp'):
            # Read the color image using PIL and convert it to OpenCV format
            with Image.open(self.image_path) as img:
                color_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            # Read the color image using OpenCV
            color_image = cv2.imread(self.image_path)

        # Convert color image to grayscale
        gray_image = ImageProcessor.convert_to_gray(color_image)

        # Perform histogram equalization
        equalized_image = ImageProcessor.histogram_equalization(gray_image)

        # Calculate histograms for all images
        color_hist = ImageProcessor.calculate_histogram(color_image)
        gray_hist = ImageProcessor.calculate_histogram(gray_image)
        equalized_hist = ImageProcessor.calculate_histogram(equalized_image)


        # Create a single output image for this test
        output_image = ImageProcessor.create_output_image(color_image, gray_image, equalized_image,
                                                          color_hist, gray_hist, equalized_hist, test_name)

        # Save the output image
        self.save_output_image(output_image, test_name)

    def draw_and_save_histogram(self, histogram, title):
        """Draw and save histogram."""
        plt.figure()
        plt.title(title)
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.plot(histogram)
        plt.xlim([0, 256])
        plt.savefig(os.path.join(self.result_dir, f'{title}.png'))

    def convert_heic_to_jpeg(self, heic_path):
        """Convert HEIC image to JPEG."""
        with Image.open(heic_path) as img:
            # Convert HEIC to JPEG
            jpeg_path = os.path.splitext(heic_path)[0] + '.jpg'
            img.convert('RGB').save(jpeg_path, 'JPEG')
        # Read the converted JPEG image using OpenCV
        jpeg_image = cv2.imread(jpeg_path)
        # Remove the temporary JPEG file
        os.remove(jpeg_path)
        return jpeg_image

    def save_output_image(self, output_image, test_name):
        """Save the output image."""
        output_path = os.path.join(self.result_dir, f'{test_name}_output_image.jpg')
        cv2.imwrite(output_path, output_image)
