import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def calculate_histogram(image):
        """Calculate histogram for a grayscale image."""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Calculate histogram
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        return histogram

    @staticmethod
    def histogram_equalization(image):
        """Perform histogram equalization on a grayscale image."""
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Convert back to BGR format
        equalized_bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

        return equalized_bgr_image

    @staticmethod
    def convert_to_gray(image):
        """Convert color image to grayscale."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def resize_and_align_images(images):
        """Resize and align a list of images."""
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)

        aligned_images = []
        for image in images:
            height_diff = max_height - image.shape[0]
            width_diff = max_width - image.shape[1]
            top = height_diff // 2
            bottom = height_diff - top
            left = width_diff // 2
            right = width_diff - left
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            aligned_images.append(padded_image)

        return aligned_images

    @staticmethod
    def create_output_image(color_image, gray_image, equalized_image,
                            color_hist, gray_hist, equalized_hist, test_name):
        """Create a single output image for the test."""
        # Resize and align images
        aligned_images = ImageProcessor.resize_and_align_images([color_image, gray_image, equalized_image])

        # Combine images horizontally
        combined_image = np.hstack(aligned_images)

        # Calculate the dimensions of the output image
        height, width = combined_image.shape[:2]
        max_image_height = max(image.shape[0] for image in aligned_images)
        hist_height = max_image_height  # Set histogram height to match max image height
        output_height = height + hist_height

        # Create a blank white image to accommodate all the processed images horizontally
        output_image = np.ones((output_height, width, 3), dtype=np.uint8) * 255  # White background

        # Paste the combined image onto the output image
        output_image[:height, :] = combined_image

        # Draw histograms and place them below the images
        hist_width = width // 3
        ImageProcessor.draw_histogram(output_image, color_hist, hist_height, height, 0, hist_width)
        ImageProcessor.draw_histogram(output_image, gray_hist, hist_height, height, hist_width, hist_width * 2)
        ImageProcessor.draw_histogram(output_image, equalized_hist, hist_height, height, hist_width * 2, hist_width * 3)

        # Add labels for each image type
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Adjust the font scale as needed
        font_thickness = 4  # Adjust the font thickness as needed
        label_color = (0, 0, 255)  #  color for the labels
        label_offset = 50  # Offset from the image height for placing the labels
        cv2.putText(output_image, 'Color Image', (10, height + label_offset), font, font_scale, label_color,
                    font_thickness, cv2.LINE_AA)
        cv2.putText(output_image, 'Gray Image', (hist_width + 10, height + label_offset), font, font_scale, label_color,
                    font_thickness, cv2.LINE_AA)
        cv2.putText(output_image, 'Equalized Image', (2 * hist_width + 10, height + label_offset), font, font_scale,
                    label_color, font_thickness, cv2.LINE_AA)

        # Add the test name as text to the output image
        test_name_offset = 20  # Offset from the bottom of the output image for placing the test name
        cv2.putText(output_image, f'Test: {test_name}', (10, output_height - test_name_offset), font, font_scale,
                    label_color, font_thickness, cv2.LINE_AA)

        return output_image

    @staticmethod
    def draw_histogram(output_image, histogram, hist_height, offset, start_col, end_col):
        """Draw histogram."""
        hist_width = end_col - start_col

        # Normalize the histogram
        max_val = np.max(histogram)
        normalized_hist = (histogram / max_val) * hist_height

        # Calculate bin width
        bin_width = hist_width // len(histogram)

        # Draw the histogram
        for i in range(len(histogram)):
            # Draw a rectangle for each bin
            cv2.rectangle(output_image, (start_col + bin_width * i, hist_height - int(normalized_hist[i]) + offset),
                          (start_col + bin_width * (i + 1), hist_height + offset), color=(255, 0, 0), thickness=-1)

        # Add a line bar to separate histograms
        cv2.line(output_image, (end_col, 0), (end_col, hist_height + offset), color=(0, 255, 0), thickness=3)