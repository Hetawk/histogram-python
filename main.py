import os
from experiment_executor import ExperimentExecutor

def get_tests_and_images(directory, allowed_extensions):
    """Get a list of tests and their corresponding image files."""
    tests_and_images = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Subdirectory found, treat it as a test
            images = get_image_files(item_path, allowed_extensions)
            if images:
                tests_and_images.append((item, images))
        elif os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in allowed_extensions):
            # Image file found in the main directory, treat it as a test
            tests_and_images.append((os.path.splitext(item)[0], [item_path]))
    return tests_and_images

def get_image_files(directory, allowed_extensions):
    """Get a list of image files in the specified directory."""
    image_files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in allowed_extensions):
            image_files.append(os.path.join(directory, filename))
    return image_files

def run_tests(tests_and_images, result_dir):
    """Run experiments for each test."""
    for test_name, image_files in tests_and_images:
        for image_path in image_files:
            # Create the result directory and any necessary subdirectories if they don't exist
            test_result_dir = os.path.join(result_dir, test_name)
            os.makedirs(test_result_dir, exist_ok=True)

            executor = ExperimentExecutor(image_path, test_result_dir)
            executor.run_experiment(test_name)

def main():
    """Main function to execute the experiments."""
    image_dir = 'data'
    result_dir = 'all_results'  # Using a single result directory for all tests
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic']

    # Get tests and their corresponding image files
    tests_and_images = get_tests_and_images(image_dir, allowed_extensions)

    # Run tests
    run_tests(tests_and_images, result_dir)

if __name__ == "__main__":
    main()
