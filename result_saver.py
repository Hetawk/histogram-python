import os
import shutil

class ResultSaver:
    @staticmethod
    def save_results_to_folder(source_dir, destination_dir, case_name):
        # Create a directory for results if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Copy results to the destination directory
        shutil.copytree(source_dir, os.path.join(destination_dir, case_name))
