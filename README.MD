## Histogram For Image Processing

### Objective
- Understand the concept of image histograms and master the method of drawing them.
- Learn the principle of histogram equalization and apply it to image processing.

### Content
Convert a color image to a grayscale image, draw histograms (gray and equalized), and analyze the effect of histogram equalization.

### Requirements
1. **Image Selection**: Choose a suitable color image and convert it to grayscale.
2. **Histogram Drawing**: Implement histogram drawing and histogram equalization using any programming language.
3. **Completeness**: Provide the complete project process, code, results, and analysis.

---

### Project Structure
- **main.py**: Main script to execute experiments.
- **experiment_executor.py**: Class to run experiments and process images.
- **image_processor.py**: Class containing image processing functions.
- **result_saver.py**: Class for saving experiment results.
- **data/**: Directory containing input images.
- **all_results/**: Directory to store experiment results.

### Getting Started
1. Clone the repository: `git clone https://github.com/Hetawk/histogram-python.git`
2. Navigate to the project directory: `cd histogram-python`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the experiments: `python main.py`

### Experiment Results
- Each experiment result will be saved in the `all_results` directory.
- Results include output images and histograms.

### Analysis
- Analyze the effect of histogram equalization on image contrast and brightness.
- Compare the original image, grayscale image, and equalized image.

---

### Contributors
- [Enoch Kwateh Dongbo](https://github.com/Hetawk/histogram-python.git)

### Sample Results
#### Natural Scenery Image
![Natural Scenery](all_results/natural_scenery/natural_scenery_output_image.jpg)

#### Portrait Image
![Portrait](all_results/portrait/portrait_output_image.jpg)

#### Another Portrait Image
![Fake](all_results/fake/fake_output_image.jpg)