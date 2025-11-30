# TCD Tracing Python Implementation

This directory contains a Python implementation of the Transcranial Doppler (TCD) ultrasound envelope tracing algorithms, originally developed in MATLAB.

## Project Structure

*   **`main.py`**: The main entry point script. It loads data, runs the tracing algorithms (Adaptive and MTCM), and plots the results.
*   **`preprocessing.py`**: Functions for loading data (`load_tcd_data`), handling time vectors (`preprocess_time_vector`), and segmenting the signal (`segment_echo_signal`).
*   **`spectrogram_utils.py`**: Functions for generating spectrograms from IQ data (`generate_spectrogram`, `compute_spectrogram`).
*   **`tracing.py`**: The core envelope tracing algorithms:
    *   `spectrogram_tracing`: Implements the Adaptive Method (Otsu thresholding, etc.).
    *   `ultrasound_tracing_mtcm`: Implements the Maximum Threshold Crossing Method (MTCM).
*   **`sqa.py`**: Signal Quality Assessment (SQA) module. Includes the WABP beat detector and artifact detection logic.
*   **`postprocessing.py`**: Filters for smoothing the resulting velocity envelopes.
*   **`spectrogram_viewer.py`**: A standalone script to visualize the spectrogram of a TCD recording.

## Dependencies

Ensure you have the following Python packages installed:

```bash
pip install numpy pandas scipy matplotlib scikit-image
```

## Usage

### Running the Main Script

To process a recording and see the full envelope tracing results:

```bash
python main.py
```

By default, it looks for data in `../Healthy Subjects/Healthy_Subjects_Recording_1.txt`. You can edit the `filepath` and `filename` variables in `main.py` to point to your data.

### Running the Spectrogram Viewer

To simply visualize the spectrogram of a file:

```bash
python spectrogram_viewer.py
```

### Using in Jupyter Notebooks

You can import the modules directly into a Jupyter Notebook. See `example_implementation.ipynb` for a step-by-step guide.

```python
from preprocessing import load_tcd_data, preprocess_time_vector
from spectrogram_utils import generate_spectrogram
# ... and so on
```

## Data Format

The code expects a text file (CSV or Tab-separated) with at least three columns:
1.  `t`: Time vector (microseconds or seconds)
2.  `I`: In-phase signal component
3.  `Q`: Quadrature signal component

The `load_tcd_data` function in `preprocessing.py` is robust and handles various text formats.
