# ATHENA Toolbox
ATHENA (Automatically Tracking Hands Expertly with No Annotations) is a Python-based toolbox designed to process multi-camera video recordings, extract 2D and 3D body and hand landmarks using MediaPipe, and perform triangulation and refinement of these landmarks. The toolbox provides a user-friendly GUI for selecting videos and configuring processing options.

## Features
- Multi-Camera Processing: Handles multiple camera inputs for comprehensive analysis.
- MediaPipe Integration: Utilizes Google’s MediaPipe for extracting 2D landmarks of the body and hands.
- 3D Triangulation: Triangulates 2D landmarks from multiple cameras to reconstruct 3D landmarks.
- Smoothing and Refinement: Applies smoothing algorithms to refine the 3D landmarks.
- Parallel Processing: Supports multiprocessing to speed up the processing of large datasets.
- GUI for Easy Configuration: Provides a graphical user interface to select folders, recordings, and set processing options.
- Visualization: Offers options to save images and videos of the processed landmarks for visualization.

## Installation
### Prerequisites
- Operating System: Windows, macOS, or Linux.
- Python Version: Python 3.9 or higher.
- Hardware Requirements:
- CPU: Multi-core processor recommended for parallel processing.
- GPU (Optional): NVIDIA GPU for accelerated processing (if GPU processing is enabled).

### Installation Steps

```console
git clone https://github.com/neural-control-and-computation-lab/athena.git
cd athena
pip install .
```

It is recommended to create a conda environment before installing to manage dependencies and avoid conflicts.

## Usage
### 1.	Organize Your Videos
Place your video recordings in a main folder, structured as follows:
```console
main_folder/
├── videos/
│   ├── recording1/
│   │   ├── cam0.avi
│   │   ├── cam1.avi
│   │   └── ...
│   ├── recording2/
│   │   ├── cam0.avi
│   │   ├── cam1.avi
│   │   └── ...
└── calibration/
    ├── cam0.yaml
    ├── cam1.yaml
    └── ...
```

- Each recording folder should contain video files from multiple cameras (e.g., cam0.avi, cam1.avi).
- The calibration folder should contain calibration files (.yaml) for each camera.

### 2.	Ensure Calibration Files are Correct
- Calibration files should be labelled with camera names that match recorded videos.
- Calibration files should contain intrinsic and extrinsic parameters for each camera.
- The calibration files are essential for accurate triangulation of 3D landmarks.

### 3. Running the Toolbox
Launch the GUI:
```console
athena
```

1. Select Main Folder and Recordings 
   - Click on the “Select Folder” button to choose your main folder containing the videos and calibration directories.
   - A new window will appear, allowing you to select specific recordings to process. Select the desired recordings and click “Select”.
2. Configure Processing Options
   - General Settings:
     - Fraction of Frames to Process: Slide to select the fraction of frames you want to process (from 0 to 1). Default is 1.0 (process all frames).
     - Number of Parallel Processes: Choose the number of processes for parallel processing. Default is the number of available CPU cores.
     - GPU Processing: Check this option to enable GPU acceleration (requires compatible NVIDIA GPU).
   - MediaPipe Processing:
     - Run Mediapipe: Check this option to run MediaPipe for extracting 2D landmarks.
     - Save Images: Save images with landmarks drawn (can consume significant storage).
     - Save Video: Save videos with landmarks overlaid (requires “Save Images” to be enabled).
     - Minimum Hand Detection & Tracking Confidence: Adjust the confidence threshold for hand detection (default is 0.9).
     - Minimum Pose Detection & Tracking Confidence: Adjust the confidence threshold for pose detection (default is 0.9).
   - Triangulation:
     - Triangulation: Check this option to perform 3D triangulation of landmarks.
     - Save Images: Save images of the 3D landmarks.
     - Save Video: Save videos of the 3D landmarks.
3. Start Processing
   - Click the “GO” button to start processing.
   - A progress window will appear, showing the processing progress and average FPS.

### 4. Output Folder Structure
After processing, the toolbox will create additional directories within your main folder:
```console
main_folder/
├── images/                # Contains images with landmarks drawn
├── imagesrefined/         # Contains refined images after triangulation
├── landmarks/             # Contains 2D and 3D landmark data
├── videos_processed/      # Contains processed videos with landmarks overlaid
└── ...                    # Original videos and calibration files
```

### Troubleshooting
- MediaPipe Errors: Ensure that MediaPipe is correctly installed and compatible with your Python version. MediaPipe may have specific requirements, especially for GPU support.
- Calibration Mismatch: Verify that the number of calibration files matches the number of camera videos.
- High Memory Usage: Processing large videos or saving images and videos can consume significant memory and storage.
- Permission Issues: Ensure that you have read/write permissions for the directories where data is stored and processed.
- Conda Environment Activation: Always make sure that the Conda environment is activated before running the scripts.

### Contributing
Contributions are welcome! If you encounter issues or have suggestions for improvements, please create an issue or submit a pull request on GitHub.

### Frequently Asked Questions (FAQ)

Q: Do I need a GPU to run the ATHENA Toolbox?\
A: No, a GPU is not required and does not generally speed up processing, which is already highly parallelized. If you have a compatible NVIDIA GPU, you can enable GPU processing in the options. Ensure that your GPU drivers and CUDA toolkit are correctly installed.

Q: Can I process videos from only one camera?\
A: Yes, you can process videos from a single camera to extract 2D landmarks. However, triangulation to 3D landmarks requires synchronized videos from at least two cameras and their corresponding calibration files.

Q: How do I obtain the calibration files?\
A: Calibration files are generated using camera calibration techniques, often involving capturing images of a known pattern (like a chessboard) from different angles. You can use OpenCV’s calibration tools or other software to create these .yaml files.

Q: The processing is slow. How can I speed it up?\
A: You can increase the number of parallel processes if your CPU has more cores. Enabling GPU processing can also significantly speed up the landmark detection step. Additionally, processing a smaller fraction of frames can reduce computation time.

Q: Where can I find the output data after processing?\
A: Processed data, images, and videos are saved in the images/, imagesrefined/, landmarks/, and videos_processed/ directories within your main folder.

Q: I encounter an error related to av or PyAV.\
A: Ensure that av is installed via Conda using the conda install av -c conda-forge command, as it may have dependencies that are better handled by Conda.