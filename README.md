# IRIS – Intelligent Recognition and Inspection System

**IRIS** is a computer vision project designed to explore object detection, image processing, recognition, and threat-analysis workflows using Python.

The project contains modules for detection, classification, image preprocessing, video-frame handling, geofencing, zone selection, and threat analysis.

## Features

- **Object Detection:** YOLO-based detection workflow.
- **Army Uniform Classification:** Classification and training scripts for uniform recognition.
- **Face Recognition:** Face-recognition processing module.
- **Geofencing:** Geographical boundary and zone-related logic.
- **Threat Analysis:** Threat-engine module for processing detection-related information.
- **Image Processing:** Image preprocessing and night-vision processing.
- **Video Processing:** Frame-source handling and video-processing pipeline.
- **Zone Selection:** Zone configuration and selection functionality.
- **Annotation and Validation:** Tools for image annotation, label validation, and testing.
- **Benchmarking:** Scripts for evaluating model performance.

## Technology Stack

- Python
- OpenCV
- YOLO-based object detection
- Computer Vision
- Machine Learning
- Image Processing

*Additional libraries and model dependencies should be documented in `requirements.txt`.*

## Project Structure

```text
IRIS/
│
├── annotate.py
├── army_uniform_classifier.py
├── backend.py
├── benchmark.py
├── detector.py
├── face_recognizer.py
├── frame_source.py
├── geofence.py
├── night_vision.py
├── pipeline.py
├── preprocessing.py
├── train_army_uniform.py
├── validate_labels.py
├── yolo_detector.py
├── zone_selector.py
│
├── edge/
│   └── threat_engine/
│       ├── threat_engine.py
│       └── test_threat_engine.py
│
├── test_8class_model.py
├── test_army_classifier.py
├── test_detector.py
├── test_face_recognition.py
├── test_geofence.py
├── test_helicopter.py
├── test_ingestion.py
├── test_preprocessing.py
└── test_yolo_detector.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sohanasaha01/I.R.I.S..git
cd I.R.I.S.
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by the individual modules. OpenCV is one confirmed dependency:

```bash
pip install opencv-python
```

## Running a YOLO Detection Test

The YOLO detector test accepts an image path as a command-line argument.

```bash
python test_yolo_detector.py path/to/your/image.jpg
```

Make sure that:
- The required Python dependencies are installed.
- The YOLO model weights are available locally.
- The image path is valid.
- The model configuration matches the detector implementation.

## Model Weights and Datasets

Large model weights, datasets, generated predictions, videos, and private data are intentionally excluded from this Git repository.

Obtain the required model files from their authorized sources and place them in the locations expected by the relevant scripts.

Do not upload private images, biometric data, credentials, or restricted datasets.

## Testing

The repository includes test scripts covering several components:

```bash
python test_detector.py
python test_geofence.py
python test_preprocessing.py
python -m pytest
```

Some tests may require local model weights, datasets, or configuration files.

## Responsible Use

Computer vision and recognition systems can produce inaccurate results. Outputs should be independently verified by qualified human reviewers.

Use the project only with appropriate authorization, respect privacy and applicable laws, and avoid treating automated detections or classifications as definitive evidence.

## Future Improvements

- Add a documented dependency file.
- Standardize configuration and model-weight paths.
- Improve automated testing.
- Add reproducible evaluation metrics.
- Document individual modules and their interfaces.
- Improve deployment and setup instructions.

## Author

**Sohana Saha**

GitHub: [@sohanasaha01](https://github.com/sohanasaha01)

---

*IRIS – Computer Vision and Intelligent Detection Project*
