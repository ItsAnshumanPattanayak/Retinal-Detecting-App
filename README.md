<div align="center">

# MediAgent-Retina

### Experimental retinal fundus image classification interface built with PyTorch and Streamlit

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Interface-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv\&logoColor=white)](https://opencv.org/)

A Streamlit application for uploading retinal fundus images, applying image preprocessing, running an optional three-model classification ensemble, inspecting model probabilities, and displaying rule-based eye-health information.

> **Important:** This is an educational and experimental project. It is not a medical device and must not be used to diagnose, treat, or rule out any disease.

</div>

---

## Project Overview

MediAgent-Retina provides a browser-based interface for experimenting with retinal fundus image classification.

The application defines a three-model PyTorch ensemble consisting of:

* EfficientNet-B3
* ResNet-50
* VGG-16

When a compatible trained checkpoint is available at the configured path, the application attempts to load its weights and use the ensemble for inference.

The interface also includes deterministic, knowledge-base-driven modules that generate:

* a Markdown analysis report;
* precaution information;
* disease details;
* demographic information;
* keyword-based chatbot responses.

These modules do not use an external Large Language Model and do not independently reason, browse medical literature, or validate the model prediction.

---

## Current Implementation Status

| Capability                                | Status                             |
| ----------------------------------------- | ---------------------------------- |
| Streamlit web interface                   | Implemented                        |
| Fundus image upload                       | Implemented                        |
| JPG, JPEG, PNG and BMP input              | Implemented                        |
| RGB, grayscale and RGBA conversion        | Implemented                        |
| Basic image-quality assessment            | Implemented                        |
| Fundus-region cropping                    | Implemented                        |
| Image resizing and enhancement            | Implemented                        |
| EfficientNet-B3 architecture              | Implemented                        |
| ResNet-50 architecture                    | Implemented                        |
| VGG-16 architecture                       | Implemented                        |
| Weighted three-model ensemble             | Implemented                        |
| Loading a local model checkpoint          | Implemented in code                |
| Trained checkpoint included in repository | Not included                       |
| Confidence and class probabilities        | Implemented                        |
| Individual model predictions              | Implemented                        |
| Model-consensus indicator                 | Implemented                        |
| Markdown report generation                | Implemented                        |
| Disease and precaution information        | Implemented                        |
| Demographic information                   | Implemented                        |
| Rule-based chatbot                        | Implemented                        |
| Downloadable Markdown report              | Implemented                        |
| Model training notebook                   | Included                           |
| Reproducible dependency file              | Not included                       |
| Automated tests                           | Not included                       |
| User authentication                       | Not implemented                    |
| Patient database or persistent history    | Not implemented                    |
| External LLM integration                  | Not implemented                    |
| Live medical-literature retrieval         | Not implemented                    |
| Retinal lesion segmentation               | Not implemented in the application |
| Clinical validation                       | Not demonstrated                   |
| Hosted deployment                         | Not included                       |

---

## Critical Model-Availability Notice

The application expects a checkpoint at:

```text
./checkpoints/ensemble_best.pth
```

That checkpoint is not included in the current repository.

If the file cannot be found or loaded, the sidebar displays:

```text
Running in demo mode
```

In demo mode, the application still initializes EfficientNet-B3, ResNet-50, and VGG-16, but their weights are not loaded from a trained project checkpoint.

Consequently:

> **Predictions produced in demo mode are not meaningful disease classifications and must not be interpreted as medical results.**

For usable inference, you must provide a compatible trained checkpoint generated for the same model architecture and eight-class output configuration.

---

## Supported Classes

The application is configured with eight output categories:

1. Normal
2. Diabetic Retinopathy
3. ARMD
4. Media Haze
5. Optic Disc Disease
6. Retinal Vascular Disease
7. Myopia
8. Other Diseases

These are broad software labels used by the current classifier configuration. Some labels group multiple possible medical conditions and should not be interpreted as precise clinical diagnoses.

---

## Implemented Features

### 1. Fundus Image Upload

The Streamlit interface accepts:

* `.jpg`
* `.jpeg`
* `.png`
* `.bmp`

Uploaded images are displayed in the interface before analysis.

The application also converts:

* grayscale images to RGB;
* RGBA images to RGB.

### 2. Image-Quality Assessment

Before classification, the application calculates a simple image-quality score.

The current checks include:

* minimum image resolution;
* mean brightness;
* excessive darkness;
* excessive brightness;
* blur estimation using Laplacian variance.

The result includes:

* quality score from 0 to 100;
* resolution;
* brightness value;
* sharpness value;
* detected quality issues;
* pass or fail status.

This is a heuristic software check and is not a clinical image-gradability assessment.

### 3. Fundus Image Preprocessing

The preprocessing pipeline performs:

1. RGB-to-grayscale conversion for foreground detection.
2. Thresholding to locate non-black image content.
3. Largest-contour detection.
4. Optional cropping around the detected fundus region.
5. Resizing to `224 × 224`.
6. Gaussian-blur-based image enhancement.
7. Conversion to LAB colour space.
8. CLAHE enhancement on the lightness channel.
9. Conversion back to RGB.
10. ImageNet-style normalization.
11. Conversion to a PyTorch tensor.

The preprocessing code is intended to improve contrast and reduce unnecessary dark borders around a fundus photograph.

### 4. Three-Model Ensemble

The application defines three image classifiers using models supplied through `timm`.

#### EfficientNet-B3

The EfficientNet classifier uses:

* an EfficientNet-B3 feature extractor;
* layer normalization;
* dropout;
* a 512-unit fully connected layer;
* GELU activation;
* an eight-class output layer.

#### ResNet-50

The ResNet classifier uses:

* a ResNet-50 feature extractor;
* batch normalization;
* dropout;
* a 512-unit fully connected layer;
* ReLU activation;
* an eight-class output layer.

#### VGG-16

The VGG classifier uses:

* a VGG-16 feature extractor;
* layer normalization;
* dropout;
* a 512-unit fully connected layer;
* ReLU activation;
* an eight-class output layer.

### 5. Weighted Ensemble Prediction

Each model produces its own softmax probability distribution.

The ensemble combines those distributions using trainable weights initialized as:

```text
EfficientNet-B3: 0.45
ResNet-50:       0.35
VGG-16:         0.20
```

The weights are passed through a softmax operation before being applied.

The interface displays:

* the ensemble's selected class;
* ensemble confidence;
* severity and urgency text from the local knowledge base;
* each individual model's prediction;
* each individual model's confidence;
* whether all three models predicted the same class;
* a horizontal probability chart for all eight classes.

### 6. Report Generator

The report module creates a Markdown-formatted analysis summary containing:

* generated report ID;
* optional patient ID;
* date and time;
* selected class;
* confidence;
* severity and urgency labels;
* condition description;
* individual model predictions;
* model-consensus status;
* three highest class probabilities;
* image-quality information;
* predefined recommended actions;
* medical disclaimer.

The generated report can be downloaded as a `.md` file.

The report is generated locally from prediction values and predefined templates. It is not reviewed by a clinician.

### 7. Precaution Information

The precautions module retrieves predefined information for the selected class.

Depending on the category, it can display:

* immediate safety guidance;
* warning signs;
* follow-up suggestions;
* lifestyle precautions;
* monitoring recommendations;
* dietary suggestions.

This content comes from the hard-coded Python knowledge base.

### 8. Disease Information

The disease-details module presents predefined content such as:

* description;
* severity;
* urgency;
* symptoms;
* risk factors;
* treatment-related information;
* recommended precautions.

It does not retrieve current medical guidelines from external sources.

### 9. Demographic Information

The demographic module displays predefined information for each class, including:

* common age group;
* gender distribution;
* estimated prevalence text;
* geographic pattern;
* high-risk population information;
* general screening suggestions.

These values are static entries in the repository and should not be treated as a verified epidemiological reference.

### 10. Rule-Based MediBot

The application contains a local chatbot called MediBot.

It answers questions by detecting keywords associated with subjects such as:

* symptoms;
* treatment;
* prevention;
* precautions;
* diet;
* risk factors;
* disease description;
* visiting a doctor;
* emergencies;
* greetings.

Responses are selected from the current disease's predefined knowledge-base entry.

The chatbot is:

* deterministic;
* keyword based;
* locally implemented;
* not an LLM;
* not connected to a medical database;
* not capable of open-ended clinical reasoning.

### 11. Session-Based Chat History

Chat messages are kept in Streamlit session state during the current application session.

Users can:

* submit a question;
* choose predefined quick questions;
* view previous messages from the current session;
* clear the current chat.

Chat history is not stored in a database and is not preserved as a permanent patient record.

---

## Application Workflow

```text
Fundus image uploaded
          │
          ▼
Image converted to RGB
          │
          ▼
Heuristic quality assessment
          │
          ▼
Fundus-region detection and cropping
          │
          ▼
Resize, contrast enhancement and normalization
          │
          ▼
┌──────────────────────────────────────┐
│ Optional trained checkpoint loaded? │
└──────────────┬───────────────────────┘
               │
       ┌───────┴────────┐
       │                │
      Yes               No
       │                │
       ▼                ▼
 Trained model      Demo mode with
 inference          untrained weights
       │                │
       └───────┬────────┘
               ▼
 EfficientNet + ResNet + VGG predictions
               │
               ▼
 Weighted probability ensemble
               │
               ▼
 Class, confidence, probabilities and consensus
               │
               ▼
 Rule-based report and educational information
```

---

## Repository Structure

```text
Retinal-Detecting-App/
├── agents.py       # Static knowledge base and five rule-based modules
├── app.py          # Streamlit application, preprocessing and model definitions
├── main.ipynb      # Development and model-training notebook
└── README.md       # Project documentation
```

The current repository does not include:

```text
requirements.txt
checkpoints/
tests/
Dockerfile
.env.example
LICENSE
```

---

## Technology Used

### Application

* Python
* Streamlit
* PyTorch
* timm
* OpenCV
* NumPy
* Pillow
* Albumentations
* Matplotlib

### Model Architectures

* EfficientNet-B3
* ResNet-50
* VGG-16

### Development

* Jupyter Notebook

---

## Installation

### Prerequisites

Install:

* Python 3.9 or newer;
* Git;
* `pip`.

A virtual environment is recommended.

### 1. Clone the Repository

```bash
git clone https://github.com/ItsAnshumanPattanayak/Retinal-Detecting-App.git
cd Retinal-Detecting-App
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv retina-env
.\retina-env\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bat
python -m venv retina-env
retina-env\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv retina-env
source retina-env/bin/activate
```

### 3. Install the Required Packages

The repository currently has no `requirements.txt`. Install the packages imported by the application manually:

```bash
pip install streamlit torch torchvision timm albumentations opencv-python pillow numpy matplotlib
```

`app.py` contains logic that attempts to install some missing libraries at runtime. Installing all dependencies explicitly before launching the application is safer and more reproducible.

---

## Adding a Trained Checkpoint

Create the expected directory:

```bash
mkdir checkpoints
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force checkpoints
```

Place a compatible checkpoint at:

```text
checkpoints/ensemble_best.pth
```

The loading code expects the checkpoint object to contain:

```python
checkpoint["model_state_dict"]
```

The state dictionary must match:

* EfficientNet-B3 branch;
* ResNet-50 branch;
* VGG-16 branch;
* ensemble weights;
* custom classification heads;
* eight output classes.

A checkpoint created for a different architecture or class ordering will not load correctly.

---

## Running the Application

Start Streamlit from the repository root:

```bash
streamlit run app.py
```

Streamlit will normally open:

```text
http://localhost:8501
```

The sidebar reports whether the checkpoint loaded successfully.

### Trained Mode

Displayed when the expected checkpoint is found and loaded:

```text
Model loaded successfully
```

### Demo Mode

Displayed when the checkpoint is absent or cannot be loaded:

```text
Running in demo mode
```

Do not interpret demo-mode predictions as valid classifications.

---

## Using the Interface

1. Start the Streamlit application.
2. Check the sidebar for the model-loading status.
3. Optionally enter a patient ID.
4. Upload a supported retinal fundus image.
5. Review the heuristic quality score.
6. Select **Analyze Image**.
7. Review:

   * predicted software class;
   * confidence;
   * individual model outputs;
   * model agreement;
   * class-probability chart.
8. Open the information tabs.
9. Download the generated Markdown report when needed.

Do not enter directly identifying patient information when testing the application unless you have appropriate consent and data-handling safeguards.

---

## Configuration

Important values are currently defined directly in `app.py`:

```python
class CFG:
    NUM_CLASSES = 8
    IMAGE_SIZE = 224
    CHECKPOINT_PATH = "./checkpoints/ensemble_best.pth"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
```

To use another checkpoint location or image size, update the configuration and ensure that it still matches the trained model.

---

## What the “Agents” Actually Are

The project uses the term “agent” for five Python classes:

| Module                | Current behaviour                                             |
| --------------------- | ------------------------------------------------------------- |
| `ReportAgent`         | Builds a Markdown report from templates and prediction values |
| `PrecautionsAgent`    | Formats predefined precaution and safety content              |
| `DiseaseDetailsAgent` | Formats predefined disease information                        |
| `DemographicsAgent`   | Formats predefined demographic information                    |
| `ChatbotAgent`        | Selects responses using keyword matching                      |

These are useful application modules, but they are not autonomous AI agents in the current implementation.

They do not:

* call an LLM;
* plan multi-step tasks;
* use tools;
* search the web;
* retrieve journal articles;
* verify medical claims;
* update the knowledge base automatically;
* independently review the classifier output.

---

## Known Limitations

### Model and Reproducibility

* The trained checkpoint expected by the application is not included.
* Demo mode uses model architectures without loaded project-trained weights.
* A `requirements.txt` file is not included.
* Dataset preparation is not packaged as a reproducible pipeline.
* Training commands are not documented as standalone scripts.
* No versioned model metadata is included.
* No model checksum is included.
* The code silently ignores checkpoint-loading exceptions.

### Evaluation

* No verified test-set metrics are documented in the repository README.
* No confusion matrix is supplied as a repository artifact.
* No sensitivity, specificity, precision, recall, F1 score or AUC is documented for the deployed checkpoint.
* No independent external validation is demonstrated.
* No subgroup or fairness evaluation is demonstrated.
* No image-source generalization study is included.
* No automated test suite is included.

### Medical Scope

* The output classes are broad categories.
* The system does not identify retinal lesions.
* The application does not perform segmentation.
* The application does not grade disease stages.
* Image-quality checking is heuristic rather than clinically validated.
* Static health information may be incomplete, generalized or outdated.
* The application cannot consider symptoms, medical history, laboratory results, OCT images, visual acuity or an ophthalmologist's examination.
* A high software confidence value does not establish a medical diagnosis.
* A “Normal” result cannot rule out retinal or other eye disease.

### Data Handling

* There is no authentication.
* There is no encrypted patient-record system.
* There is no persistent database.
* There is no audit trail.
* There is no role-based access control.
* There is no documented regulatory or privacy compliance.
* The optional patient ID is inserted into a downloadable report but is not managed as a secure clinical identifier.

---

## Medical and Safety Disclaimer

This repository is intended for:

* education;
* software demonstrations;
* machine-learning experimentation;
* interface prototyping;
* non-clinical research.

It is **not** intended for:

* clinical diagnosis;
* screening without professional oversight;
* emergency decision-making;
* treatment recommendations;
* replacing an ophthalmologist;
* processing real patient data without appropriate governance;
* use as a certified medical device.

Always consult a qualified ophthalmologist or other licensed healthcare professional for medical advice.

Seek urgent medical attention for symptoms such as:

* sudden vision loss;
* new flashes or numerous floaters;
* a curtain-like shadow;
* severe eye pain;
* sudden distortion;
* rapid worsening of vision.

---

## Recommended Next Development Steps

The following items are proposed improvements and are not currently implemented:

* add a pinned `requirements.txt`;
* include a safe model-download procedure;
* document the training dataset and licence;
* save class mappings with the checkpoint;
* fail closed when no trained checkpoint is present;
* disable the Analyze button in demo mode;
* expose checkpoint-loading errors instead of silently ignoring them;
* add unit tests for preprocessing and agents;
* add integration tests for Streamlit workflows;
* publish reproducible evaluation metrics;
* add Grad-CAM or another interpretability method;
* separate medical content from application logic;
* attach sources and review dates to health information;
* add model cards and dataset cards;
* add configuration through environment variables;
* add linting and continuous integration;
* add a licence;
* add Docker support;
* obtain clinical review before any healthcare deployment.

---

## Suggested Safety Fix

The current application still allows analysis when the trained checkpoint is missing.

A safer behaviour would be:

```python
if not model_loaded:
    st.error(
        "No trained checkpoint is available. "
        "Image classification has been disabled."
    )
    st.stop()
```

This prevents users from mistaking untrained demo outputs for retinal disease predictions.

---

## Contributing

Contributions are welcome, particularly for:

* reproducible installation;
* model-validation documentation;
* automated testing;
* safer checkpoint handling;
* code organization;
* accessibility;
* medically reviewed documentation.

Suggested contribution workflow:

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Add: concise description"
git push origin feature/your-feature
```

Then open a pull request describing:

* what changed;
* why it was needed;
* how it was tested;
* whether model behaviour changed;
* any medical or safety implications.

Do not present experimental functionality as clinically validated.

---

## Author

**Anshuman Pattanayak**

GitHub: [@ItsAnshumanPattanayak](https://github.com/ItsAnshumanPattanayak)

Repository: [Retinal-Detecting-App](https://github.com/ItsAnshumanPattanayak/Retinal-Detecting-App)

---

<div align="center">

Built for educational experimentation with retinal-image classification and Streamlit.

**Not for clinical use.**

</div>
