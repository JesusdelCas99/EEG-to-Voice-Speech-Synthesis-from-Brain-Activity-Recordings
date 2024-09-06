# EEG-to-Voice: Speech Synthesis from Brain Activity Recordings

This repository contains the code developed as part of the master's thesis "EEG-to-Voice: Speech Synthesis from Brain Activity Recordings," submitted in fulfillment of the requirements for a Master's degree in Telecommunications Engineering from the Universidad de Granada, during the 2023/2024 academic year.

## Overview

This project aims to develop a speech synthesis system from brain activity recordings using EEG data, particularly for individuals suffering from speech impairments due to neurodegenerative diseases such as amyotrophic lateral sclerosis (ALS), brain injuries, or cerebral palsy. The ultimate goal is to restore the ability to communicate verbally for patients who have lost this function by using Brain-Computer Interfaces (BCIs).

This research leverages variational autoencoding (VAE) to perform speech synthesis from EEG signals recorded from epileptic patients implanted with deep electrodes during language production tasks. By learning the correlation between brain activity and speech, this project aims to provide a natural and effective communication solution for individuals with speech disabilities, ultimately helping those in locked-in syndrome and similar conditions.

## Repository Structure

The project consists of several key scripts and directories:

### Folders
- **`cancor/`**  
  Contains plotted files generated from canonical correlation analysis for each patient, saved as `acc_XXX.eps` files, where `XXX` denotes the patient ID.
  
- **`iBDS-dataset/`**  
  Stores i-BDS formatted data for each patient, including audio and iEEG recordings after segmentation:
  - `audio_data/`: Raw audio data from patients.
  - `ieeg_data/`: Raw iEEG recordings from patients.
  - `audio_features/`: Extracted audio features.
  - `ieeg_features/`: Extracted iEEG features.
  - `validation/`: Contains test files evaluated for each fold per patient.
  
- **`metrics/`**  
  Stores performance metrics for the models, calculated for each patient, validation fold, and test record.

- **`transience/`**  
  Contains auxiliary functions used throughout the project.

### Key Scripts
- **`compute_audio_metrics.py`**  
  Evaluates performance metrics for the test samples by patient and fold, generating a JSON file that records results for each metric. These results are stored in the `metrics/` folder.

- **`extract_save_features.py`**  
  Extracts audio and iEEG features from the segmented recordings and saves them in the appropriate subfolders within `iBDS-dataset/`.

- **`plot_evaluation_metrics.py`**  
  Plots the validation results from the data in the JSON file generated by `compute_audio_metrics.py`, enabling the visualization of model performance.

- **`s2a_feature_synthesis.py`**  
  Implements the design, training, validation, and testing of the speech synthesis models.

- **`cca_analysis.py`**  
  Performs Canonical Correlation Analysis (CCA) to assess the relationship between characteristics derived from audio and EEG data for each patient.
  
- **`setup.cfg`**  
  Configuration file containing hyperparameters for feature extraction, model design, and training.


## Setup and Installation

1. **Windows Installation**  
   Install the required dependencies:
   ```bash
   pip install -r requirements-windows.txt
   ```

2. **Linux Installation or WSL 2 (Recommended for Full Functionality)**  
   Some scripts require Linux-based environments for full compatibility, particularly `compute_audio_metrics.py`, due to unsupported packages on Windows. We recommend using **Windows Subsystem for Linux 2 (WSL 2)** if a Linux machine is unavailable. To install the required dependencies, run:
   ```bash
   pip install -r requirements-linux.txt
   ```

## Usage Instructions

1. **Feature Extraction**  
   Extract audio and iEEG features by running:
   ```bash
   python extract_save_features.py
   ```

2. **Training and Testing the Speech Synthesis Model**  
   Design, train, and test the models by executing:
   ```bash
   python s2a_feature_synthesis.py
   ```

3. **Evaluating Model Performance**  
   Compute performance metrics using:
   ```bash
   python compute_audio_metrics.py
   ```
   This script will generate a JSON file with the evaluation results.

4. **Plotting Evaluation Results**  
   Visualize the results by running:
   ```bash
   python plot_evaluation_metrics.py
   ```

## Contributions and Future Work

This project lays the foundation for non-verbal communication systems using brain activity recordings. In the future, the system could be enhanced by:
- Extending the dataset with more patients and diverse tasks.
- Improving the synthesis model architecture to produce more natural speech.
- Exploring alternative BCI modalities to enhance accuracy and usability.

## License

This project is licensed under the MIT License.
