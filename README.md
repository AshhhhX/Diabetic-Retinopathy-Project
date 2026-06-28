# Diabetic Retinopathy Severity Grading using CNN and ResNet50

## Project Overview

This project aims to classify diabetic retinopathy severity levels from retinal fundus images using deep learning techniques.

## Dataset

Source:
https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset

Classes:

* Healthy
* Mild DR
* Moderate DR
* Severe DR
* Proliferate DR

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn

## Methodology

1. Data preprocessing
2. Image resizing and normalization
3. CNN model implementation
4. ResNet50 transfer learning
5. Model evaluation

## Results

* Best Validation Accuracy: 72.91%
* Transfer Learning Model: ResNet50

## Repository Structure

* diabetic-retinopathy.ipynb
* Proposal.docx
* outputs/
* models/

## How to Run

### Option 1: Run on Kaggle

1. Open the Kaggle notebook:
   https://www.kaggle.com/code/ashroxx/diabetic-retinopathy-ipynb

2. Add the Diabetic Retinopathy dataset from:
   https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset

3. Run all notebook cells sequentially.

4. The notebook will:

   * Load the dataset
   * Preprocess the images
   * Apply data augmentation
   * Train the CNN model
   * Train the ResNet50 transfer learning model
   * Generate accuracy and loss curves
   * Generate the confusion matrix
   * Generate the classification report

### Option 2: Run Locally

1. Clone the repository:

   git clone https://github.com/AshhhhX/Diabetic-Retinopathy-Severity-Grading.git

2. Navigate to the project directory:

   cd Diabetic-Retinopathy-Severity-Grading

3. Install required packages:

   pip install tensorflow numpy matplotlib seaborn pandas scikit-learn

4. Download the dataset from Kaggle and place it in the Dataset folder.

5. Open the Jupyter Notebook:

   jupyter notebook diabetic-retinopathy.ipynb

6. Run all cells sequentially.

### Output

The project generates:

* Trained CNN model
* Trained ResNet50 model
* Accuracy and Loss Curves
* Confusion Matrix
* Classification Report
* Performance Comparison Results


## Author

Ashik Kirmani
