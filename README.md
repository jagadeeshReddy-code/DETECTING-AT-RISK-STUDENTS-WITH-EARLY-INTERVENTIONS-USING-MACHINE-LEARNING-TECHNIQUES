# DETECTING-AT-RISK-STUDENTS-WITH-EARLY-INTERVENTIONS-USING-MACHINE-LEARNING-TECHNIQUES

## Overview

This project is a Tkinter-based graphical user interface (GUI) application that utilizes various machine learning algorithms to predict at-risk students based on their academic data. The application allows users to upload a dataset, preprocess it, apply dimensionality reduction, and then run several machine learning algorithms to evaluate their performance. The results are displayed through metrics such as accuracy, sensitivity, specificity, and F1-score, as well as visualized using plots.

## Features

- **Upload Dataset:** Load a dataset from a CSV file for analysis.
- **Preprocessing:** Handle missing values, encode categorical features, and apply PCA for dimensionality reduction.
- **Model Training and Evaluation:** Run and evaluate multiple machine learning models, including:
  - Random Forest Classifier
  - Generalized Linear Model (Logistic Regression)
  - Gradient Boosting Machine
  - Multi-Layer Perceptron (MLP)
  - Feed Forward Neural Network (FFNN)
- **Visualization:** View plots of feature correlations and model accuracy comparisons.
- **Prediction:** Make predictions on new datasets to identify at-risk students.

## Installation

To run this application, ensure you have Python and the necessary libraries installed. You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## Usage

1. **Run the Application:**
   ```bash
   python your_script_name.py
   ```

2. **Upload a Dataset:**
   - Click the "Upload Student Dataset" button to load your CSV file.

3. **Preprocess Data:**
   - Click "Preprocess & PCA Feature Selection" to handle missing values, encode categorical variables, and apply PCA.

4. **Run Algorithms:**
   - Select one of the following buttons to run the corresponding algorithm:
     - "Run Random Forest Algorithm"
     - "Run Generalized Linear Model Algorithm"
     - "Run Gradient Boosting Machine Algorithm"
     - "Run MLP Algorithm"
     - "Run Feed Forward Neural Network"
   
5. **View Accuracy Graph:**
   - Click "Accuracy Graph" to visualize the performance of the algorithms.

6. **Make Predictions:**
   - Click "Risk Prediction" to predict at-risk students based on a new dataset.

## Example Dataset

Ensure your dataset is in CSV format and includes columns for categorical features such as `code_module`, `code_presentation`, `assessment_type`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, and `final_result`.

## Code Structure

- `upload()`: Handles dataset loading and initial processing.
- `preprocess()`: Manages data preprocessing and PCA.
- `runRF()`, `runGLM()`, `runGBM()`, `runMLP()`, `runFeedForward()`: Execute and evaluate different machine learning models.
- `predict()`: Makes predictions on new datasets.
- `graph()`: Displays a bar graph comparing the accuracy of different models.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Make sure to include tests for any new features or changes.

## Contact

For any questions or feedback, please contact [jagadeesh reddy](mailto:jagadeeshreddy7876@gmail.com).

---

Feel free to modify the content to better match your project details, such as the script name or contact information.
