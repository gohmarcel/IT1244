# IT1244

This is a machine learning pipeline developed for predicting bank telemarketing success.

## Description

This project implements a predictive modeling pipeline to determine the likelihood of a customer subscribing to a term deposit. It utilizes Decision Trees and XGBoost, incorporating data balancing via SMOTE and automated hyperparameter optimization.

### Project Structure

LR_model.ipynb: Focuses on a linear approach using Logistic Regression, statistical significance testing (p-values), and multicollinearity checks (VIF).

DT_model.ipynb: Focuses on tree-based architectures, including Decision Trees and XGBoost, with advanced handling of imbalanced data via SMOTE.

## Getting Started

Before running the code, ensure you have a Python environment (e.g., Anaconda or a virtual environment) with the following libraries installed:

### Dependencies

* Data Manipulation: numpy, pandas 

* Visualization: matplotlib 

* Machine Learning: scikit-learn, xgboost 

* Imbalanced Data: imbalanced-learn 

### Installing

* Installation of dependencies
```
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn
```
* Obtaining dataset
Obtain the Bank Marketing Dataset (specifically the dataset.csv file) from this [link](https://drive.google.com/drive/folders/1SE5Ma9aC1OA_pue8e4qHTGDpYLtSEkNs)

### Executing program

* Ensure the dataset file is named exactly dataset.csv (or as specified in the "Reading in Data" section of the notebook)
* Place the CSV file in the same directory as the model.ipynb notebook.
* Launch Jupyter: Open your terminal or Anaconda Navigator and launch Jupyter Notebook or JupyterLab.
* Open the Notebook: Navigate to and open either DT_model.ipynb or LR_model.ipynb.
* Check Data Path: Locate the "Reading in Data" cell (Cell 3) and verify that the file path matches your local file location.
* In the top menu, click Cell > Run All (or Kernel > Restart & Run All).

### Methods and Key Features

* Statistical Rigor (Logistic Regression)
    * Multicollinearity Detection: Utilized the Variance Inflation Factor (VIF) test to identify and remove socioeconomic variables with high correlation, ensuring model stability.
    * Feature Pruning: Variables were filtered based on p-value significance, retaining only features with a statistically significant impact on the outcome.
    * Hyperparameter Tuning: Applied GridSearchCV to optimize the regularization strength ($C$) and penalty types (l1, l2, or elastic net).
* Advanced Pipeline (Decision Tree & XGBoost)
    * Data Balancing (SMOTE): Implemented the Synthetic Minority Over-sampling Technique (SMOTE) to address the heavy class imbalance in the "yes" vs. "no" responses.
    * Leakage Prevention: Used ImbPipeline to ensure that over-sampling was only applied to training folds during cross-validation, maintaining the integrity of the test sets.
    * Ensemble Boosting: Deployed XGBoost to correct residual errors from the baseline decision tree, significantly improving predictive power for complex customer profiles.

### Reviewing Results

* Classification Reports: The models output Precision, Recall, and F1-Scores. Recall was prioritized to ensure fewer potential subscribers were missed.

* Confusion Matrices: Visual representations of True Positives vs. False Positives are provided for each model.

* Feature Importance:

    * In the XGBoost model, the importance of the 'education' variable significantly increased, jumping to the 3rd most important feature compared to the baseline.

    * Economic indicators like emp.var.rate and nr.employed also remained high-impact predictors.

### Help

* ModuleNotFoundError: Ensure imblearn is installed via pip install imbalanced-learn. In the code, it is imported as from imblearn....

* FileNotFoundError: Double-check that the CSV file name in the pd.read_csv() call matches your file exactly.

* Long Execution Time: The GridSearchCV cells for XGBoost may take 1–3 minutes to execute depending on your hardware.