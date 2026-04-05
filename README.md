# IT1244

This is a machine learning pipeline developed for predicting bank telemarketing success.

## Description

This project implements a predictive modeling pipeline to determine the likelihood of a customer subscribing to a term deposit. It utilizes Decision Trees and XGBoost, incorporating data balancing via SMOTE and automated hyperparameter optimization.

## Getting Started

Before running the code, ensure you have a Python environment (e.g., Anaconda or a virtual environment) with the following libraries installed:

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

* Data Manipulation: numpy, pandas 


* Visualization: matplotlib 


* Machine Learning: scikit-learn, xgboost 


* Imbalanced Data: imbalanced-learn 

### Installing

* Installation of dependencies
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn* Obtaining dataset
Obtain the Bank Marketing Dataset (specifically the dataset.csv file) from this [link](https://drive.google.com/drive/folders/1SE5Ma9aC1OA_pue8e4qHTGDpYLtSEkNs)

### Executing program

* Ensure the dataset file is named exactly dataset.csv (or as specified in the "Reading in Data" section of the notebook)
* Place the CSV file in the same directory as the model.ipynb notebook.
* Launch Jupyter: Open your terminal or Anaconda Navigator and launch Jupyter Notebook or JupyterLab.
* Open the Notebook: Navigate to and open model.ipynb.
* Check Data Path: Locate the "Reading in Data" cell (Cell 3) and verify that the file path matches your local file location.
* In the top menu, click Cell > Run All (or Kernel > Restart & Run All).

### What the model will do

* Preprocessing: Categorical variables are transformed using OneHotEncoder within a ColumnTransformer pipeline.


* Baseline Modeling: A standard Decision Tree is trained first to establish a performance benchmark.


* Data Balancing: SMOTE is applied within an ImbPipeline to handle class imbalance.


* Hyperparameter Tuning: GridSearchCV is executed to find the optimal parameters for the Decision Tree (e.g., max_depth, min_samples_split).


* Ensemble Modeling: The XGBClassifier is trained to improve upon the baseline results.

### Reviewing results

* The notebook will output Classification Reports (Precision, Recall, F1-Score) and Confusion Matrices for each model.

* A Feature Importance plot will be generated at the end to show which variables (e.g., education) most influenced the XGBoost model's decisions.

## Help

* ModuleNotFoundError: Ensure imblearn is installed via pip install imbalanced-learn. In the code, it is imported as from imblearn....

* FileNotFoundError: Double-check that the CSV file name in the pd.read_csv() call matches your file exactly.