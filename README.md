# Well Log Facies Classification & Prediction

## Overview

This repository contains machine learning workflows for predicting missing facies in well logs using classification techniques. It is designed to address the challenge of incomplete facies data: given several wells with labeled facies and one well with missing facies, the models are trained on available data and used to predict the facies for the unlabeled well.

Facies are geological units distinguished by physical, chemical, and biological properties, which are crucial in reservoir characterization, hydrocarbon exploration, and geological modeling.

---

## Problem Statement

In subsurface reservoir studies, facies are often missing in some well logs due to limitations in data acquisition or manual labeling. Predicting these missing facies can improve geological interpretations and reduce human effort.

- **Data**: 5 well logs (CSV, LAS, or similar), 4 with facies labels, 1 without.
- **Objective**: Train a classification model to accurately predict facies for the 5th well using measured log features (e.g., GR, RHOB, NPHI, etc.).

---

## Directory Structure

```
facies-prediction-ml/
│
├── data/                   # Raw and processed well log data files
│   ├── well_log1.csv
│   ├── well_log2.csv
│   ├── well_log3.csv
│   ├── well_log4.csv
│   └── well_log5_missing_facies.csv
│
├── notebooks/              # Jupyter notebooks for EDA, modeling & prediction
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── predict_missing_facies.ipynb
│
├── src/                    # Source code modules
│   ├── preprocessing.py    # Data cleaning & transformation functions
│   ├── model.py            # ML model building & evaluation scripts
│   └── predict.py          # Facies prediction for missing labels
│
├── requirements.txt        # Python package dependencies
└── README.md               # Project overview & setup instructions
```

---

## Workflow

1. **Data Preparation**
    - Load all well logs.
    - Clean and preprocess features (handle missing values, normalization, encoding, etc.).
    - Split wells: Use the 4 labeled wells as training/testing data; reserve the unlabeled well for prediction.

2. **Exploratory Data Analysis (EDA)**
    - Visualize facies and log distributions.
    - Analyze well-to-well feature similarity.
    - Assess class imbalance.

3. **Feature Engineering**
    - Select relevant log features.
    - Optionally apply dimensionality reduction (PCA, t-SNE).
    - Engineer new features (ratios, rolling averages).

4. **Model Selection & Training**
    - Train classification models (e.g., Random Forest, XGBoost, SVM, Neural Network).
    - Hyperparameter tuning via cross-validation.
    - Evaluate on held-out wells (blind well test).

5. **Prediction**
    - Apply the trained model to the 5th well.
    - Post-process predictions (smoothing, filtering).
    - Visualize predicted facies distribution.

6. **Results & Interpretation**
    - Compare predicted facies with geologist expectations (if available).
    - Discuss model confidence, uncertainties, and possible improvements.

---

## Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/facies-prediction-ml.git
    cd facies-prediction-ml
    ```

2. **Prepare Data**
    - Place all well log files in the `data/` folder.
    - Ensure facies column exists in four logs.

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    Dependencies include: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter

4. **Run Notebooks**
    - Launch JupyterLab or Jupyter Notebook.
    ```bash
    jupyter lab
    ```
    - Explore the provided notebooks for step-by-step guidance on EDA, training, and facies prediction.

5. **Run Scripts**
    - Use code in `src/` for custom data processing or automation.
    - Example usage:
        ```python
        from src.model import train_model
        from src.predict import predict_facies
        # Training and prediction code here
        ```

---

## Example Facies Prediction Workflow

```python
from src.preprocessing import prepare_data
from src.model import train_model, evaluate_model
from src.predict import predict_facies

# Load and preprocess datasets
X_train, y_train, X_test = prepare_data()

# Train model
clf = train_model(X_train, y_train)

# Predict missing facies
y_pred = predict_facies(clf, X_test)
```

---

## Data Format

- **Inputs**: Well log files with columns like:
    - Depth, GR, NPHI, RHOB, DTC, Facies (optional)
- **Outputs**: Predicted facies labels for unlabeled well

Example CSV snippet:
```
DEPTH,GR,NPHI,RHOB,DTC,FACIES
1200,85,0.45,2.50,90,3
1200.5,88,0.47,2.47,93,3
...
```

---

## Contributing

Feel free to submit issues, feature requests, or pull requests!

1. Fork the repo & create branches for your changes.
2. Format code and comments clearly.
3. Describe your changes in PRs.

---

## License

[MIT](LICENSE) — Free for academic and personal use.

---

## References

- Dubois, S., et al. (2007). "Facies classification from well logs: A machine learning approach."
- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
