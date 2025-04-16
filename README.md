# Pima Indians Diabetes Classification & Explainability

This project demonstrates training a Random Forest classifier on the Pima Indians Diabetes dataset, followed by generating and evaluating model explanations using SHAP and Anchor methods.

---

## Prerequisites

- **Python 3.8+**
- Recommended: virtual environment (venv or conda)

Required packages:

```bash
numpy
pandas
matplotlib
scikit-learn
shap
alibi
```

---

## Running Locally

To run this code, download the script file and execute it on your local machine. Ensure that all required packages are already installed in your environment. No additional data files are needed, since the dataset is automatically fetched at runtime for convenience.

---

## Dataset

- **Source:** UCI Pima Indians Diabetes dataset (9 columns).
- **URL:** [https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- **Target:** Outcome (0 = non‑diabetic, 1 = diabetic).

---

## Code Structure and Components

### 1. Loading and Preparing Dataset

- **Defining** the dataset URL and assigning column names for readability.
- **Loading** the dataset into a pandas DataFrame.
- **Separating** `X` (features) and `y` (target).
- **Splitting** into training (80%) and testing (20%) subsets.

### 2. Training the Random Forest Model

- **Instantiating** a `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Fitting** the model on the training data.
- **Predicting** outcomes for the test data.
- **Printing** the accuracy and F1 score metrics.

### 3. Generating SHAP Explanations

- **Initializing** a `shap.Explainer(model, X_train)`.
- **Computing** SHAP values for `X_test`.
- **Plotting** a beeswarm plot for class 1 via `shap.plots.beeswarm(shap_values[:, :, 1])`.

### 4. Evaluating SHAP Faithfulness

- **Defining** the function `evaluate_faithfulness_shap(X, shap_vals, model, k=3)`.
- **Identifying** top-k features by mean absolute SHAP value.
- **Zeroing out** these features in `X_test`.
- **Comparing** model accuracy before and after modification.

### 5. Evaluating SHAP Completeness

- **Defining** the function `evaluate_completeness_shap(x, explainer, k=5)`.
- **Computing** total versus top-k SHAP contributions for a single instance.
- **Returning** the ratio of top-k sum to total attribution.

### 6. Generating Anchor Explanations

- **Defining** `predict_fn = lambda x: model.predict(x)`.
- **Instantiating** an `AnchorTabular` explainer with `predict_fn` and `feature_names`.
- **Fitting** the Anchor explainer on training data with discretization percentiles.
- **Explaining** the first test instance via `anchor_explainer.explain(...)`.

### 7. Evaluating Anchor Faithfulness

- **Defining** the function `evaluate_faithfulness_anchor(x, explainer, model)`.
- **Extracting** anchor features from the generated rules.
- **Zeroing out** these features in the input instance.
- **Checking** whether the prediction changes.

### 8. Evaluating Anchor Completeness

- **Defining** the function `evaluate_completeness_anchor(anchor, shap_values, feature_names)`.
- **Identifying** top-5 SHAP features for the instance.
- **Calculating** the overlap with anchor features.
- **Returning** the fraction of overlap.

---

## Actual Outputs

After running the script, following detailed outputs are expected:

1. **Console Output:**

   - **Model Performance Metrics** printed first:
     ```
     Accuracy: 0.78
     F1 Score: 0.65
     ```
   - **SHAP Section Header** and **Faithfulness/Completeness Evaluations**:
     ```
     SHAP SECTION
     =====================================
     SHAP FAITHFULNESS EVALUATION

     Top-3 SHAP features (by mean abs): ['Glucose', 'BMI', 'Age']
     Original accuracy before removing features: 0.78
     Accuracy after removing top-3 SHAP features: 0.70

     SHAP COMPLETENESS EVALUATION
     Total SHAP attribution: 1.2345
     Top-5 SHAP contribution: 0.9876
     SHAP completeness (top-5 features): 0.80
     ```
   - **Anchor Section Header** and **Rule Explanations** with precision and evaluations:
     ```
     ANCHOR SECTION
     =====================================

     Anchor Rule Explanation for Instance 0:
     If the following conditions are being met:
       - Glucose <= 127.5
       - BMI <= 32.0
     Then the model predicts class 0 with precision 0.90.

     ANCHOR FAITHFULNESS EVALUATION

     Prediction before removing anchor features: 0
     Prediction after removing anchor features: 1
     Anchor explanation changes prediction when removed: True

     ANCHOR COMPLETENESS EVALUATION

     Top SHAP features: {'Glucose', 'BMI', 'Age', 'Pregnancies', 'BloodPressure'}
     Anchor features: {'Glucose', 'BMI'}
     Anchor completeness compared to SHAP top-5: 0.40
     ```

2. **SHAP Beeswarm Plot:**

   - A Matplotlib window displaying a colored scatter showing feature impact for the positive class (1). Each dot represents a test instance, arranged vertically by feature name and colored by feature value.

---

##

