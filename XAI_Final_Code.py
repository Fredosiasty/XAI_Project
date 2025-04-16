import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from alibi.explainers import AnchorTabular
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------
# Loading and Preparing Dataset
# ---------------------------------------------

# Defining the URL for the dataset (Pima Indians Diabetes dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

# Defining column names for clarity and readability.
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Loading the dataset using pandas and assigning column names.
df = pd.read_csv(url, header=None, names=column_names)

# Separating features (X) and the target variable (y).
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting the data into training and test sets (80% for training, 20% for testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Training Random Forest Model
# ---------------------------------------------

# Instantiating a Random Forest classifier with 100 trees and a fixed random state for reproducibility.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model with the training dataset.
model.fit(X_train, y_train)

# Predicting the target values for the test set.
y_pred = model.predict(X_test)

# Printing performance metrics: Accuracy and F1 Score.
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}\n")

# ---------------------------------------------
# Generating SHAP Explanation and Visualization
# ---------------------------------------------

# Initializing a SHAP explainer for the Random Forest model using the training dataset.
explainer = shap.Explainer(model, X_train)

# Computing SHAP values for the test dataset.
shap_values = explainer(X_test)

# Generating a beeswarm plot showing the impact of features for the positive class (class 1).
# Using slicing "[:, :, 1]" for selecting the SHAP values corresponding to class 1.
shap.plots.beeswarm(shap_values[:, :, 1])

# ---------------------------------------------
# Evaluating Faithfulness (SHAP)
# ---------------------------------------------
print("SHAP SECTION")
print("=====================================")
print("SHAP FAITHFULNESS EVALUATION")

def evaluate_faithfulness_shap(X, shap_vals, model, k=3):
    """
    Evaluating the faithfulness of SHAP feature attributions by measuring
    the drop in model accuracy when the top-k features (by mean absolute SHAP value)
    are being removed from the dataset.

    Parameters:
        X (pd.DataFrame): The dataset on which to evaluate the model.
        shap_vals: SHAP values being computed for the dataset.
        model: The trained model being used for predictions.
        k (int): The number of top features (based on SHAP values) to be removed.

    Returns:
        float: The accuracy of the model after being removing the top-k features.
    """
    # Identifying the indices of the top-k features using mean absolute SHAP values for class 1.
    top_features = np.argsort(np.abs(shap_vals.values[:, :, 1]).mean(0))[::-1][:k]
    print(f"\nTop-{k} SHAP features (by mean abs): {[X.columns[i] for i in top_features]}")
    
    # Getting predictions on the original dataset to determining the baseline accuracy.
    original_preds = model.predict(X)
    original_acc = accuracy_score(y_test, original_preds)
    print(f"Original accuracy before removing features: {original_acc:.2f}")

    # Creating a modified copy of the dataset with the top-k features being zeroed out.
    X_modified = X.copy()
    X_modified.iloc[:, top_features] = 0

    # Getting predictions using the modified dataset.
    new_preds = model.predict(X_modified)
    # Returning the new accuracy.
    return accuracy_score(y_test, new_preds)

# Evaluating the change in accuracy after removing the top-3 SHAP features.
acc_drop = evaluate_faithfulness_shap(X_test, shap_values, model)
print(f"Accuracy after removing top-3 SHAP features: {acc_drop:.2f}\n")

# ---------------------------------------------
# Evaluating Completeness (SHAP)
# ---------------------------------------------
print("SHAP COMPLETENESS EVALUATION\n")

def evaluate_completeness_shap(x, explainer, k=5):
    """
    Evaluating the completeness of a SHAP explanation for a single instance by
    computing the ratio of the sum of the top-k absolute SHAP values to the total
    sum of absolute SHAP values.

    Parameters:
        x (pd.DataFrame or np.array): A single instance from the dataset.
        explainer: The SHAP explainer being used to compute SHAP values.
        k (int): The number of top features to be considered.

    Returns:
        float: The completeness ratio for the explanation.
    """
    # Computing absolute SHAP values for the input instance.
    shap_vals = np.abs(explainer(x).values[0])
    # Computing the total attribution.
    total = np.sum(shap_vals)
    # Computing the sum of the top-k contributions.
    top_k_sum = np.sum(np.sort(shap_vals)[-k:])
    
    print(f"Total SHAP attribution: {total:.4f}")
    print(f"Top-{k} SHAP contribution: {top_k_sum:.4f}")
    # Returning the ratio of the top-k sum to the total attribution.
    return top_k_sum / total if total != 0 else 0

# Evaluating the SHAP completeness ratio for the first test instance.
completeness_ratio = evaluate_completeness_shap(X_test.iloc[0:1], explainer, k=5)
print(f"SHAP completeness (top-5 features): {completeness_ratio:.2f}\n")

# ---------------------------------------------
# Generating Anchor Explanation (Single Instance)
# ---------------------------------------------
print("ANCHOR SECTION")
print("=====================================\n")

# Defining a prediction function that is accepting an array of inputs and returning model predictions.
predict_fn = lambda x: model.predict(x)
# Retrieving the list of feature names.
feature_names = X.columns.tolist()

# Initializing the AnchorTabular explainer with the prediction function and feature names.
anchor_explainer = AnchorTabular(predict_fn, feature_names)
# Fitting the Anchor explainer with the training data. The `disc_perc` parameter is setting the percentiles used for discretization.
anchor_explainer.fit(X_train.values, disc_perc=(25, 50, 75))

# Generating an anchor explanation for the first test instance.
anchor_exp = anchor_explainer.explain(X_test.values[0])
print("Anchor Rule Explanation for Instance 0:")
print("If the following conditions are being met:")
# Iterating through and displaying the conditions (anchor rules) from the explanation.
for clause in anchor_exp.data['anchor']:
    print(f"  - {clause}")
# Getting the model prediction for the same test instance.
predicted_class = model.predict(X_test.iloc[0:1])[0]
print(f"Then the model predicts class {predicted_class} with precision {anchor_exp.precision:.2f}.")

# ---------------------------------------------
# Evaluating Faithfulness (Anchors)
# ---------------------------------------------
print("\nANCHOR FAITHFULNESS EVALUATION\n")

def evaluate_faithfulness_anchor(x, explainer, model):
    """
    Evaluating the faithfulness of an Anchor explanation by testing if removing
    the features mentioned in the anchor is changing the model prediction.

    Parameters:
        x (np.array): A single instance of input features.
        explainer: The AnchorTabular explainer for generating explanations.
        model: The trained machine learning model.

    Returns:
        bool: True if the model prediction is changing upon removing anchor features, otherwise False.
    """
    # Generating an anchor explanation for the input instance.
    anchor = explainer.explain(x)
    # Extracting indices of the features mentioned in the anchor rules.
    anchor_features = [feature_names.index(name.split(' ')[0]) for name in anchor.data['anchor']]
    x_modified = x.copy()
    # Setting the selected anchor features to zero in the modified instance.
    x_modified[0, anchor_features] = 0

    # Getting predictions before and after modification.
    pred_original = model.predict(x)
    pred_modified = model.predict(x_modified)

    print(f"Prediction before removing anchor features: {pred_original[0]}")
    print(f"Prediction after removing anchor features: {pred_modified[0]}")
    # Returning True if the prediction differs, indicating that the anchor explanation is being faithful.
    return pred_original[0] != pred_modified[0]

# Evaluating if removing the anchor features is changing the prediction for the first test instance.
faithful_anchor = evaluate_faithfulness_anchor(X_test.values[0:1], anchor_explainer, model)
print(f"Anchor explanation changes prediction when removed: {faithful_anchor}")

# ---------------------------------------------
# Evaluating Completeness (Anchors)
# ---------------------------------------------
print("\nANCHOR COMPLETENESS EVALUATION\n")

def evaluate_completeness_anchor(anchor, shap_values, feature_names):
    """
    Evaluating the completeness of an anchor explanation by comparing the anchor features
    with the top features being identified by SHAP for a given instance. Completeness is being measured
    as the fraction of top SHAP features that are being present in the anchor explanation.

    Parameters:
        anchor: The anchor explanation object.
        shap_values (np.array): The SHAP values for a single instance.
        feature_names (list): List of feature names corresponding to the SHAP values.

    Returns:
        float: The fraction of top SHAP features that are overlapping with the anchor features.
    """
    # Identifying the indices of the top-5 features in the SHAP values.
    top_shap_idx = np.argsort(np.abs(shap_values[0]))[::-1][:5]
    # Converting the indices to the corresponding feature names.
    top_shap_features = set([feature_names[i] for i in top_shap_idx])
    # Extracting feature names from the anchor explanation.
    anchor_features = set([name.split(' ')[0] for name in anchor.data['anchor']])

    print(f"Top SHAP features: {top_shap_features}")
    print(f"Anchor features: {anchor_features}")

    # Calculating the overlap between the sets.
    overlap = anchor_features.intersection(top_shap_features)
    return len(overlap) / len(top_shap_features) if top_shap_features else 0

# Evaluating the completeness of the anchor explanation by comparing it with SHAP’s top-5 features.
anchor_completeness = evaluate_completeness_anchor(anchor_exp, shap_values.values[0], feature_names)
print(f"Anchor completeness compared to SHAP top-5: {anchor_completeness:.2f}")