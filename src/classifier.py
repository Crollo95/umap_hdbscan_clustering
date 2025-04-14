from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
import os

def prepare_data_for_classification(data, labels, test_size=0.2, random_state=42):
    """
    Prepare stratified train-test split excluding noise points.
    """
    mask = labels != -1
    data_clean = data[mask]
    labels_clean = labels[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        data_clean, labels_clean,
        test_size=test_size,
        stratify=labels_clean,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, n_estimators=400, random_state=42):
    """
    Train Random Forest classifier.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=10,
                                random_state=random_state,
                                class_weight='balanced')
    rf.fit(X_train, y_train)
    
    pred_train = rf.predict(X_train)
    print(confusion_matrix(y_train, pred_train))
    
    return rf

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate the classifier and return classification metrics including Balanced Accuracy and MCC.
    """
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)

    # Compute additional metrics
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    metrics_summary = {
        "balanced_accuracy": balanced_acc
    }

    # MCC computation (multiclass supported)
    try:
        mcc = matthews_corrcoef(y_test, predictions)
        metrics_summary["mcc"] = mcc
    except Exception as e:
        metrics_summary["mcc"] = None
        print("Could not compute MCC:", e)

    return report, conf_matrix, metrics_summary



import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_and_plot_shap(model, X_train, feature_names, output_dir, top_n=10):
    """
    Compute SHAP values and plot beeswarm plots correctly for multiclass classification.
    SHAP plots are saved in an inner folder named 'shap_plots'.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Define SHAP plot directory
    shap_plot_dir = os.path.join(output_dir, "shap_plots")
    if not os.path.exists(shap_plot_dir):
        os.makedirs(shap_plot_dir)

    # Check if shap_values are 3-dimensional (multiclass)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Multiclass scenario
        n_classes = shap_values.shape[2]
        for class_idx in range(n_classes):
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values[:, :, class_idx],
                features=X_train,
                feature_names=feature_names,
                max_display=top_n,
                show=False
            )
            plt.title(f"SHAP Beeswarm Plot (Class {class_idx})")
            plt.tight_layout()
            shap_plot_path = os.path.join(shap_plot_dir, f"shap_beeswarm_class_{class_idx}.png")
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()
    else:
        # Binary classification or regression scenario
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=X_train,
            feature_names=feature_names,
            max_display=top_n,
            show=False
        )
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()
        shap_plot_path = os.path.join(shap_plot_dir, "shap_beeswarm.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()