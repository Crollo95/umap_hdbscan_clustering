import yaml
import argparse
import pandas as pd
import os
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.classifier import (
    prepare_data_for_classification,
    train_random_forest,
    evaluate_classifier
)

def main(config_path):
    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    data_df = load_data(config['data'])
    scaled_data = scale_data(data_df, config.get('scaling_method', None))
    
    # Load cluster assignments
    cluster_assignments_df = pd.read_csv(config['assignments_output'])
    labels = cluster_assignments_df['cluster_label'].values

    # Data preparation
    X_train, X_test, y_train, y_test = prepare_data_for_classification(scaled_data, labels)

    # Train classifier
    classifier = train_random_forest(X_train, y_train)

    # Evaluate classifier
    report, conf_matrix, additional_metrics = evaluate_classifier(classifier, X_test, y_test)

    # Output directory
    output_dir = os.path.dirname(config['classification_output'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(config['classification_output'], index=True)
    print(f"Classification report saved to: {config['classification_output']}")
    
    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.csv")
    conf_matrix_df.to_csv(conf_matrix_path, index=False)
    print(f"Confusion matrix saved to: {conf_matrix_path}")
    
    # Save additional metrics
    metrics_df = pd.DataFrame([additional_metrics])
    metrics_path = os.path.join(output_dir, "additional_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Additional metrics saved to: {metrics_path}")

    
    # After evaluating classifier, compute SHAP values
    from src.classifier import compute_and_plot_shap
    
    # Compute and save SHAP feature importance
    feature_names = data_df.columns.tolist()
    compute_and_plot_shap(classifier, X_train, feature_names, output_dir, top_n=10)
    print(f"SHAP feature importance plots saved in: {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Forest classification on clusters")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    main(args.config)
