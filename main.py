import pandas as pd
from datetime import datetime
from etl_module import extract_data, preprocess_data
from ml_module import train_classification_model, evaluate_classification_model, train_regression_model, evaluate_regression_model

def main():
    file_path = 'kpopidols.csv'
    
    # ETL Process
    data = extract_data(file_path)
    preprocessed_data = preprocess_data(data)
    
    # ML Classification Process
    classification_model = train_classification_model(preprocessed_data)
    classification_accuracy = evaluate_classification_model(classification_model, preprocessed_data)
    print("Classification Accuracy:", classification_accuracy)
    
    # ML Regression Process
    regression_model = train_regression_model(preprocessed_data)
    regression_mse = evaluate_regression_model(regression_model, preprocessed_data)
    print("Regression Mean Squared Error:", regression_mse)

if __name__ == "__main__":
    main()
