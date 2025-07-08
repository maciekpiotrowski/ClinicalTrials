import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

def train_and_validate(df: pd.DataFrame):
    """
    Trains a model and validates it on a hold-out set.

    Args:
        df: The preprocessed data.

    Returns:
        A tuple containing the trained model, the validation dataframe with predictions, and the mae.
    """
    # Use 'phases' and 'country' as features
    features = ['phases', 'country']
    target = 'enrollment_rate'

    # Drop rows with missing target or features
    df = df.dropna(subset=[target] + features)
    
    # Separate features and target
    X = df[features]
    y = df[target]

    # Split data into training and validation sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing for categorical features
    categorical_features = ['phases', 'country']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on Validation Set: {mae}")

    # Create a validation dataframe with predictions
    validation_df = X_test.copy()
    validation_df['actual_enrollment_rate'] = y_test
    validation_df['predicted_enrollment_rate'] = y_pred

    return model, validation_df, mae

def plot_validation_results(validation_df: pd.DataFrame):
    """
    Plots the actual vs. predicted enrollment rates.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(validation_df['actual_enrollment_rate'], validation_df['predicted_enrollment_rate'], alpha=0.5)
    plt.plot([0, validation_df['actual_enrollment_rate'].max()], [0, validation_df['actual_enrollment_rate'].max()], 'r--')
    plt.xlabel("Actual Enrollment Rate (patients/month)")
    plt.ylabel("Predicted Enrollment Rate (patients/month)")
    plt.title("Actual vs. Predicted Enrollment Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/validation_plot.png")
    print("Validation plot saved to data/validation_plot.png")


if __name__ == '__main__':
    # Example usage:
    preprocessed_df = pd.read_csv("data/preprocessed_studies.csv")
    
    # For now, we will fill missing country and phases with a placeholder
    preprocessed_df['country'] = preprocessed_df['country'].fillna('Unknown')
    preprocessed_df['phases'] = preprocessed_df['phases'].fillna('Unknown')
    
    _, validation_results, _ = train_and_validate(preprocessed_df)
    print("\nRetrospective Validation Results:")
    print(validation_results.head())
    
    plot_validation_results(validation_results)