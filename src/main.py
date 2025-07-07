import pandas as pd
from model import train_model
import joblib

def main():
    """
    Main function to run the application.
    """
    # Load the preprocessed data
    preprocessed_df = pd.read_csv("data/preprocessed_studies.csv")
    
    # For now, we will fill missing country and phases with a placeholder
    preprocessed_df['country'] = preprocessed_df['country'].fillna('Unknown')
    preprocessed_df['phases'] = preprocessed_df['phases'].fillna('Unknown')

    # Train the model
    model = train_model(preprocessed_df)

    # Save the trained model
    joblib.dump(model, "data/enrollment_model.joblib")
    print("Model saved to data/enrollment_model.joblib")

    # Example of how to load the model and make a prediction
    loaded_model = joblib.load("data/enrollment_model.joblib")
    
    # Create a sample new trial for prediction
    new_trial = pd.DataFrame({
        'phases': ['Phase 2'],
        'country': ['United States']
    })

    prediction = loaded_model.predict(new_trial)
    print(f"Predicted enrollment for the new trial: {prediction[0]}")

if __name__ == '__main__':
    main()