import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model(df: pd.DataFrame):
    """
    Trains a model to predict enrollment rate.

    Args:
        df: The preprocessed data.

    Returns:
        A trained model.
    """
    # For now, we'll use a simple feature engineering approach
    # We'll use the 'phases' and 'country' columns as features
    features = ['phases', 'country']
    target = 'enrollment'

    # Drop rows with missing target
    df = df.dropna(subset=[target])
    
    # Separate features and target
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
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

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    return model

if __name__ == '__main__':
    # Example usage:
    preprocessed_df = pd.read_csv("data/preprocessed_studies.csv")
    # For now, we will fill missing country and phases with a placeholder
    preprocessed_df['country'] = preprocessed_df['country'].fillna('Unknown')
    preprocessed_df['phases'] = preprocessed_df['phases'].fillna('Unknown')
    
    trained_model = train_model(preprocessed_df)
    print("Model training complete.")
