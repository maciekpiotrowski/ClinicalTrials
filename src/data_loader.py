import requests
import pandas as pd

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_studies(search_expr: str, max_studies: int = 100) -> pd.DataFrame:
    """
    Fetches a limited number of clinical trial studies from ClinicalTrials.gov.

    Args:
        search_expr: The search expression to use for finding studies.
        max_studies: The maximum number of studies to retrieve.

    Returns:
        A pandas DataFrame containing the study data.
    """
    params = {
        "query.intr": search_expr,
        "pageSize": max_studies,
        "format": "json"
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    
    studies = data.get('studies', [])
    
    return pd.json_normalize(studies)

if __name__ == '__main__':
    # Example usage:
    search_expression = "cancer"
    studies_df = fetch_studies(search_expression, max_studies=50)
    
    if not studies_df.empty:
        print(f"Successfully fetched {len(studies_df)} studies.")
        print(studies_df.head())
        # Save to a CSV file for inspection
        studies_df.to_csv("data/raw_studies.csv", index=False)
    else:
        print("No studies found for the given search expression.")