import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw clinical trial data.

    Args:
        df: The raw clinical trial data.

    Returns:
        The preprocessed data.
    """
    # Select relevant columns
    columns_to_keep = [
        'protocolSection.identificationModule.nctId',
        'protocolSection.statusModule.overallStatus',
        'protocolSection.statusModule.startDateStruct.date',
        'protocolSection.statusModule.completionDateStruct.date',
        'protocolSection.conditionsModule.conditions',
        'protocolSection.designModule.studyType',
        'protocolSection.designModule.phases',
        'protocolSection.designModule.enrollmentInfo.count',
        'protocolSection.eligibilityModule.eligibilityCriteria',
        'protocolSection.contactsLocationsModule.locations'
    ]
    
    # Check which of the desired columns are actually present
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    processed_df = df[existing_columns].copy()

    # Rename columns for easier access
    processed_df = processed_df.rename(columns={
        'protocolSection.identificationModule.nctId': 'nct_id',
        'protocolSection.statusModule.overallStatus': 'status',
        'protocolSection.statusModule.startDateStruct.date': 'start_date',
        'protocolSection.statusModule.completionDateStruct.date': 'completion_date',
        'protocolSection.conditionsModule.conditions': 'conditions',
        'protocolSection.designModule.studyType': 'study_type',
        'protocolSection.designModule.phases': 'phases',
        'protocolSection.designModule.enrollmentInfo.count': 'enrollment',
        'protocolSection.eligibilityModule.eligibilityCriteria': 'eligibility_criteria',
        'protocolSection.contactsLocationsModule.locations': 'locations'
    })

    # Handle missing values
    processed_df['enrollment'] = pd.to_numeric(processed_df['enrollment'], errors='coerce').fillna(0)
    
    # Convert date columns to datetime
    processed_df['start_date'] = pd.to_datetime(processed_df['start_date'], errors='coerce')
    processed_df['completion_date'] = pd.to_datetime(processed_df['completion_date'], errors='coerce')

    # Calculate duration in months
    processed_df['duration_months'] = (processed_df['completion_date'] - processed_df['start_date']).dt.days / 30.44
    
    # Calculate enrollment rate
    processed_df['enrollment_rate'] = processed_df['enrollment'] / processed_df['duration_months']
    processed_df['enrollment_rate'] = processed_df['enrollment_rate'].replace([pd.NA, float('inf'), -float('inf')], 0).fillna(0)


    # Extract country from locations
    if 'locations' in processed_df.columns:
        processed_df['country'] = processed_df['locations'].apply(lambda x: x[0]['country'] if isinstance(x, list) and len(x) > 0 and 'country' in x[0] else None)
        processed_df = processed_df.drop(columns=['locations'])

    return processed_df

if __name__ == '__main__':
    # Example usage:
    raw_df = pd.read_csv("data/raw_studies.csv")
    preprocessed_df = preprocess_data(raw_df)
    print("Preprocessing complete.")
    print(preprocessed_df.head())
    preprocessed_df.to_csv("data/preprocessed_studies.csv", index=False)
