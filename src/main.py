import os
from data_loader import fetch_clinical_trials

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Fetch the data
search_term = "cancer"
print(f"Fetching clinical trials for: {search_term}")
df = fetch_clinical_trials(search_term, max_studies=1000)

# Save the data
output_path = "data/clinical_trials.csv"
df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
