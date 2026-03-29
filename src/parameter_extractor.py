import argparse
import json
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    'original_interest_rate',
    'original_upb',
    'original_loan_term',
    'original_ltv',
    'dti',
    'borrower_credit_score'
]

def extract_parameters(df, features=None):
    """
    Calculates the statistical mean vector and covariance matrix for a set of features.

    Args:
        df (pd.DataFrame): The input dataframe.
        features (list): List of feature names to extract.

    Returns:
        dict: A dictionary containing 'mean' and 'covariance'.
    """
    if features is None:
        features = DEFAULT_FEATURES

    # Check if features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    # Subset dataframe to features and drop missing values
    df_subset = df[features].dropna()

    if len(df_subset) == 0:
        raise ValueError("Dataframe is empty after dropping missing values.")

    logger.info(f"Calculating parameters using {len(df_subset)} rows.")

    # Calculate mean and covariance
    mean_vector = df_subset.mean().to_dict()
    cov_matrix = df_subset.cov().to_dict()

    return {
        "mean": mean_vector,
        "covariance": cov_matrix
    }

def main():
    parser = argparse.ArgumentParser(description="Extract statistical parameters (Mean and Covariance) for Quantum Circuit.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/loan_level_combined.parquet",
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/quantum_parameters.json",
        help="Path to the output JSON file."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path, columns=DEFAULT_FEATURES)

    parameters = extract_parameters(df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(parameters, f, indent=4)

    logger.info(f"Successfully saved parameters to {output_path}")

if __name__ == "__main__":
    main()
