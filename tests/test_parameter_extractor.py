import os
import sys
import unittest
import pandas as pd
import numpy as np

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(project_root, 'src'))

from parameter_extractor import extract_parameters

class TestParameterExtractor(unittest.TestCase):
    def setUp(self):
        # Create a small, mocked DataFrame
        self.data = {
            'original_interest_rate': [6.0, 5.5, 6.5, np.nan],
            'original_upb': [100000, 200000, 150000, 300000],
            'original_loan_term': [360, 180, 360, 360],
            'original_ltv': [80, 60, 90, 80],
            'dti': [35, 20, 45, 40],
            'borrower_credit_score': [720, 800, 650, 680],
            'ignored_feature': ['A', 'B', 'C', 'D']
        }
        self.df = pd.DataFrame(self.data)

        # Calculate expected values mathematically for the non-null rows
        self.valid_rows = self.df.dropna(subset=[
            'original_interest_rate', 'original_upb', 'original_loan_term',
            'original_ltv', 'dti', 'borrower_credit_score'
        ])
        self.expected_mean = self.valid_rows[[
            'original_interest_rate', 'original_upb', 'original_loan_term',
            'original_ltv', 'dti', 'borrower_credit_score'
        ]].mean().to_dict()

        self.expected_cov = self.valid_rows[[
            'original_interest_rate', 'original_upb', 'original_loan_term',
            'original_ltv', 'dti', 'borrower_credit_score'
        ]].cov().to_dict()

    def test_extract_parameters_success(self):
        # Run extraction
        result = extract_parameters(self.df)

        # Verify mean keys and values
        self.assertEqual(set(result['mean'].keys()), set(self.expected_mean.keys()))
        for key in self.expected_mean:
            self.assertAlmostEqual(result['mean'][key], self.expected_mean[key])

        # Verify covariance keys and values
        self.assertEqual(set(result['covariance'].keys()), set(self.expected_cov.keys()))
        for key1 in self.expected_cov:
            self.assertEqual(set(result['covariance'][key1].keys()), set(self.expected_cov[key1].keys()))
            for key2 in self.expected_cov[key1]:
                self.assertAlmostEqual(result['covariance'][key1][key2], self.expected_cov[key1][key2])

    def test_extract_parameters_missing_columns(self):
        # Remove a required column
        df_missing = self.df.drop(columns=['original_interest_rate'])

        with self.assertRaises(ValueError) as context:
            extract_parameters(df_missing)

        self.assertTrue("Missing features in dataframe" in str(context.exception))

    def test_extract_parameters_empty_after_dropna(self):
        # Create a dataframe where every row has at least one missing value
        data_all_null = {
            'original_interest_rate': [np.nan, 5.5, 6.5],
            'original_upb': [100000, np.nan, 150000],
            'original_loan_term': [360, 180, np.nan],
            'original_ltv': [80, 60, 90],
            'dti': [35, 20, 45],
            'borrower_credit_score': [720, 800, 650]
        }
        df_empty = pd.DataFrame(data_all_null)

        with self.assertRaises(ValueError) as context:
            extract_parameters(df_empty)

        self.assertTrue("Dataframe is empty after dropping missing values." in str(context.exception))

if __name__ == '__main__':
    unittest.main()
