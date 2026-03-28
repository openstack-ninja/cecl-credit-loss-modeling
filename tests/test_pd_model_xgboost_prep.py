import sys
import unittest
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import the code to test
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pd_model import prepare_features_for_xgboost

class TestPrepareFeaturesForXGBoost(unittest.TestCase):
    def test_missing_columns_skipped(self):
        df = pd.DataFrame({'a': [1, 2]})
        features = ['b'] # b is not in df columns

        result, encoders = prepare_features_for_xgboost(df, features)

        # Result should be an empty DataFrame with the same index
        self.assertEqual(list(result.columns), [])
        self.assertEqual(encoders, {})
        self.assertTrue((result.index == df.index).all())

    def test_numerical_imputation(self):
        # Numeric values with a missing value (None/NaN)
        df = pd.DataFrame({'num_col': [10.0, np.nan, 20.0, 30.0]})
        features = ['num_col']

        result, encoders = prepare_features_for_xgboost(df, features)

        self.assertIn('num_col', result.columns)
        self.assertEqual(list(result['num_col']), [10.0, 20.0, 20.0, 30.0])
        self.assertEqual(encoders, {})

    def test_categorical_label_encoding(self):
        # Categorical values with a missing value
        df = pd.DataFrame({'cat_col': pd.Series(['A', None, 'B'], dtype='object')})
        features = ['cat_col']

        result, encoders = prepare_features_for_xgboost(df, features)

        self.assertIn('cat_col', result.columns)
        self.assertIn('cat_col', encoders)

        self.assertEqual(list(encoders['cat_col'].classes_), ['A', 'B', 'MISSING'])
        self.assertEqual(list(result['cat_col']), [0, 2, 1])

    def test_combined_features(self):
        # A dataframe with categorical, numerical, and a missing feature
        df = pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0],
            'cat1': pd.Series(['X', 'Y', 'Y', np.nan], dtype='category'),
            'ignored': [1, 2, 3, 4]
        })
        features = ['num1', 'cat1', 'missing_col']

        result, encoders = prepare_features_for_xgboost(df, features)

        self.assertEqual(set(result.columns), {'num1', 'cat1'})
        self.assertEqual(list(result['num1']), [1.0, 2.0, 2.0, 4.0])
        self.assertEqual(list(encoders['cat1'].classes_), ['MISSING', 'X', 'Y'])
        self.assertEqual(list(result['cat1']), [1, 2, 2, 0])

if __name__ == "__main__":
    unittest.main()
