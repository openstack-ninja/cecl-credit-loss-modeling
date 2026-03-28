import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pd_model import apply_woe_transformation

class TestPDModel(unittest.TestCase):
    def test_apply_woe_transformation_fallback(self):
        """Test the fallback path when bin edge extraction fails (e.g., KeyError)."""
        # Create a dummy dataframe with a continuous feature (nunique > 10)
        df = pd.DataFrame({
            "continuous_feat": range(20)
        })

        # Create woe_results where woe_table is missing the 'bin' column
        # This will trigger a KeyError in the try block
        woe_results = {
            "continuous_feat": {
                "woe_map": {"5": 0.5, "15": 1.5},
                "woe_table": pd.DataFrame({"not_bin": [1, 2]}) # Missing 'bin' column
            }
        }

        selected_features = ["continuous_feat"]

        # Apply transformation
        result = apply_woe_transformation(df, woe_results, selected_features)

        # It should fall back to string-based mapping
        # Value 5 string -> 0.5, Value 15 string -> 1.5, rest -> 0.0
        self.assertEqual(result.loc[5, "continuous_feat_woe"], 0.5)
        self.assertEqual(result.loc[15, "continuous_feat_woe"], 1.5)
        self.assertEqual(result.loc[0, "continuous_feat_woe"], 0.0)
        self.assertEqual(result.loc[19, "continuous_feat_woe"], 0.0)

if __name__ == "__main__":
    unittest.main()
