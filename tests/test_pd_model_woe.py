import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure 'src' is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pd_model import calculate_woe_iv_for_feature

class TestWoEIV(unittest.TestCase):
    def test_calculate_woe_iv_for_feature_value_error(self):
        """
        Test the error path in calculate_woe_iv_for_feature when pd.qcut raises ValueError.
        This occurs when a continuous feature has too few unique values (e.g., all identical).
        The function should catch the ValueError and fallback to treating it as categorical.
        """
        # Create a DataFrame with a continuous feature where all values are identical
        # 50 defaults (target=1), 50 non-defaults (target=0)
        data = {
            "identical_feature": [10.0] * 100,
            "target": [1] * 50 + [0] * 50
        }
        df = pd.DataFrame(data)

        # When all values are identical, pd.qcut natively throws ValueError: "Bin edges must be unique"
        # calculate_woe_iv_for_feature should catch this and process it as categorical (astype(str))
        result = calculate_woe_iv_for_feature(df, "identical_feature", "target", n_bins=10)

        # Assertions
        self.assertEqual(result["feature"], "identical_feature")
        self.assertIn("iv", result)
        self.assertIn("woe_table", result)

        # Since it was treated as categorical, there should only be 1 bin ("10.0")
        woe_table = result["woe_table"]
        self.assertEqual(len(woe_table), 1)
        self.assertEqual(woe_table.iloc[0]["bin"], "10.0")

        # Since the single bin contains 50 defaults and 50 non-defaults out of 100 total,
        # dist_events = 1.0, dist_non_events = 1.0
        # woe = ln(dist_non_events_smooth / dist_events_smooth) = ln(1/1) ~ 0
        self.assertAlmostEqual(woe_table.iloc[0]["woe"], 0.0, places=4)
        self.assertAlmostEqual(result["iv"], 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
