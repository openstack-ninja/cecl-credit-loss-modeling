import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import io
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engine import fetch_fred_macro_data

class TestFeatureEngine(unittest.TestCase):
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_fetch_fred_macro_data_error_handling(self, mock_stdout):
        # We need to mock 'fredapi.Fred' where it is used. It's imported inside the function
        # from fredapi import Fred
        with patch('fredapi.Fred') as MockFred:
            # Configure the mock Fred instance
            mock_fred_instance = MagicMock()
            MockFred.return_value = mock_fred_instance

            # We need a side effect for get_series to return a Series normally, but raise an Exception for CPIAUCSL
            def get_series_side_effect(series_id, *args, **kwargs):
                if series_id == "CPIAUCSL":
                    raise Exception("Simulated API failure")
                else:
                    return pd.Series([1.0, 2.0], index=pd.to_datetime(['2020-01-01', '2020-02-01']))

            mock_fred_instance.get_series.side_effect = get_series_side_effect

            # Execute the function
            result_df = fetch_fred_macro_data(fred_api_key="fake_key", start_date="2020-01-01", end_date="2020-02-01")

            # Get the printed output
            output = mock_stdout.getvalue()

            # Verify the warning message was printed for CPIAUCSL
            self.assertIn("WARNING: Failed to fetch CPIAUCSL: Simulated API failure", output)

            # Verify that despite the error, the DataFrame was created and returned
            # (It should contain columns for the series that didn't fail, like UNRATE)
            self.assertIn("unemployment_rate", result_df.columns)
            self.assertNotIn("cpi_index", result_df.columns) # cpi_index is from CPIAUCSL, which failed

if __name__ == "__main__":
    unittest.main()
