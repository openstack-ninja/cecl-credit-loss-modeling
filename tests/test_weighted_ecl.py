import sys
import unittest
import os

# Import the code to test
# Ensure 'src' is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ecl_engine import compute_scenario_weighted_ecl

class TestWeightedECL(unittest.TestCase):
    def test_standard_weighted_ecl(self):
        """Test happy path with valid weights summing to 1.0."""
        summaries = [
            {"scenario": "Baseline", "total_ecl": 100_000_000.0, "portfolio_ecl_rate": 0.10, "total_balance": 1_000_000_000.0},
            {"scenario": "Adverse", "total_ecl": 200_000_000.0, "portfolio_ecl_rate": 0.20, "total_balance": 1_000_000_000.0}
        ]
        weights = {"Baseline": 0.6, "Adverse": 0.4}

        result = compute_scenario_weighted_ecl(summaries, weights)

        # 0.6 * 100M + 0.4 * 200M = 140M
        self.assertEqual(result["weighted_ecl"], 140_000_000.0)
        self.assertEqual(result["weighted_ecl_rate"], 0.14)
        self.assertEqual(result["total_balance"], 1_000_000_000.0)

    def test_weights_not_summing_to_one_assertion(self):
        """Test that an AssertionError is raised when weights do not sum to 1.0."""
        summaries = [
            {"scenario": "Baseline", "total_ecl": 100.0, "portfolio_ecl_rate": 0.1, "total_balance": 1000.0}
        ]
        weights = {"Baseline": 0.5} # Sum = 0.5 != 1.0

        with self.assertRaises(AssertionError) as cm:
            compute_scenario_weighted_ecl(summaries, weights)

        self.assertIn("Weights must sum to 1.0", str(cm.exception))

    def test_missing_scenario_in_weights(self):
        """Test that scenarios not in weights dictionary are treated as 0 weight."""
        summaries = [
            {"scenario": "Baseline", "total_ecl": 100_000_000.0, "portfolio_ecl_rate": 0.10, "total_balance": 1_000_000_000.0},
            {"scenario": "Unweighted", "total_ecl": 500_000_000.0, "portfolio_ecl_rate": 0.50, "total_balance": 1_000_000_000.0}
        ]
        weights = {"Baseline": 1.0} # 'Unweighted' is missing

        result = compute_scenario_weighted_ecl(summaries, weights)

        self.assertEqual(result["weighted_ecl"], 100_000_000.0)
        self.assertEqual(result["weighted_ecl_rate"], 0.10)

    def test_zero_total_balance_handling(self):
        """Test division by zero safety when total balance is 0."""
        summaries = [
            {"scenario": "Baseline", "total_ecl": 10.0, "portfolio_ecl_rate": 0.01, "total_balance": 0.0}
        ]
        weights = {"Baseline": 1.0}

        result = compute_scenario_weighted_ecl(summaries, weights)

        self.assertEqual(result["weighted_ecl"], 10.0)
        self.assertEqual(result["weighted_ecl_rate"], 0.0)
        self.assertEqual(result["total_balance"], 0.0)

if __name__ == "__main__":
    unittest.main()
