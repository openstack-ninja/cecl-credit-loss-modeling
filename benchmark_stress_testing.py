import time
import pandas as pd
import numpy as np

# Mock models and data
class MockModel:
    def predict_proba(self, X):
        return np.random.rand(len(X), 2)
    def predict(self, X):
        return np.random.rand(len(X))

def apply_woe_transformation_mock(df, woe_results, features):
    return df[features]

import sys
sys.modules['pd_model'] = type('MockPDModel', (), {'apply_woe_transformation': apply_woe_transformation_mock})

from src.stress_testing import score_portfolio_under_scenario

# Create a large dummy dataframe (e.g., 500,000 rows, 100 columns)
n_rows = 500000
n_cols = 100
portfolio_df = pd.DataFrame(np.random.rand(n_rows, n_cols), columns=[f'col_{i}' for i in range(n_cols)])

# Add required columns
portfolio_df['borrower_credit_score'] = np.random.randint(300, 850, n_rows)
portfolio_df['unemployment_rate'] = np.random.rand(n_rows)
portfolio_df['fico_x_unemployment'] = portfolio_df['borrower_credit_score'] * portfolio_df['unemployment_rate']

pd_features = ['col_1', 'col_2', 'col_3']
lgd_features = ['col_4', 'col_5', 'col_6']
macro_overrides = {'unemployment_rate': 0.05, 'col_1': 0.1}

pd_model = MockModel()
lgd_model = MockModel()
woe_results = {}

print("Benchmarking score_portfolio_under_scenario...")
t0 = time.time()
for _ in range(10): # Simulate 10 quarters
    pd_preds, lgd_preds = score_portfolio_under_scenario(
        portfolio_df,
        pd_model,
        woe_results,
        pd_features,
        lgd_model,
        lgd_features,
        macro_overrides,
    )
t1 = time.time()
print(f"Elapsed time: {t1 - t0:.4f} seconds")
