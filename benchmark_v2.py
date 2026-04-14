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

# Modified version
def score_portfolio_under_scenario_optimized(
    portfolio_df,
    pd_model,
    woe_results,
    pd_features,
    lgd_model,
    lgd_features,
    macro_overrides,
):
    from pd_model import apply_woe_transformation

    required_cols = set(pd_features) | set(lgd_features) | set(macro_overrides.keys())
    if "fico_x_unemployment" in portfolio_df.columns:
        required_cols.add("fico_x_unemployment")
        required_cols.add("borrower_credit_score")

    cols_to_copy = list(required_cols.intersection(portfolio_df.columns))
    df_scenario = portfolio_df[cols_to_copy].copy()

    for feat, val in macro_overrides.items():
        if feat in df_scenario.columns:
            df_scenario[feat] = val

    if "fico_x_unemployment" in df_scenario.columns and "unemployment_rate" in macro_overrides:
        df_scenario["fico_x_unemployment"] = (
            df_scenario["borrower_credit_score"] * macro_overrides["unemployment_rate"]
        )

    X_woe = apply_woe_transformation(df_scenario, woe_results, pd_features)
    pd_preds = pd_model.predict_proba(X_woe)[:, 1]

    X_lgd = df_scenario[lgd_features].copy()
    lgd_fill = {
        "loan_age_at_default": 48.0,
        "was_modified": 0.0,
    }
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())

    lgd_preds = lgd_model.predict(X_lgd)
    lgd_preds = np.clip(lgd_preds, 0.0, 1.0)

    return pd_preds, lgd_preds


n_rows = 500000
n_cols = 100
portfolio_df = pd.DataFrame(np.random.rand(n_rows, n_cols), columns=[f'col_{i}' for i in range(n_cols)])
portfolio_df['borrower_credit_score'] = np.random.randint(300, 850, n_rows)
portfolio_df['unemployment_rate'] = np.random.rand(n_rows)
portfolio_df['fico_x_unemployment'] = portfolio_df['borrower_credit_score'] * portfolio_df['unemployment_rate']

pd_features = ['col_1', 'col_2', 'col_3']
lgd_features = ['col_4', 'col_5', 'col_6']
macro_overrides = {'unemployment_rate': 0.05, 'col_1': 0.1}

pd_model = MockModel()
lgd_model = MockModel()
woe_results = {}

print("Benchmarking optimized score_portfolio_under_scenario...")
t0 = time.time()
for _ in range(10): # Simulate 10 quarters
    pd_preds, lgd_preds = score_portfolio_under_scenario_optimized(
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
