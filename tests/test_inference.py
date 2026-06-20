from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference import HomeCreditInferenceService, InferenceInputError


REQUIRED_ASSETS = [
    ROOT / "models" / "stage5_lightgbm_smote_gpu.joblib",
    ROOT / "models" / "stage5_lightgbm_raw_gpu.joblib",
    ROOT / "data" / "processed" / "home_credit_features.csv",
    ROOT / "data" / "raw" / "home_credit" / "application_train.csv",
]


@unittest.skipUnless(all(path.exists() for path in REQUIRED_ASSETS), "Local model/data artifacts are unavailable.")
class HomeCreditInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = HomeCreditInferenceService(ROOT)

    @staticmethod
    def read_raw_row(row_index: int) -> pd.DataFrame:
        path = ROOT / "data" / "raw" / "home_credit" / "application_train.csv"
        return pd.read_csv(path, skiprows=lambda line: line != 0 and line - 1 != row_index)

    def test_recommendations_match_stage6_contract(self) -> None:
        expected = {
            "fn5_fp1": ("stage5_lightgbm_smote_gpu", 0.18),
            "fn10_fp1": ("stage5_lightgbm_smote_gpu", 0.10),
            "fn20_fp1": ("stage5_lightgbm_raw_gpu", 0.05),
        }
        for scenario, (model, threshold) in expected.items():
            recommendation = self.service.recommendation(scenario)
            self.assertEqual(recommendation["model"], model)
            self.assertAlmostEqual(float(recommendation["selected_threshold"]), threshold, places=8)

    def test_raw_case_matches_saved_stage7_score(self) -> None:
        cases = self.service.demo_cases()
        case = cases.loc[cases["case_name"].eq("main_fn10_tp_high_risk")].iloc[0]
        raw = self.read_raw_row(int(case["row_index"]))
        scored = self.service.score_raw(raw, "fn10_fp1")
        self.assertAlmostEqual(float(scored.results.iloc[0]["risk_score"]), float(case["y_score"]), places=5)

    def test_target_column_does_not_change_score(self) -> None:
        raw = self.read_raw_row(294310)
        with_target = self.service.score_raw(raw, "fn10_fp1").results.iloc[0]["risk_score"]
        without_target = self.service.score_raw(raw.drop(columns=["TARGET"]), "fn10_fp1").results.iloc[0]["risk_score"]
        self.assertAlmostEqual(float(with_target), float(without_target), places=10)

    def test_unknown_category_is_handled_by_encoder(self) -> None:
        raw = self.read_raw_row(294310)
        raw.loc[:, "NAME_INCOME_TYPE"] = "UNSEEN_APP_CATEGORY"
        scored = self.service.score_raw(raw, "fn10_fp1")
        self.assertEqual(len(scored.results), 1)

    def test_missing_required_column_is_rejected(self) -> None:
        raw = self.read_raw_row(294310).drop(columns=["NAME_CONTRACT_TYPE"])
        with self.assertRaises(InferenceInputError):
            self.service.score_raw(raw, "fn10_fp1")


if __name__ == "__main__":
    unittest.main()
