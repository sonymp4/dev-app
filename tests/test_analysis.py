import unittest

import pandas as pd

from project1.analysis import filter_data, prepare_data


class PrepareDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = pd.DataFrame(
            {
                "mpg": [20, 30],
                "cylinders": [4, 6],
                "horsepower": ["100", "?"],
                "weight": [2000, 3000],
                "model_year": [70, 82],
                "origin": ["USA", "Japan"],
                "name": ["car one", "car two"],
            }
        )

    def test_adds_calendar_year_and_converted_measurements(self) -> None:
        result = prepare_data(self.raw)

        self.assertEqual(result["year"].tolist(), [1970, 1982])
        self.assertEqual(result["origin"].tolist(), ["usa", "japan"])
        self.assertAlmostEqual(result.loc[0, "consumption_l_100km"], 11.76072915)
        self.assertAlmostEqual(result.loc[0, "weight_kg"], 907.18474)
        self.assertTrue(pd.isna(result.loc[1, "horsepower"]))

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Colonnes requises manquantes"):
            prepare_data(self.raw.drop(columns=["origin"]))

    def test_filters_by_origin_year_and_cylinders(self) -> None:
        prepared = prepare_data(self.raw)

        result = filter_data(prepared, "japan", (1980, 1982), [6])

        self.assertEqual(result["name"].tolist(), ["car two"])

    def test_filters_by_minimum_mpg_threshold(self) -> None:
        prepared = prepare_data(self.raw)

        result = filter_data(prepared, None, (1970, 1982), [4, 6], minimum_mpg=25)

        self.assertEqual(result["name"].tolist(), ["car two"])
