from __future__ import annotations

import pandas as pd

MPG_TO_LITRES_PER_100_KM = 235.214583
POUNDS_TO_KILOGRAMS = 0.45359237


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public Auto MPG dataset and add display-ready metrics."""
    data = raw.copy()
    data.columns = data.columns.str.strip().str.lower()

    required_columns = {
        "mpg",
        "cylinders",
        "horsepower",
        "weight",
        "model_year",
        "origin",
        "name",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes requises manquantes dans le CSV : {missing}")

    for column in ("mpg", "cylinders", "horsepower", "weight", "model_year"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["mpg", "cylinders", "weight", "model_year", "origin"])
    data = data.loc[data["mpg"] > 0].copy()
    data["year"] = data["model_year"].astype(int) + 1900
    data["origin"] = data["origin"].astype(str).str.strip().str.lower()
    data["consumption_l_100km"] = MPG_TO_LITRES_PER_100_KM / data["mpg"]
    data["weight_kg"] = data["weight"] * POUNDS_TO_KILOGRAMS
    return data.reset_index(drop=True)


def filter_data(
    data: pd.DataFrame,
    origin: str | None,
    years: tuple[int, int],
    cylinders: list[int],
    minimum_mpg: float | None = None,
) -> pd.DataFrame:
    filtered = data.loc[data["year"].between(years[0], years[1])]
    if origin is not None:
        filtered = filtered.loc[filtered["origin"] == origin]
    if cylinders:
        filtered = filtered.loc[filtered["cylinders"].isin(cylinders)]
    if minimum_mpg is not None:
        filtered = filtered.loc[filtered["mpg"] >= minimum_mpg]
    return filtered.copy()
