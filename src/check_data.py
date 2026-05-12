"""Verificación inicial del dataset de Airbnb (listings)."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "listings.csv"

NULL_REPORT_COLS = [
    "accommodates",
    "bedrooms",
    "beds",
    "neighbourhood_cleansed",
    "price",
    "bathrooms_qty",
]


def clean_price(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )


def extract_bathrooms_qty(series: pd.Series) -> pd.Series:
    pattern = re.compile(r"(\d+(?:\.\d+)?)")

    def first_number(text: object) -> float | pd.NA:
        if pd.isna(text):
            return pd.NA
        m = pattern.search(str(text))
        if not m:
            return pd.NA
        return float(m.group(1))

    return series.map(first_number)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df["price"] = clean_price(df["price"])

    if "bathrooms_text" not in df.columns:
        raise KeyError("Falta la columna 'bathrooms_text' en el CSV.")
    df["bathrooms_qty"] = extract_bathrooms_qty(df["bathrooms_text"])

    print(f"Total de registros: {len(df)}")
    print("\nColumnas:")
    print(list(df.columns))
    print("\nValores nulos (conteo):")
    for col in NULL_REPORT_COLS:
        if col not in df.columns:
            print(f"  {col}: (columna no presente en el dataset)")
        else:
            print(f"  {col}: {df[col].isna().sum()}")


if __name__ == "__main__":
    main()
