"""Entrenamiento base (Regresión Lineal) para predicción de precio en Airbnb.

Lee el CSV crudo, aplica la limpieza acordada en el EDA, monta un Pipeline
de preprocesamiento + modelo y reporta métricas en hold-out.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "listings.csv"

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "availability_365",
    "bathrooms_qty",
    "amenities_count",
]

CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "room_type",
    "host_is_superhost",
    "instant_bookable",
]

TARGET = "price"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_price(series: pd.Series) -> pd.Series:
    """Convierte 'price' textual ('$1,234.50') a float."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )


def extract_bathrooms_qty(series: pd.Series) -> pd.Series:
    """Extrae el primer número de 'bathrooms_text' (p. ej. '1.5 baths' -> 1.5)."""
    pattern = re.compile(r"(\d+(?:\.\d+)?)")

    def first_number(text: object) -> float | None:
        if pd.isna(text):
            return np.nan
        m = pattern.search(str(text))
        return float(m.group(1)) if m else np.nan

    return series.map(first_number)


def count_amenities(val: object) -> float:
    """Cuenta los ítems de la lista serializada en 'amenities'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return np.nan
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return float(len(parsed))
    except (ValueError, SyntaxError, TypeError):
        pass
    inner = s.strip("[]")
    if not inner:
        return 0.0
    parts = [p.strip().strip('"') for p in inner.split(",")]
    return float(len([p for p in parts if p]))


def load_and_prepare(data_path: Path) -> pd.DataFrame:
    """Carga el CSV crudo y aplica las transformaciones del EDA."""
    head = data_path.read_bytes()[:200]
    if b"git-lfs.github.com" in head:
        raise RuntimeError(
            "data/listings.csv es el puntero de Git LFS. "
            "Ejecuta `git lfs pull` en la raíz del repositorio."
        )

    df = pd.read_csv(data_path, low_memory=False)
    df["price"] = clean_price(df["price"])
    df["bathrooms_qty"] = extract_bathrooms_qty(df["bathrooms_text"])
    df["amenities_count"] = df["amenities"].map(count_amenities)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Preprocesamiento: imputación + escalado/one-hot por tipo de variable."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Pipeline completo: preprocesamiento + regresión lineal base."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
    }


def main() -> None:
    df = load_and_prepare(DATA_PATH)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df_model = df.dropna(subset=[TARGET]).copy()
    X = df_model[feature_cols]
    y = df_model[TARGET]

    print(f"Filas con target válido: {len(df_model)}")
    print(f"Features numéricas:   {NUMERIC_FEATURES}")
    print(f"Features categóricas: {CATEGORICAL_FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate(y_train, pipeline.predict(X_train))
    test_metrics = evaluate(y_test, pipeline.predict(X_test))

    print("\n=== Métricas de entrenamiento ===")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:,.4f}")

    print("\n=== Métricas en hold-out (test) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:,.4f}")


if __name__ == "__main__":
    main()
