"""Entrenamiento avanzado de precio de Airbnb con XGBoost.

- Limpieza de precio + recorte por percentiles (1, 95).
- Target en escala logarítmica (log1p).
- Pipeline con ColumnTransformer + XGBRegressor.
- Evaluación en escala real (USD) tras invertir el log.
- Exporta el pipeline entrenado a models/modelo_final_airbnb.joblib.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "listings.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "modelo_final_airbnb.joblib"

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

# Recorte por percentiles del precio para excluir errores de carga (extremos
# muy bajos) y propiedades de lujo que distorsionan el ajuste.
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.95

RANDOM_STATE = 42
TEST_SIZE = 0.2

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


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

    def first_number(text: object) -> float:
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


def filter_price_percentiles(
    df: pd.DataFrame, lower: float, upper: float
) -> pd.DataFrame:
    """Mantiene filas con price entre los percentiles `lower` y `upper`."""
    df_valid = df.dropna(subset=[TARGET])
    low_q = df_valid[TARGET].quantile(lower)
    high_q = df_valid[TARGET].quantile(upper)
    mask = df_valid[TARGET].between(low_q, high_q)
    filtered = df_valid.loc[mask].copy()

    print(
        f"Recorte de precio: p{int(lower * 100)}={low_q:,.2f} USD | "
        f"p{int(upper * 100)}={high_q:,.2f} USD"
    )
    print(
        f"Filas tras recorte: {len(filtered):,} "
        f"(antes: {len(df_valid):,}, eliminadas: {len(df_valid) - len(filtered):,})"
    )
    return filtered


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer con imputación + escalado / one-hot."""
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
    """Pipeline: preprocesamiento + XGBoost."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(**XGB_PARAMS)),
        ]
    )


def evaluate_in_usd(
    y_true_log: pd.Series, y_pred_log: np.ndarray
) -> dict[str, float]:
    """Métricas calculadas en escala real (USD) tras invertir log1p."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> None:
    df = load_and_prepare(DATA_PATH)
    df_model = filter_price_percentiles(df, LOWER_QUANTILE, UPPER_QUANTILE)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df_model[feature_cols]
    y = np.log1p(df_model[TARGET])

    print(f"\nFeatures numéricas:   {NUMERIC_FEATURES}")
    print(f"Features categóricas: {CATEGORICAL_FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate_in_usd(y_train, pipeline.predict(X_train))
    test_metrics = evaluate_in_usd(y_test, pipeline.predict(X_test))

    print("\n=== Métricas en USD (train) ===")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:,.4f}")

    print("\n=== Métricas en USD (test) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:,.4f}")

    print(f"\nR² final (test): {test_metrics['R2']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nPipeline guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
