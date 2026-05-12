"""Comparación de modelos y optimización de hiperparámetros.

Implementación sin PyCaret (no compatible con Python 3.14 a fecha de hoy).
Mismo objetivo: validar si XGBoost sigue siendo el mejor frente a otros
regresores (LightGBM, CatBoost, Random Forest, Gradient Boosting, Ridge).

Flujo:
- Limpieza + recorte por percentiles (p1, p95) como en `train_advanced.py`.
- `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` para que
  el modelo aprenda en escala logarítmica pero las métricas de CV se reporten
  en USD directos.
- `cross_validate` 5-fold sobre cada candidato → leaderboard.
- `RandomizedSearchCV` afina el ganador.
- Pipeline final exportado a `models/modelo_optimo.joblib`.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "listings.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "modelo_optimo.joblib"

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

LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.95
RANDOM_STATE = 42
N_FOLDS = 3
TUNE_FOLDS = 3
N_ITER_SEARCH = 10

CV_SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
}


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

    def first_number(text: object) -> float:
        if pd.isna(text):
            return np.nan
        m = pattern.search(str(text))
        return float(m.group(1)) if m else np.nan

    return series.map(first_number)


def count_amenities(val: object) -> float:
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
    df_valid = df.dropna(subset=[TARGET])
    low_q = df_valid[TARGET].quantile(lower)
    high_q = df_valid[TARGET].quantile(upper)
    mask = df_valid[TARGET].between(low_q, high_q)
    filtered = df_valid.loc[mask].copy()
    print(
        f"Recorte de precio: p{int(lower * 100)}={low_q:,.2f} | "
        f"p{int(upper * 100)}={high_q:,.2f} USD"
    )
    print(f"Filas tras recorte: {len(filtered):,} (de {len(df_valid):,}).")
    return filtered


def build_preprocessor() -> ColumnTransformer:
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


def wrap_in_pipeline(estimator) -> TransformedTargetRegressor:
    """Pipeline = preprocesador + modelo, envuelto en log1p/expm1."""
    inner = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(
        regressor=inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def candidate_models() -> dict[str, object]:
    """Estimadores base para la comparación."""
    return {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    }


def param_grid_for(model_name: str) -> dict[str, list]:
    """Espacios de búsqueda para `RandomizedSearchCV`. Vacío => no se tunea."""
    if model_name == "XGBoost":
        return {
            "regressor__model__n_estimators": [300, 500, 800, 1200],
            "regressor__model__max_depth": [4, 6, 8, 10],
            "regressor__model__learning_rate": [0.02, 0.05, 0.1],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
            "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
            "regressor__model__reg_lambda": [0.5, 1.0, 2.0],
            "regressor__model__min_child_weight": [1, 5, 10],
        }
    if model_name == "LightGBM":
        return {
            "regressor__model__n_estimators": [300, 500, 800, 1200],
            "regressor__model__learning_rate": [0.02, 0.05, 0.1],
            "regressor__model__num_leaves": [31, 63, 127],
            "regressor__model__max_depth": [-1, 6, 10],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
            "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
            "regressor__model__min_child_samples": [10, 20, 40],
        }
    if model_name == "CatBoost":
        return {
            "regressor__model__iterations": [300, 500, 800, 1200],
            "regressor__model__depth": [4, 6, 8, 10],
            "regressor__model__learning_rate": [0.02, 0.05, 0.1],
            "regressor__model__l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            "regressor__model__bagging_temperature": [0.0, 0.5, 1.0],
        }
    if model_name == "RandomForest":
        return {
            "regressor__model__n_estimators": [200, 400, 800],
            "regressor__model__max_depth": [None, 10, 20, 30],
            "regressor__model__min_samples_split": [2, 5, 10],
            "regressor__model__min_samples_leaf": [1, 2, 4],
            "regressor__model__max_features": ["sqrt", 0.5, 1.0],
        }
    return {}


def benchmark(
    X: pd.DataFrame, y: pd.Series, models: dict[str, object]
) -> pd.DataFrame:
    """Cross-validate cada candidato y devuelve leaderboard."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, estimator in models.items():
        print(f"  Evaluando {name}...", flush=True)
        pipeline = wrap_in_pipeline(estimator)
        cv = cross_validate(
            pipeline,
            X,
            y,
            cv=kf,
            scoring=CV_SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "R2": cv["test_R2"].mean(),
                "R2_std": cv["test_R2"].std(),
                "MAE": -cv["test_MAE"].mean(),
                "RMSE": -cv["test_RMSE"].mean(),
                "fit_time_s": cv["fit_time"].mean(),
            }
        )
    lb = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return lb


def tune_winner(
    name: str, estimator, X: pd.DataFrame, y: pd.Series
) -> tuple[TransformedTargetRegressor, dict | None, float | None]:
    """Si hay grilla definida, hace RandomizedSearchCV y devuelve el ganador."""
    grid = param_grid_for(name)
    pipeline = wrap_in_pipeline(estimator)
    if not grid:
        print(f"\n[!] Sin grilla definida para {name}; no se realiza tuning.")
        pipeline.fit(X, y)
        return pipeline, None, None

    print(f"\nAfinando {name} con RandomizedSearchCV "
          f"(n_iter={N_ITER_SEARCH}, cv={TUNE_FOLDS})...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=grid,
        n_iter=N_ITER_SEARCH,
        cv=KFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_and_prepare(DATA_PATH)
    df = filter_price_percentiles(df, LOWER_QUANTILE, UPPER_QUANTILE)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(float)

    print(f"\nFilas para CV: {len(df):,}")
    print(f"Features numéricas:   {NUMERIC_FEATURES}")
    print(f"Features categóricas: {CATEGORICAL_FEATURES}\n")

    models = candidate_models()
    print(f"Comparando {len(models)} modelos en {N_FOLDS}-fold CV...")
    leaderboard = benchmark(X, y, models)

    print("\n=== Leaderboard (CV en escala USD) ===")
    print(
        leaderboard.to_string(
            index=False,
            formatters={
                "R2": "{:.4f}".format,
                "R2_std": "{:.4f}".format,
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "fit_time_s": "{:.2f}".format,
            },
        )
    )

    best_name = leaderboard.iloc[0]["model"]
    best_base = models[best_name]
    print(f"\nGanador: {best_name} "
          f"(R² CV = {leaderboard.iloc[0]['R2']:.4f})")

    tuned_pipeline, best_params, best_cv_r2 = tune_winner(
        best_name, best_base, X, y
    )
    if best_params is not None:
        print("\nMejores hiperparámetros encontrados:")
        for k, v in best_params.items():
            print(f"  {k.replace('regressor__model__', '')}: {v}")
        print(f"\nR² CV tras tuning ({TUNE_FOLDS}-fold): {best_cv_r2:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipeline, MODEL_PATH)
    print(f"\nModelo final guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
