"""Streamlit app — Predicción de precios de Airbnb en Río de Janeiro.

Usa el pipeline LightGBM optimizado guardado en `models/modelo_optimo.joblib`.
El pipeline es un `TransformedTargetRegressor` (log1p / expm1) que devuelve
precio en USD directamente; no se aplica `expm1` adicional en la app.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modelo_optimo.joblib"
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

# Mismas funciones de limpieza que en `train_advanced.py`,
# adaptadas a escalares para la app.

_BATH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def extract_bathrooms_qty(text: str | None) -> float:
    """De '2 baths' -> 2.0; '1.5 private baths' -> 1.5; vacío -> NaN."""
    if text is None:
        return float("nan")
    s = str(text).strip()
    if not s:
        return float("nan")
    m = _BATH_PATTERN.search(s)
    return float(m.group(1)) if m else float("nan")


def count_amenities(text: str | None) -> float:
    """Cuenta los elementos en una lista textual de amenities.

    Acepta tanto el formato del dataset (`["Wifi", "TV"]`) como entrada
    natural de usuario (`Wifi, TV, Kitchen`).
    """
    if text is None:
        return 0.0
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return 0.0
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return float(len(parsed))
    except (ValueError, SyntaxError, TypeError):
        pass
    inner = s.strip("[]")
    if not inner:
        return 0.0
    parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
    return float(len([p for p in parts if p]))


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero `python src/optimize_model.py`."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Cargando catálogos del dataset...")
def load_choices() -> tuple[list[str], list[str]]:
    """Lista de barrios y `room_type` desde el dataset original."""
    fallback_barrios = [
        "Copacabana", "Ipanema", "Leblon", "Botafogo", "Barra da Tijuca",
        "Centro", "Santa Teresa", "Lapa", "Flamengo", "Catete",
    ]
    fallback_rooms = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

    if not DATA_PATH.is_file():
        return fallback_barrios, fallback_rooms
    head = DATA_PATH.read_bytes()[:200]
    if b"git-lfs.github.com" in head:
        return fallback_barrios, fallback_rooms

    df = pd.read_csv(
        DATA_PATH,
        usecols=["neighbourhood_cleansed", "room_type"],
        low_memory=False,
    )
    barrios = sorted(df["neighbourhood_cleansed"].dropna().astype(str).unique().tolist())
    rooms = sorted(df["room_type"].dropna().astype(str).unique().tolist())
    return barrios or fallback_barrios, rooms or fallback_rooms


def predict_price(model, payload: dict) -> float:
    """Construye el DataFrame con el orden de columnas esperado y predice."""
    feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pd.DataFrame([{col: payload[col] for col in feature_order}])
    pred = float(model.predict(X)[0])
    # El modelo (TransformedTargetRegressor con inverse_func=expm1) ya devuelve
    # el precio en escala USD, así que NO aplicamos `np.expm1` aquí.
    return max(pred, 0.0)


def render() -> None:
    st.set_page_config(
        page_title="Predicción de Precios Airbnb - Río",
        page_icon="🏠",
        layout="centered",
    )
    st.title("🏠 Predicción de Precios Airbnb — Río de Janeiro")
    st.caption(
        "Modelo LightGBM con preprocesamiento "
        "(imputación mediana + one-hot) y target en escala log1p."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    barrios, rooms = load_choices()
    default_neigh = barrios.index("Copacabana") if "Copacabana" in barrios else 0
    default_room = rooms.index("Entire home/apt") if "Entire home/apt" in rooms else 0

    with st.sidebar:
        st.header("Características del alojamiento")

        st.subheader("Capacidad")
        accommodates = st.slider("Personas (`accommodates`)", 1, 16, 2)
        bedrooms = st.number_input("Habitaciones (`bedrooms`)", 0, 10, 1, step=1)
        beds = st.number_input("Camas (`beds`)", 0, 20, 1, step=1)
        bathrooms_text = st.text_input(
            "Baños (texto, p.ej. '1 bath', '1.5 baths')", "1 bath"
        )

        st.subheader("Ubicación y tipo")
        neighbourhood = st.selectbox(
            "Barrio (`neighbourhood_cleansed`)", barrios, index=default_neigh
        )
        room_type = st.selectbox("Tipo de habitación", rooms, index=default_room)

        st.subheader("Política de reserva")
        minimum_nights = st.number_input(
            "Noches mínimas", min_value=1, max_value=365, value=2, step=1
        )
        availability_365 = st.slider(
            "Días disponibles (últimos 365)", 0, 365, 200
        )
        superhost = st.radio("¿Superhost?", ["Sí", "No"], index=1, horizontal=True)
        instant = st.radio(
            "¿Reserva instantánea?", ["Sí", "No"], index=1, horizontal=True
        )

        st.subheader("Reviews")
        number_of_reviews = st.number_input(
            "Número de reviews", min_value=0, max_value=2000, value=25, step=1
        )
        review_scores_rating = st.slider(
            "Score de reviews (0–5)", 0.0, 5.0, 4.7, 0.1
        )

        st.subheader("Amenities")
        amenities_text = st.text_area(
            "Lista (separadas por comas)",
            value="Wifi, TV, Cocina, Aire acondicionado, Lavadora",
            height=80,
        )

        submit = st.button(
            "Calcular precio", type="primary", use_container_width=True
        )

    st.markdown(
        "Configura las características del alojamiento en la barra lateral y "
        "pulsa **Calcular precio** para ver la estimación del modelo."
    )

    if not submit:
        return

    bathrooms_qty = extract_bathrooms_qty(bathrooms_text)
    amenities_count = count_amenities(amenities_text)

    payload = {
        "accommodates": int(accommodates),
        "bedrooms": float(bedrooms),
        "beds": float(beds),
        "minimum_nights": int(minimum_nights),
        "number_of_reviews": int(number_of_reviews),
        "review_scores_rating": float(review_scores_rating),
        "availability_365": int(availability_365),
        "bathrooms_qty": bathrooms_qty,
        "amenities_count": amenities_count,
        "neighbourhood_cleansed": neighbourhood,
        "room_type": room_type,
        # El dataset original codifica superhost / instant_bookable como 't'/'f'.
        "host_is_superhost": "t" if superhost == "Sí" else "f",
        "instant_bookable": "t" if instant == "Sí" else "f",
    }

    price_usd = predict_price(model, payload)
    price_brl = price_usd * 5.0  # tipo de cambio aproximado USD->BRL

    st.subheader("Resultado de la predicción")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio estimado (USD/noche)", f"${price_usd:,.2f}")
    with col2:
        st.metric("Aprox. en BRL (≈ 5 USD/BRL)", f"R$ {price_brl:,.2f}")

    with st.expander("Ver features enviadas al modelo"):
        feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        df_payload = pd.DataFrame(
            [(col, payload[col]) for col in feature_order],
            columns=["feature", "valor"],
        )
        st.dataframe(df_payload, use_container_width=True, hide_index=True)

    st.caption(
        "Métricas de validación cruzada (5-fold): R² ≈ 0.46, "
        "MAE ≈ 145 USD, RMSE ≈ 239 USD."
    )


if __name__ == "__main__":
    render()
