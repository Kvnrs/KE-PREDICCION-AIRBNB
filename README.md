# Predicción de Precios de Airbnb en Río de Janeiro

> Proyecto end-to-end de Machine Learning para estimar el precio por noche de
> alojamientos de Airbnb en Río de Janeiro, desde el EDA hasta una **app
> interactiva en Streamlit**.

## Descripción

A partir del dataset público de Inside Airbnb (Río), el proyecto cubre el
flujo completo de MLOps:

1. **EDA interactivo** con Plotly: distribución de precios, mapa
   geográfico (`open-street-map`), barrios más caros y relación
   `amenities ↔ price`.
2. **Limpieza específica del dataset**: parseo de `price` (`$`, `,`),
   extracción del número desde `bathrooms_text` y conteo de elementos en
   `amenities`.
3. **Modelos comparados**: Regresión Lineal (baseline), XGBoost, LightGBM
   y CatBoost, con `TransformedTargetRegressor(log1p / expm1)` para
   reportar métricas en USD reales.
4. **Optimización**: recorte de precios por percentiles (p1, p95) +
   `RandomizedSearchCV` sobre el ganador.
5. **App Streamlit**: formulario lateral con todas las features y
   estimación del precio por noche con `st.metric`.

## Tecnologías

| Capa | Librerías |
|---|---|
| Datos / análisis | `pandas`, `numpy`, `plotly`, `nbformat` |
| Modelado | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Optimización | `RandomizedSearchCV`, `optuna`, `scikit-optimize` |
| App | `streamlit` |
| Persistencia | `joblib` |
| Datos grandes | **Git LFS** (`data/*.csv`) |

## Estructura del repositorio

```
KE-PREDICCION-AIRBNB/
├── app/
│   └── main.py                  # Streamlit
├── data/
│   ├── listings.csv             # CSV crudo (Git LFS)
│   └── listings_cleaned.csv     # CSV tras limpieza del EDA (Git LFS)
├── models/
│   ├── modelo_final_airbnb.joblib   # XGBoost (paso 3)
│   └── modelo_optimo.joblib         # LightGBM tuneado (paso 4, producción)
├── notebooks/
│   └── 01_EDA_Airbnb.ipynb
├── src/
│   ├── check_data.py            # Verificación inicial
│   ├── train.py                 # Baseline LinearRegression
│   ├── train_advanced.py        # XGBoost + log1p + recorte
│   └── optimize_model.py        # Benchmark + tuning
├── .gitattributes               # Reglas Git LFS
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación

### Requisitos
- Python **3.10+** (probado en 3.14; en Streamlit Community Cloud usar 3.12).
- [Git LFS](https://git-lfs.github.com/) instalado para descargar los CSV.

### Pasos

```bash
git clone https://github.com/Kvnrs/KE-PREDICCION-AIRBNB.git
cd KE-PREDICCION-AIRBNB
git lfs pull

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

## Uso

### Verificación rápida de los datos

```bash
python src/check_data.py
```
Imprime total de registros, columnas y conteo de nulos.

### EDA con Plotly

Abre el notebook con tu editor preferido (Cursor, VS Code, Jupyter):

```bash
jupyter notebook notebooks/01_EDA_Airbnb.ipynb
```

Genera `data/listings_cleaned.csv` al final.

### Entrenamiento

```bash
python src/train.py            # Baseline LinearRegression
python src/train_advanced.py   # XGBoost + log1p + recorte de outliers
python src/optimize_model.py   # Benchmark (5 modelos) + tuning del ganador
```

### App Streamlit

```bash
streamlit run app/main.py
```
Abre `http://localhost:8501` y configura los inputs del alojamiento en la
barra lateral para obtener una estimación del precio.

## Resultados

Comparativa final en escala USD (precio por noche). El XGBoost del paso 3 se
midió en *hold-out 20%* y los de la fase de comparación con CV 3-fold tras
recortar `price` a `[p1, p95]`.

| Modelo | R² | MAE (USD) | RMSE (USD) |
|---|---:|---:|---:|
| Baseline — LinearRegression | 0.030 | 632.35 | 3 645.87 |
| XGBoost (paso 3) | 0.437 | 147.90 | 242.32 |
| **LightGBM tuneado (producción)** | **0.461** | **145.28** | **239.17** |
| CatBoost | 0.442 | 147.34 | 242.93 |
| RandomForest | 0.425 | 150.78 | 246.68 |

Mejor modelo: **LightGBM** con `n_estimators=300`, `learning_rate=0.05`,
`num_leaves=127`, `subsample=0.7`, `colsample_bytree=0.7`.

Pipeline en producción: `models/modelo_optimo.joblib`
(`TransformedTargetRegressor(log1p / expm1)` envolviendo
`Pipeline(preprocessor + LGBMRegressor)`). `model.predict(X)` devuelve el
precio directamente en USD.

## Despliegue en Streamlit Community Cloud

[Streamlit Community Cloud](https://share.streamlit.io) permite desplegar la
app directamente desde el repo de GitHub.

### Prerrequisitos
- Repositorio **público** o autorizado en la cuenta de Streamlit.
- `requirements.txt` en la raíz (✓).
- Punto de entrada: `app/main.py` (✓).

### Pasos
1. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión
   con la cuenta de GitHub.
2. Pulsa **New app**.
3. Configura:
   - **Repository:** `Kvnrs/KE-PREDICCION-AIRBNB`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **Python version (Advanced settings):** `3.12` (recomendado;
     Streamlit Cloud aún no soporta 3.14).
4. (Opcional pero recomendado) En **Advanced settings → Repository
   secrets**, no hace falta configurar nada para este proyecto.
5. **LFS:** la app no necesita el CSV completo para funcionar (los
   selectbox tienen un fallback de barrios y `room_type`). Si quieres que
   se carguen los catálogos completos desde `data/listings.csv`, activa
   **Use Git LFS** en *Advanced settings*. En caso contrario, la app
   detecta el puntero LFS y usa el fallback automáticamente.
6. Pulsa **Deploy**. La primera build instala `requirements.txt`; tarda
   unos minutos.

### Cada push a `main` redepliega automáticamente

Para forzar un rebuild manual: en la app desplegada, **Manage app →
Reboot app**.

### Notas para producción
- El `joblib` pesa ~3.7 MB → entra holgadamente en los límites del plan
  gratuito (~1 GB RAM).
- Si `pip install` se vuelve lento, fija versiones en `requirements.txt`
  para evitar resoluciones de dependencias largas.
- El warning *“X does not have valid feature names…”* de LightGBM es
  cosmético: no afecta predicciones.

## Roadmap

- Validación temporal (splits por mes) para precios estacionales.
- Features adicionales: distancia a playas/aeropuerto, longitud del
  título, embeddings de `description`.
- Tuning más amplio con **Optuna** y *early stopping*.
- Tests automatizados (`pytest`) sobre las funciones de limpieza.

## Licencia

Uso académico / portafolio. Dataset original: Inside Airbnb (CC0).
