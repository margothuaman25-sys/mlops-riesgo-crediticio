"""
ft_engineering.py
------------------
Ingeniería de características para el modelo de riesgo crediticio (Pago_atiempo).
Pipeline reproducible: se reutiliza en entrenamiento (model_training_evaluation.py)
y en producción (model_deploy.py) para garantizar que los datos nuevos se transformen
exactamente igual que los de entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COL = "Pago_atiempo"

# Columnas identificadas en el EDA (comprension_eda.ipynb)
NUM_COLS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
]
CAT_COLS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos_limpia"]


def load_data(path: str = "../Base_de_datos_eda.csv") -> pd.DataFrame:
    """Carga el dataset ya tratado en comprension_eda.ipynb."""
    return pd.read_csv(path)


def clean_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recrea la limpieza de tendencia_ingresos por si se llama este módulo
    directamente sobre el csv crudo (fuera del notebook de EDA).
    """
    if "tendencia_ingresos_limpia" not in df.columns and "tendencia_ingresos" in df.columns:
        categorias_validas = ["Estable", "Creciente", "Decreciente"]
        es_valida = df["tendencia_ingresos"].isin(categorias_validas)
        df["tendencia_ingresos_limpia"] = df["tendencia_ingresos"].where(es_valida, other="Desconocido")
    return df


RANGOS_VALIDOS = {
    "edad_cliente": (18, 100),
    "puntaje": (0, 100),
    "puntaje_datacredito": (150, 999),
}


def filtrar_outliers_por_rango(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con valores físicamente imposibles (detectado en el EDA),
    en vez de imputarlos: son errores de captura, no ausencias de dato.
    """
    df = df.copy()
    for col, (lo, hi) in RANGOS_VALIDOS.items():
        if col in df.columns:
            df = df[df[col].between(lo, hi)]
    return df


def winsorizar(df: pd.DataFrame, cols, percentil: float = 0.99) -> pd.DataFrame:
    """
    Recorta (cap) la cola extrema de columnas con outliers absurdos
    (ej. salario_cliente, total_otros_prestamos, cant_creditosvigentes) al
    percentil dado, en vez de eliminar esas filas: preserva casos reales
    (ej. clientes con varios microcréditos) sin dejar que distorsionen el modelo.
    """
    df = df.copy()
    for col in cols:
        limite = df[col].quantile(percentil)
        df[col] = df[col].clip(upper=limite)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas identificadas como útiles en el EDA."""
    df = df.copy()
    df["ratio_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"].replace(0, np.nan)
    df["ratio_cuota_salario"] = df["ratio_cuota_salario"].fillna(df["ratio_cuota_salario"].median())
    df["ratio_deuda_capital"] = df["saldo_total"] / df["capital_prestado"].replace(0, np.nan)
    df["ratio_deuda_capital"] = df["ratio_deuda_capital"].fillna(0)
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Pipeline de imputación + escalado + encoding, listo para producción."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    extra_num_cols = ["ratio_cuota_salario", "ratio_deuda_capital"]
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUM_COLS + extra_num_cols),
        ("cat", categorical_pipeline, CAT_COLS),
    ])
    return preprocessor


if __name__ == "__main__":
    df = load_data()
    df = clean_tendencia_ingresos(df)
    df = filtrar_outliers_por_rango(df)
    df = winsorizar(df, ["salario_cliente", "total_otros_prestamos", "cant_creditosvigentes"])
    df = add_derived_features(df)
    print(f"Filas: {df.shape[0]}, Columnas finales: {df.shape[1]}")
    print("Numéricas:", NUM_COLS)
    print("Categóricas:", CAT_COLS)
