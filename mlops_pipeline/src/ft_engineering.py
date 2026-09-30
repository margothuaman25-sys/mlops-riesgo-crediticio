"""
ft_engineering.py
------------------
Ingenieria de caracteristicas para el modelo de riesgo crediticio.

Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autor: Shekina

Tech stack usado: pandas, numpy, scikit-learn, feature-engine.

Este script toma la base cruda (Base_de_datos.csv) y devuelve un dataframe
listo para entrenar (Base_de_datos_tipada.csv), aplicando:
    1. Limpieza de outliers imposibles por rango de negocio (regla de
       negocio manual, ya que no es un caso estadistico generico).
    2. Imputacion de esos outliers + Winsorizacion de colas extremas,
       usando feature_engine.imputation y feature_engine.outliers.
    3. Limpieza de la corrupcion en tendencia_ingresos (regla de negocio)
       + imputacion como 4ta categoria, con feature_engine.imputation.
    4. Imputacion de promedio_ingresos_datacredito con la tecnica
       "flag + imputacion por grupo" (flag con feature_engine.imputation.
       AddMissingIndicator).
    5. Imputacion de saldos con feature_engine.imputation.ArbitraryNumberImputer.
    6. Codificacion de variables categoricas con feature_engine.encoding.OneHotEncoder.
"""

import numpy as np
import pandas as pd

from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder


# ---------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga la base de datos cruda."""
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------
# 2. OUTLIERS IMPOSIBLES POR RANGO DE NEGOCIO
# ---------------------------------------------------------------
def marcar_outliers_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca como nulos los valores que son imposibles segun el negocio
    (no segun estadistica): edad_cliente [18-100], puntaje [0-100],
    puntaje_datacredito [150-999]. La regla de negocio es especifica
    de este dominio, por eso se hace manual; la IMPUTACION posterior
    ya usa feature-engine (ver imputar_outliers_negocio).
    """
    df = df.copy()
    df.loc[(df['edad_cliente'] < 18) | (df['edad_cliente'] > 100), 'edad_cliente'] = np.nan
    df.loc[(df['puntaje'] < 0) | (df['puntaje'] > 100), 'puntaje'] = np.nan
    df.loc[(df['puntaje_datacredito'] < 150) | (df['puntaje_datacredito'] > 999), 'puntaje_datacredito'] = np.nan
    print("Outliers de negocio marcados como nulos en edad_cliente, puntaje y puntaje_datacredito.")
    return df


def imputar_outliers_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa con la MEDIANA (feature_engine.imputation.MeanMedianImputer)
    los valores marcados como nulos en el paso anterior. Son pocos casos
    (<2% cada uno), asi que la mediana es una estimacion segura.
    """
    df = df.copy()
    columnas = ['edad_cliente', 'puntaje', 'puntaje_datacredito']
    n_antes = df[columnas].isna().sum()

    imputer = MeanMedianImputer(imputation_method='median', variables=columnas)
    df = imputer.fit_transform(df)

    for col in columnas:
        print(f"  {col}: {n_antes[col]} nulos (ex-outliers) imputados con la mediana.")
    return df


# ---------------------------------------------------------------
# 3. WINSORIZACION (CAP AL PERCENTIL 99) CON FEATURE-ENGINE
# ---------------------------------------------------------------
def winsorizar(df: pd.DataFrame, columnas: list, fold: float = 0.01) -> pd.DataFrame:
    """
    Usa feature_engine.outliers.Winsorizer para recortar (cap) los
    valores por encima del percentil 99 en las columnas indicadas.
    Equivale a: limite = percentil 99, y cualquier valor mayor se
    recorta a ese limite.
    """
    df = df.copy()
    limites_antes = {col: df[col].quantile(1 - fold) for col in columnas}
    conteos = {col: (df[col] > limites_antes[col]).sum() for col in columnas}

    winsor = Winsorizer(capping_method='quantiles', tail='right', fold=fold, variables=columnas)
    df = winsor.fit_transform(df)

    for col in columnas:
        print(f"  {col}: {conteos[col]} valores recortados al percentil {1-fold} (limite={limites_antes[col]:,.0f})")
    return df


# ---------------------------------------------------------------
# 4. LIMPIEZA DE tendencia_ingresos (CORRUPCION + NULOS)
# ---------------------------------------------------------------
def limpiar_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    tendencia_ingresos deberia tener solo 3 categorias:
    'Estable', 'Creciente', 'Decreciente'. Se detectaron ~58 filas con
    numeros en vez de esas categorias (error de mapeo de columnas en el
    origen de datos) -> se tratan como corruptos y se pasan a nulo
    (regla de negocio manual).

    Los nulos (originales + corruptos) se imputan como una 4ta categoria
    explicita "Sin_dato", usando feature_engine.imputation.CategoricalImputer.
    No se imputa con la moda para no inventar informacion que no existe;
    la ausencia del dato puede ser informativa en si misma.
    """
    df = df.copy()
    categorias_validas = ['Estable', 'Creciente', 'Decreciente']

    es_corrupto = ~df['tendencia_ingresos'].isin(categorias_validas) & df['tendencia_ingresos'].notna()
    n_corruptos = es_corrupto.sum()
    df.loc[es_corrupto, 'tendencia_ingresos'] = np.nan
    n_nulos_total = df['tendencia_ingresos'].isna().sum()

    imputer = CategoricalImputer(imputation_method='missing', fill_value='Sin_dato',
                                  variables=['tendencia_ingresos'])
    df = imputer.fit_transform(df)

    print(f"tendencia_ingresos: {n_corruptos} valores corruptos detectados y anulados.")
    print(f"tendencia_ingresos: {n_nulos_total} nulos totales imputados como 'Sin_dato'.")
    return df


# ---------------------------------------------------------------
# 5. IMPUTACION DE promedio_ingresos_datacredito (FLAG + GRUPO)
# ---------------------------------------------------------------
def imputar_promedio_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tecnica "flag + imputacion":
    1. Se crea una columna binaria (feature_engine.imputation.
       AddMissingIndicator) que indica si el dato existia originalmente
       (1) o va a ser imputado (0).
    2. Se imputa el nulo con la MEDIANA calculada dentro de cada grupo
       de tipo_laboral (Independiente / Empleado). feature-engine no
       soporta imputacion agrupada nativamente, asi que este paso puntual
       se hace con pandas (groupby), manteniendo el resto del pipeline
       en feature-engine.
    """
    df = df.copy()
    col = 'promedio_ingresos_datacredito'
    n_nulos = df[col].isna().sum()

    flag_imputer = AddMissingIndicator(variables=[col])
    df = flag_imputer.fit_transform(df)
    df = df.rename(columns={f"{col}_na": "tiene_promedio_ingresos"})
    df["tiene_promedio_ingresos"] = 1 - df["tiene_promedio_ingresos"]  # 1=dato real, 0=imputado

    df[col] = df.groupby('tipo_laboral')[col].transform(lambda x: x.fillna(x.median()))

    print(f"{col}: {n_nulos} nulos imputados con mediana por tipo_laboral.")
    print("Flag 'tiene_promedio_ingresos' creado con feature_engine.imputation.AddMissingIndicator.")
    return df


# ---------------------------------------------------------------
# 6. IMPUTACION DE SALDOS CON feature-engine
# ---------------------------------------------------------------
def imputar_saldos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Los nulos en saldo_mora, saldo_total, saldo_principal y
    saldo_mora_codeudor representan, con alta probabilidad, ausencia de
    deuda registrada -> se imputan con 0 usando
    feature_engine.imputation.ArbitraryNumberImputer.
    """
    df = df.copy()
    columnas_saldo = [c for c in
                       ['saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor']
                       if c in df.columns]
    n_antes = df[columnas_saldo].isna().sum()

    imputer = ArbitraryNumberImputer(arbitrary_number=0, variables=columnas_saldo)
    df = imputer.fit_transform(df)

    for col in columnas_saldo:
        print(f"  {col}: {n_antes[col]} nulos imputados con 0.")
    return df


# ---------------------------------------------------------------
# 7. CODIFICACION DE VARIABLES CATEGORICAS (feature-engine OneHotEncoder)
# ---------------------------------------------------------------
def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las variables categoricas en columnas dummy (0/1) usando
    feature_engine.encoding.OneHotEncoder, para que puedan entrar a los
    modelos de scikit-learn / xgboost.
    """
    df = df.copy()
    columnas_categoricas = [c for c in ['tipo_laboral', 'tipo_credito', 'tendencia_ingresos']
                             if c in df.columns]

    # tipo_credito llega como codigo numerico (1,2,3...) pero es categorico
    # por naturaleza (representa un tipo de credito, no una cantidad).
    if 'tipo_credito' in df.columns:
        df['tipo_credito'] = df['tipo_credito'].astype(str)

    encoder = OneHotEncoder(variables=columnas_categoricas, drop_last=True)
    df = encoder.fit_transform(df)

    print(f"Variables codificadas (One-Hot, feature-engine): {columnas_categoricas}")
    return df


# ---------------------------------------------------------------
# 8. PIPELINE COMPLETO
# ---------------------------------------------------------------
def ejecutar_feature_engineering(ruta_entrada: str, ruta_salida: str) -> pd.DataFrame:
    """Orquesta todo el proceso de feature engineering, paso a paso."""
    print("=" * 60)
    print("INICIANDO FEATURE ENGINEERING")
    print("=" * 60)

    df = cargar_datos(ruta_entrada)

    print("\n--- Paso 1: Outliers imposibles por rango de negocio ---")
    df = marcar_outliers_negocio(df)
    df = imputar_outliers_negocio(df)

    print("\n--- Paso 2: Winsorizacion de colas extremas ---")
    df = winsorizar(df, ['salario_cliente', 'total_otros_prestamos', 'cant_creditosvigentes'])

    print("\n--- Paso 3: Limpieza de tendencia_ingresos ---")
    df = limpiar_tendencia_ingresos(df)

    print("\n--- Paso 4: Imputacion de promedio_ingresos_datacredito ---")
    df = imputar_promedio_ingresos(df)

    print("\n--- Paso 5: Imputacion de saldos ---")
    df = imputar_saldos(df)

    print("\n--- Paso 6: Codificacion de categoricas ---")
    df = codificar_categoricas(df)

    df.to_csv(ruta_salida, index=False)
    print(f"\nBase final guardada en: {ruta_salida}")
    print(f"Shape final: {df.shape}")
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETADO")
    print("=" * 60)

    return df


if __name__ == "__main__":
    ejecutar_feature_engineering(
        ruta_entrada="../../Base_de_datos.csv",
        ruta_salida="../../Base_de_datos_tipada.csv"
    )
