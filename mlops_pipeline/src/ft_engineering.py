"""
ft_engineering.py
------------------
Ingenieria de caracteristicas para el modelo de riesgo crediticio.
 
Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autora: Margot Huaman Quispe
 
Tech stack usado: pandas, numpy, scikit-learn, feature-engine.
 
El proceso se divide en dos partes para evitar FUGA DE DATOS (data leakage):
 
A) limpiar_base(df)  ->  reglas de NEGOCIO, fila por fila, sin aprender
   nada de los datos (rangos imposibles, categorias corruptas). Se puede
   aplicar a toda la base sin riesgo, porque no calcula estadisticas.
 
B) construir_preprocesador()  ->  pipeline de scikit-learn con transformadores
   de feature-engine que SI aprenden de los datos (medianas, limites de
   winsorizacion, categorias del One-Hot). Se ajusta (fit) SOLO con el
   conjunto de entrenamiento y luego se aplica al test y a los datos
   nuevos en produccion. Pasos del pipeline:
    1. Imputacion con mediana de los outliers de negocio (MeanMedianImputer).
    2. Winsorizacion de colas extremas (Winsorizer).
    3. Imputacion de nulos categoricos (tendencia_ingresos y tipo_laboral)
       como categoria "Sin_dato" (CategoricalImputer). tipo_laboral no
       tiene nulos hoy, pero un cliente nuevo en produccion podria traerlo.
    4. Flag de dato faltante en promedio_ingresos_datacredito
       (AddMissingIndicator).
    5. Imputacion de promedio_ingresos_datacredito con la mediana por
       tipo_laboral (transformador propio: feature-engine no la trae).
    6. Imputacion de saldos con 0 (ArbitraryNumberImputer).
    7. Codificacion One-Hot de variables categoricas (OneHotEncoder).
 
Uso por script (genera una base transformada solo para inspeccion/EDA):
    python ft_engineering.py
El entrenamiento (model_training_evaluation.py) NO usa ese CSV: importa
limpiar_base y construir_preprocesador directamente.
"""
 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
 
from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder
 
 
# ---------------------------------------------------------------
# CONFIGURACION (columnas por paso)
# ---------------------------------------------------------------
TARGET = "Pago_atiempo"
COLUMNAS_NO_PREDICTORAS = ["fecha_prestamo", TARGET]
 
COLUMNAS_OUTLIERS_NEGOCIO = ["edad_cliente", "puntaje", "puntaje_datacredito"]
COLUMNAS_WINSORIZAR = ["salario_cliente", "total_otros_prestamos", "cant_creditosvigentes"]
COLUMNA_PROMEDIO_INGRESOS = "promedio_ingresos_datacredito"
COLUMNA_GRUPO_INGRESOS = "tipo_laboral"
COLUMNAS_SALDO = ["saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor"]
COLUMNAS_CATEGORICAS = ["tipo_laboral", "tipo_credito", "tendencia_ingresos"]
CATEGORIAS_TENDENCIA_VALIDAS = ["Estable", "Creciente", "Decreciente"]
 
 
# ---------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga la base de datos cruda."""
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df
 
 
# ---------------------------------------------------------------
# 2. REGLAS DE NEGOCIO (SIN ESTADO, SIN FUGA DE DATOS)
# ---------------------------------------------------------------
def limpiar_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas de negocio fila por fila. No calcula ninguna estadistica
    a partir de los datos, por eso es seguro aplicarla antes del split y
    tambien a un cliente nuevo en produccion.
 
    - Valores imposibles -> NaN: edad_cliente fuera de [18-100], puntaje
      fuera de [0-100], puntaje_datacredito fuera de [150-999]. Luego el
      pipeline los imputa con la mediana del TRAIN.
    - tendencia_ingresos solo admite 'Estable', 'Creciente', 'Decreciente'.
      Cualquier otro valor (por ejemplo numeros por un error de mapeo en
      el origen) se anula.
    - tipo_credito llega como codigo numerico pero es categorico: se
      convierte a texto para que el One-Hot lo trate como categoria.
    """
    df = df.copy()
 
    if "edad_cliente" in df.columns:
        df.loc[(df["edad_cliente"] < 18) | (df["edad_cliente"] > 100), "edad_cliente"] = np.nan
    if "puntaje" in df.columns:
        df.loc[(df["puntaje"] < 0) | (df["puntaje"] > 100), "puntaje"] = np.nan
    if "puntaje_datacredito" in df.columns:
        df.loc[(df["puntaje_datacredito"] < 150) | (df["puntaje_datacredito"] > 999), "puntaje_datacredito"] = np.nan
 
    if "tendencia_ingresos" in df.columns:
        # Se pasa a texto con NaN reales para poder comparar con las categorias validas.
        serie = df["tendencia_ingresos"].astype("object")
        es_corrupto = serie.notna() & ~serie.isin(CATEGORIAS_TENDENCIA_VALIDAS)
        serie = serie.mask(es_corrupto, np.nan)
        df["tendencia_ingresos"] = serie
 
    if "tipo_credito" in df.columns:
        df["tipo_credito"] = df["tipo_credito"].astype(str)
 
    return df
 
 
# ---------------------------------------------------------------
# 3. TRANSFORMADOR PROPIO: MEDIANA POR GRUPO
# ---------------------------------------------------------------
class ImputadorMedianaPorGrupo(BaseEstimator, TransformerMixin):
    """
    Imputa los nulos de `variable` con la mediana calculada dentro de cada
    grupo de `grupo` (por ejemplo, tipo_laboral). feature-engine no trae
    imputacion agrupada, por eso se escribe este transformador con la
    interfaz fit/transform de scikit-learn.
 
    fit aprende las medianas SOLO con los datos que recibe (el train);
    transform las aplica. Si aparece un grupo desconocido o nulo, usa la
    mediana global del train.
    """
 
    def __init__(self, variable: str, grupo: str):
        self.variable = variable
        self.grupo = grupo
 
    def fit(self, X: pd.DataFrame, y=None):
        self.medianas_ = X.groupby(self.grupo)[self.variable].median().to_dict()
        self.mediana_global_ = X[self.variable].median()
        return self
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        relleno = X[self.grupo].map(self.medianas_).fillna(self.mediana_global_)
        X[self.variable] = X[self.variable].fillna(relleno)
        return X
 
 
# ---------------------------------------------------------------
# 4. PIPELINE DE PREPROCESAMIENTO (SE AJUSTA SOLO CON TRAIN)
# ---------------------------------------------------------------
def _presentes(columnas_deseadas: list, columnas_df) -> list:
    """Devuelve solo las columnas que existen en el dataframe."""
    if columnas_df is None:
        return list(columnas_deseadas)
    return [c for c in columnas_deseadas if c in columnas_df]
 
 
def construir_preprocesador(columnas=None) -> Pipeline:
    """
    Construye el pipeline de feature engineering (sin ajustar).
    `columnas` (opcional) es la lista de columnas de X; si se entrega, los
    pasos solo usan las que existan.
    """
    out_neg = _presentes(COLUMNAS_OUTLIERS_NEGOCIO, columnas)
    wins = _presentes(COLUMNAS_WINSORIZAR, columnas)
    saldos = _presentes(COLUMNAS_SALDO, columnas)
    cats = _presentes(COLUMNAS_CATEGORICAS, columnas)
    cats_sin_dato = _presentes(["tendencia_ingresos", "tipo_laboral"], columnas)
 
    pasos = [
        ("mediana_outliers_negocio", MeanMedianImputer(imputation_method="median", variables=out_neg)),
        ("winsorizacion", Winsorizer(capping_method="quantiles", tail="right", fold=0.01,
                                      variables=wins, missing_values="ignore")),
        ("categoricas_sin_dato", CategoricalImputer(imputation_method="missing",
                                                     fill_value="Sin_dato",
                                                     variables=cats_sin_dato)),
        ("flag_promedio_ingresos", AddMissingIndicator(variables=[COLUMNA_PROMEDIO_INGRESOS],
                                                        missing_only=False)),
        ("promedio_ingresos_por_grupo", ImputadorMedianaPorGrupo(COLUMNA_PROMEDIO_INGRESOS,
                                                                  COLUMNA_GRUPO_INGRESOS)),
        ("saldos_cero", ArbitraryNumberImputer(arbitrary_number=0, variables=saldos)),
        ("one_hot", OneHotEncoder(variables=cats, drop_last=True)),
    ]
    return Pipeline(pasos)
 
 
# ---------------------------------------------------------------
# 5. EJECUCION COMO SCRIPT (SOLO INSPECCION / EDA)
# ---------------------------------------------------------------
def ejecutar_feature_engineering(ruta_entrada: str, ruta_salida: str) -> pd.DataFrame:
    """
    Aplica limpieza + pipeline a TODA la base y la guarda en un CSV.
    Es solo para inspeccionar el resultado (por ejemplo en notebooks).
    OJO: aqui el pipeline se ajusta con todos los datos, por eso este CSV
    NO debe usarse para entrenar ni evaluar modelos.
    """
    print("=" * 60)
    print("INICIANDO FEATURE ENGINEERING (SOLO INSPECCION)")
    print("=" * 60)
 
    df = limpiar_base(cargar_datos(ruta_entrada))
    X = df.drop(columns=[c for c in COLUMNAS_NO_PREDICTORAS if c in df.columns])
 
    preprocesador = construir_preprocesador(X.columns)
    X_transformado = preprocesador.fit_transform(X)
 
    salida = X_transformado.copy()
    for col in COLUMNAS_NO_PREDICTORAS:
        if col in df.columns:
            salida[col] = df[col].values
 
    salida.to_csv(ruta_salida, index=False)
    print(f"Base transformada guardada en: {ruta_salida}")
    print(f"Shape final: {salida.shape}")
    print("=" * 60)
    return salida
 
 
if __name__ == "__main__":
    ejecutar_feature_engineering(
        ruta_entrada="../../Base_de_datos.csv",
        ruta_salida="../../Base_de_datos_transformada.csv",
    )