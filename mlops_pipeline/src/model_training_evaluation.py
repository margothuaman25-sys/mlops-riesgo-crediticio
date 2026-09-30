"""
model_training_evaluation.py
------------------------------
Entrenamiento y evaluacion de modelos supervisados para predecir
Pago_atiempo (1 = paga a tiempo, 0 = no paga a tiempo).

Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autor: Shekina

Modelos comparados:
    1. Regresion Logistica (linea base, simple e interpretable)
    2. XGBoost (suele rendir mejor con relaciones no lineales)

Ambos usan class_weight='balanced' (o su equivalente en XGBoost:
scale_pos_weight) porque la variable objetivo esta desbalanceada
(~95% / 5%). El profesor del curso confirmo que no es obligatorio
tratar el desbalance para la entrega (el dataset es sintetico), pero
se deja esta tecnica simple porque mejora la deteccion de la clase
minoritaria sin modificar los datos.

Metricas usadas: precision, recall, F1 y ROC-AUC (accuracy no es
confiable aqui por el desbalance de clases).
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------
# 1. CARGA Y PREPARACION DE DATOS
# ---------------------------------------------------------------
def cargar_datos(ruta_csv: str):
    """
    Carga la base tipada (ya con feature engineering aplicado) y separa
    variables predictoras (X) de la variable objetivo (y).

    fecha_prestamo se excluye del entrenamiento: es una fecha de
    originacion del credito y no se repite en produccion, asi que
    incluirla haria que el modelo generalice mal (aprenderia fechas
    especificas en vez de patrones de riesgo reales).
    """
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    columnas_excluir = ['fecha_prestamo', 'Pago_atiempo']
    X = df.drop(columns=[c for c in columnas_excluir if c in df.columns])
    y = df['Pago_atiempo']

    print(f"Variables predictoras: {X.shape[1]}")
    print(f"Distribucion de la variable objetivo:\n{y.value_counts(normalize=True).round(4) * 100}")
    return X, y


def dividir_datos(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Separa en train/test de forma ESTRATIFICADA (stratify=y), para que
    ambos conjuntos mantengan la misma proporcion de clases (95/5).
    Sin estratificar, el test set podria quedar con muy pocos o ningun
    caso de la clase minoritaria, arruinando la evaluacion.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    print(f"Proporcion clase minoritaria en train: {y_train.mean():.4f}")
    print(f"Proporcion clase minoritaria en test:  {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# 2. ENTRENAMIENTO DE MODELOS
# ---------------------------------------------------------------
def entrenar_regresion_logistica(X_train, y_train, scaler: StandardScaler):
    """
    Regresion Logistica como modelo base. Se escalan las variables
    numericas porque este modelo es sensible a la escala (a diferencia
    de los modelos basados en arboles como XGBoost).
    class_weight='balanced' le da mas peso a la clase minoritaria
    automaticamente, sin necesidad de sobremuestrear los datos.
    """
    X_train_scaled = scaler.fit_transform(X_train)
    modelo = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    modelo.fit(X_train_scaled, y_train)
    print("Regresion Logistica entrenada.")
    return modelo


def entrenar_xgboost(X_train, y_train):
    """
    XGBoost: modelo de arboles potenciados (boosting), no necesita
    escalado de variables. En vez de class_weight, XGBoost usa el
    parametro scale_pos_weight = (negativos / positivos) para lograr
    el mismo efecto de balanceo de clases.
    """
    ratio_desbalance = (y_train == 0).sum() / (y_train == 1).sum()
    modelo = XGBClassifier(
        scale_pos_weight=ratio_desbalance,
        eval_metric='logloss',
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    print(f"XGBoost entrenado (scale_pos_weight={ratio_desbalance:.2f}).")
    return modelo


# ---------------------------------------------------------------
# 3. EVALUACION DE MODELOS
# ---------------------------------------------------------------
def evaluar_modelo(modelo, X_test, y_test, nombre: str) -> dict:
    """
    Calcula precision, recall, F1 y ROC-AUC. No se usa accuracy como
    metrica principal porque con 95%/5% de desbalance, un modelo que
    siempre prediga "paga a tiempo" tendria 95% de accuracy sin ser
    util para detectar el riesgo real.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        'modelo': nombre,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- Resultados: {nombre} ---")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall:    {metricas['recall']:.4f}")
    print(f"F1-score:  {metricas['f1']:.4f}")
    print(f"ROC-AUC:   {metricas['roc_auc']:.4f}")
    print(f"\nMatriz de confusion:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nReporte de clasificacion:\n{classification_report(y_test, y_pred)}")

    return metricas


# ---------------------------------------------------------------
# 4. SELECCION DEL MEJOR MODELO
# ---------------------------------------------------------------
def seleccionar_mejor_modelo(resultados: list, metrica: str = 'f1') -> dict:
    """
    Selecciona el modelo con mejor desempeno segun la metrica indicada.
    Se usa F1 por defecto porque equilibra precision y recall, ambas
    importantes en riesgo crediticio: un falso negativo (decir que
    "si paga" cuando en realidad no) cuesta dinero a la empresa, y un
    falso positivo (rechazar a un buen cliente) cuesta oportunidad de
    negocio.
    """
    mejor = max(resultados, key=lambda r: r[metrica])
    print(f"\n{'='*60}")
    print(f"MEJOR MODELO SEGUN {metrica.upper()}: {mejor['modelo']} ({mejor[metrica]:.4f})")
    print(f"{'='*60}")
    return mejor


# ---------------------------------------------------------------
# 5. GUARDADO DEL MODELO GANADOR
# ---------------------------------------------------------------
def guardar_modelo(modelo, ruta_salida: str, scaler: StandardScaler = None):
    """
    Guarda el modelo entrenado (y el scaler, si aplica) en un archivo
    .pkl para que model_deploy.py pueda cargarlo despues sin tener
    que reentrenar.
    """
    objeto_guardar = {'modelo': modelo, 'scaler': scaler}
    with open(ruta_salida, 'wb') as f:
        pickle.dump(objeto_guardar, f)
    print(f"\nModelo guardado en: {ruta_salida}")


# ---------------------------------------------------------------
# 6. PIPELINE COMPLETO
# ---------------------------------------------------------------
def ejecutar_entrenamiento_evaluacion(ruta_datos: str, ruta_modelo_salida: str):
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO Y EVALUACION DE MODELOS")
    print("=" * 60)

    X, y = cargar_datos(ruta_datos)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    print("\n--- Entrenando Regresion Logistica ---")
    scaler = StandardScaler()
    modelo_lr = entrenar_regresion_logistica(X_train, y_train, scaler)
    X_test_scaled = scaler.transform(X_test)
    resultados_lr = evaluar_modelo(modelo_lr, X_test_scaled, y_test, "Regresion Logistica")

    print("\n--- Entrenando XGBoost ---")
    modelo_xgb = entrenar_xgboost(X_train, y_train)
    resultados_xgb = evaluar_modelo(modelo_xgb, X_test, y_test, "XGBoost")

    mejor = seleccionar_mejor_modelo([resultados_lr, resultados_xgb], metrica='f1')

    if mejor['modelo'] == "Regresion Logistica":
        guardar_modelo(modelo_lr, ruta_modelo_salida, scaler=scaler)
    else:
        guardar_modelo(modelo_xgb, ruta_modelo_salida, scaler=None)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO Y EVALUACION COMPLETADOS")
    print("=" * 60)

    return modelo_lr, modelo_xgb, resultados_lr, resultados_xgb


if __name__ == "__main__":
    ejecutar_entrenamiento_evaluacion(
        ruta_datos="../../Base_de_datos_tipada.csv",
        ruta_modelo_salida="../../modelo_riesgo_crediticio.pkl",
    )
