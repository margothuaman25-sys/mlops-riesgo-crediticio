"""
model_training_evaluation.py
------------------------------
Entrenamiento y evaluacion de modelos supervisados para predecir
Pago_atiempo (1 = paga a tiempo, 0 = no paga a tiempo).
 
Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autora: Margot Huaman Quispe
 
Modelos comparados:
    1. Regresion Logistica (linea base, simple e interpretable)
    2. XGBoost (suele rendir mejor con relaciones no lineales)
 
Decisiones de diseno:
- La variable objetivo esta desbalanceada (~95% paga / ~5% no paga). El
  profesor del curso confirmo que no es obligatorio tratar el desbalance,
  pero se usa una tecnica simple que no modifica los datos:
  class_weight='balanced' en la regresion logistica y scale_pos_weight en
  XGBoost.
- El feature engineering va DENTRO de un Pipeline de scikit-learn y se
  ajusta solo con el conjunto de entrenamiento (sin fuga de datos). El
  artefacto guardado incluye preprocesamiento + modelo, listo para que
  model_deploy.py reciba un cliente nuevo y devuelva la prediccion.
- El negocio se juega en la CLASE 0 (clientes que NO pagan a tiempo). Por
  eso las metricas se reportan para esa clase y el mejor modelo se elige
  con el F1 de la clase 0 (promedio de validacion cruzada). El F1 de la
  clase 1 sale ~0.99 casi con cualquier modelo porque esa clase es el 95%
  de los datos, asi que no sirve para comparar.
"""
 
import pickle
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
 
from ft_engineering import (
    COLUMNAS_NO_PREDICTORAS,
    TARGET,
    construir_preprocesador,
    limpiar_base,
)
 
SEMILLA = 42
 
 
# ---------------------------------------------------------------
# 1. CARGA Y PREPARACION DE DATOS
# ---------------------------------------------------------------
def cargar_datos(ruta_csv: str):
    """
    Carga la base CRUDA, aplica solo las reglas de negocio (limpiar_base,
    que no aprende nada de los datos) y separa X (predictoras) de y.
 
    fecha_prestamo se excluye del entrenamiento: es una fecha de
    originacion del credito y no se repite en produccion, asi que
    incluirla haria que el modelo aprendiera fechas especificas en vez de
    patrones de riesgo reales.
    """
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
 
    df = limpiar_base(df)
    X = df.drop(columns=[c for c in COLUMNAS_NO_PREDICTORAS if c in df.columns])
    y = df[TARGET]
 
    print(f"Variables predictoras (antes del preprocesamiento): {X.shape[1]}")
    print(f"Distribucion de la variable objetivo (%):\n{(y.value_counts(normalize=True) * 100).round(2)}")
    return X, y
 
 
def dividir_datos(X, y, test_size: float = 0.2, random_state: int = SEMILLA):
    """
    Separa en train/test de forma ESTRATIFICADA (stratify=y): ambos
    conjuntos mantienen la misma proporcion de clases (95/5). El
    preprocesamiento se ajusta DESPUES de este split, solo con train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    print(f"Clientes que NO pagan (clase 0) en train: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    print(f"Clientes que NO pagan (clase 0) en test:  {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")
    return X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------
# 2. CONSTRUCCION DE MODELOS (PIPELINES COMPLETOS)
# ---------------------------------------------------------------
def construir_regresion_logistica(columnas) -> Pipeline:
    """
    Preprocesamiento + escalado + regresion logistica. Se escala porque
    este modelo es sensible a la escala. class_weight='balanced' le da
    mas peso a la clase minoritaria sin sobremuestrear.
    """
    return Pipeline([
        ("preprocesador", construir_preprocesador(columnas)),
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(class_weight="balanced", max_iter=1000,
                                            random_state=SEMILLA)),
    ])
 
 
def construir_xgboost(y_train, columnas) -> Pipeline:
    """
    Preprocesamiento + XGBoost (no necesita escalado).
 
    scale_pos_weight = negativos / positivos. En XGBoost la clase
    "positiva" es la etiqueta 1, que aqui es la MAYORITARIA (paga a
    tiempo): el valor ~0.05 reduce su peso para equilibrar ambas clases.
    """
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"XGBoost: scale_pos_weight = {ratio:.3f}")
    return Pipeline([
        ("preprocesador", construir_preprocesador(columnas)),
        ("clasificador", XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss",
                                       random_state=SEMILLA)),
    ])
 
 
# ---------------------------------------------------------------
# 3. VALIDACION CRUZADA (SOLO SOBRE TRAIN)
# ---------------------------------------------------------------
def validar_cruzado(pipeline: Pipeline, X_train, y_train, nombre: str, n_folds: int = 5) -> dict:
    """
    Validacion cruzada estratificada. En cada fold el pipeline completo
    (incluido el preprocesamiento) se reajusta solo con los datos de
    entrenamiento de ese fold. Con solo ~100 casos de la clase 0 en el
    test, una sola particion es inestable; el promedio de varios folds es
    una estimacion mas confiable.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    scoring = {
        "precision_clase0": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_clase0": make_scorer(recall_score, pos_label=0),
        "f1_clase0": make_scorer(f1_score, pos_label=0),
        "roc_auc": "roc_auc",
    }
    cv = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=None)
 
    resumen = {"modelo": nombre}
    print(f"\n--- Validacion cruzada ({n_folds} folds): {nombre} ---")
    for clave in scoring:
        valores = cv[f"test_{clave}"]
        resumen[f"cv_{clave}"] = valores.mean()
        resumen[f"cv_{clave}_std"] = valores.std()
        print(f"{clave:18s}: {valores.mean():.4f} +/- {valores.std():.4f}")
    return resumen
 
 
# ---------------------------------------------------------------
# 4. EVALUACION EN TEST (metricas por clase)
# ---------------------------------------------------------------
def evaluar_modelo(pipeline: Pipeline, X_test, y_test, nombre: str) -> dict:
    """
    Evalua en el conjunto de test con foco en la clase 0 (no paga).
 
    - precision clase 0: de los clientes marcados como "no pagara",
      cuantos realmente no pagaron.
    - recall clase 0: de los que realmente no pagaron, cuantos detecto el
      modelo. Los que se escapan son creditos otorgados a quien no paga.
    - PR-AUC clase 0: resumen de precision/recall en todos los umbrales,
      muy usado cuando la clase de interes es rara.
    """
    y_pred = pipeline.predict(X_test)
    prob_paga = pipeline.predict_proba(X_test)[:, 1]
    y_es_mora = (y_test == 0).astype(int)
 
    metricas = {
        "modelo": nombre,
        "precision_clase0": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall_clase0": recall_score(y_test, y_pred, pos_label=0),
        "f1_clase0": f1_score(y_test, y_pred, pos_label=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_clase1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, prob_paga),
        "pr_auc_clase0": average_precision_score(y_es_mora, 1 - prob_paga),
    }
 
    print(f"\n--- Resultados en TEST: {nombre} ---")
    print("Clase 0 (NO paga a tiempo, la que importa al negocio):")
    print(f"  Precision: {metricas['precision_clase0']:.4f}")
    print(f"  Recall:    {metricas['recall_clase0']:.4f}")
    print(f"  F1:        {metricas['f1_clase0']:.4f}")
    print(f"  PR-AUC:    {metricas['pr_auc_clase0']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f} | F1 clase 1 (referencia): {metricas['f1_clase1']:.4f} | ROC-AUC: {metricas['roc_auc']:.4f}")
 
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMatriz de confusion (filas = real, columnas = prediccion):")
    print(pd.DataFrame(matriz, index=["real 0 (no paga)", "real 1 (paga)"],
                       columns=["pred 0", "pred 1"]))
    print(f"\nReporte de clasificacion:\n{classification_report(y_test, y_pred, digits=3)}")
    return metricas
 
 
# ---------------------------------------------------------------
# 5. SELECCION DEL MEJOR MODELO
# ---------------------------------------------------------------
def seleccionar_mejor_modelo(resumenes_cv: list, metrica: str = "cv_f1_clase0") -> str:
    """
    Elige el modelo con mayor F1 de la clase 0 en validacion cruzada
    (calculada sobre train, sin mirar el test). F1 equilibra los dos
    errores del negocio:
    - falso negativo (decir "paga" cuando NO paga): se presta dinero que
      no se recupera -> baja el recall de la clase 0;
    - falso positivo (decir "no paga" cuando SI paga): se rechaza a un
      buen cliente -> baja la precision de la clase 0.
    """
    mejor = max(resumenes_cv, key=lambda r: r[metrica])
    print(f"\n{'=' * 60}")
    print(f"MEJOR MODELO SEGUN {metrica}: {mejor['modelo']} ({mejor[metrica]:.4f})")
    print(f"{'=' * 60}")
    return mejor["modelo"]
 
 
# ---------------------------------------------------------------
# 6. VARIABLES MAS IMPORTANTES (REVISION DE POSIBLE FUGA)
# ---------------------------------------------------------------
def mostrar_importancias(pipeline_xgb: Pipeline, X_muestra, top: int = 10):
    """
    Muestra las variables que mas pesan en XGBoost. Si una sola variable
    domina de forma extrema, conviene revisar que no contenga informacion
    que solo se conoce DESPUES de otorgar el credito.
    """
    X_proc = pipeline_xgb[:-1].transform(X_muestra)
    importancias = pd.Series(pipeline_xgb[-1].feature_importances_, index=X_proc.columns)
    print(f"\nTop {top} variables mas importantes (XGBoost):")
    print(importancias.sort_values(ascending=False).head(top).round(4).to_string())
 
 
# ---------------------------------------------------------------
# 7. GUARDADO DEL MODELO GANADOR
# ---------------------------------------------------------------
def guardar_modelo(pipeline: Pipeline, nombre: str, columnas_entrada: list,
                   metricas_test: dict, ruta_salida: str):
    """
    Guarda en un .pkl el pipeline completo (preprocesamiento + modelo) y
    metadatos. model_deploy.py debe:
      1. aplicar limpiar_base (ft_engineering.py) al cliente nuevo,
      2. quedarse con 'columnas_entrada',
      3. llamar a pipeline.predict_proba(...).
    Para cargar el .pkl, la carpeta src debe estar en el PYTHONPATH porque
    el pipeline usa una clase de ft_engineering.py.
    """
    artefacto = {
        "pipeline": pipeline,
        "nombre_modelo": nombre,
        "columnas_entrada": columnas_entrada,
        "metricas_test": metricas_test,
        "target": TARGET,
        "significado_target": "1 = paga a tiempo, 0 = no paga a tiempo",
    }
    with open(ruta_salida, "wb") as f:
        pickle.dump(artefacto, f)
    print(f"\nModelo guardado en: {ruta_salida}")
 
 
# ---------------------------------------------------------------
# 8. PIPELINE COMPLETO
# ---------------------------------------------------------------
def ejecutar_entrenamiento_evaluacion(ruta_datos: str, ruta_modelo_salida: str):
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO Y EVALUACION DE MODELOS")
    print("=" * 60)
 
    X, y = cargar_datos(ruta_datos)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    columnas = list(X.columns)
 
    modelos = {
        "Regresion Logistica": construir_regresion_logistica(columnas),
        "XGBoost": construir_xgboost(y_train, columnas),
    }
 
    # a) Validacion cruzada en train -> criterio de seleccion
    resumenes_cv = [validar_cruzado(pipe, X_train, y_train, nombre) for nombre, pipe in modelos.items()]
    nombre_ganador = seleccionar_mejor_modelo(resumenes_cv)
 
    # b) Ajuste final de cada modelo con TODO el train y evaluacion en test
    resultados_test = {}
    for nombre, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        resultados_test[nombre] = evaluar_modelo(pipe, X_test, y_test, nombre)
 
    mostrar_importancias(modelos["XGBoost"], X_test)
 
    guardar_modelo(modelos[nombre_ganador], nombre_ganador, columnas,
                   resultados_test[nombre_ganador], ruta_modelo_salida)
 
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO Y EVALUACION COMPLETADOS")
    print("=" * 60)
    return modelos, resultados_test, resumenes_cv
 
 
if __name__ == "__main__":
    ejecutar_entrenamiento_evaluacion(
        ruta_datos="../../Base_de_datos.csv",
        ruta_modelo_salida="../../modelo_riesgo_crediticio.pkl",
    )
 