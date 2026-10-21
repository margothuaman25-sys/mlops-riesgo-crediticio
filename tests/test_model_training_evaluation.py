"""Pruebas de model_training_evaluation.py (con datos sinteticos)."""
import pickle

import pandas as pd

from model_training_evaluation import (
    cargar_datos,
    construir_regresion_logistica,
    construir_xgboost,
    dividir_datos,
    ejecutar_entrenamiento_evaluacion,
    evaluar_modelo,
    guardar_modelo,
    mostrar_importancias,
    seleccionar_mejor_modelo,
    validar_cruzado,
)


def test_cargar_datos_separa_predictoras_y_objetivo(ruta_csv):
    X, y = cargar_datos(str(ruta_csv))
    assert "Pago_atiempo" not in X.columns and "fecha_prestamo" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)


def test_dividir_datos_es_estratificado(ruta_csv):
    X, y = cargar_datos(str(ruta_csv))
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_validar_cruzado_devuelve_metricas_de_la_clase_0(ruta_csv):
    X, y = cargar_datos(str(ruta_csv))
    pipeline = construir_regresion_logistica(list(X.columns))
    resumen = validar_cruzado(pipeline, X, y, "Regresion Logistica", n_folds=3)
    assert resumen["modelo"] == "Regresion Logistica"
    assert 0 <= resumen["cv_f1_clase0"] <= 1 and "cv_recall_clase0_std" in resumen


def test_evaluar_modelo_calcula_metricas_y_matriz(ruta_csv, capsys):
    X, y = cargar_datos(str(ruta_csv))
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    pipeline = construir_regresion_logistica(list(X.columns)).fit(X_train, y_train)
    metricas = evaluar_modelo(pipeline, X_test, y_test, "LR")
    for clave in ("precision_clase0", "recall_clase0", "f1_clase0", "roc_auc", "pr_auc_clase0"):
        assert 0 <= metricas[clave] <= 1
    assert metricas["roc_auc"] > 0.8            # puntaje separa las clases en la base sintetica
    assert "Matriz de confusion" in capsys.readouterr().out


def test_seleccionar_mejor_modelo_usa_el_f1_de_la_clase_0():
    resumenes = [{"modelo": "A", "cv_f1_clase0": 0.60}, {"modelo": "B", "cv_f1_clase0": 0.80}]
    assert seleccionar_mejor_modelo(resumenes) == "B"


def test_xgboost_e_importancias(ruta_csv, capsys):
    X, y = cargar_datos(str(ruta_csv))
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    modelo = construir_xgboost(y_train, list(X.columns)).fit(X_train, y_train)
    mostrar_importancias(modelo, X_test, top=5)
    assert "Top 5 variables" in capsys.readouterr().out
    assert modelo.predict_proba(X_test).shape == (len(X_test), 2)


def test_guardar_modelo_y_recargarlo(tmp_path, ruta_csv):
    X, y = cargar_datos(str(ruta_csv))
    pipeline = construir_regresion_logistica(list(X.columns)).fit(X, y)
    ruta = tmp_path / "modelo.pkl"
    guardar_modelo(pipeline, "LR", list(X.columns), {"f1_clase0": 0.5}, str(ruta))
    with open(ruta, "rb") as f:
        artefacto = pickle.load(f)
    assert artefacto["nombre_modelo"] == "LR" and artefacto["columnas_entrada"] == list(X.columns)
    assert artefacto["pipeline"].predict_proba(X.head(3)).shape == (3, 2)


def test_flujo_completo_de_entrenamiento(tmp_path, ruta_csv):
    salida = tmp_path / "modelo_final.pkl"
    modelos, resultados, resumenes_cv = ejecutar_entrenamiento_evaluacion(str(ruta_csv), str(salida))
    assert salida.exists()
    assert set(modelos) == {"Regresion Logistica", "XGBoost"} == set(resultados)
    assert len(resumenes_cv) == 2
    with open(salida, "rb") as f:
        assert pickle.load(f)["nombre_modelo"] in modelos
    assert isinstance(pd.DataFrame(resultados).T, pd.DataFrame)
