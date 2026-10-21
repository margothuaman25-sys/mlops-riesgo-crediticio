"""Pruebas de ft_engineering.py: reglas de negocio y pipeline de preprocesamiento."""
import numpy as np
import pandas as pd

from ft_engineering import (
    COLUMNAS_NO_PREDICTORAS,
    ImputadorMedianaPorGrupo,
    cargar_datos,
    construir_preprocesador,
    ejecutar_feature_engineering,
    limpiar_base,
)


def _entradas(base):
    return limpiar_base(base).drop(columns=COLUMNAS_NO_PREDICTORAS)


def test_limpiar_base_anula_valores_imposibles():
    df = pd.DataFrame({
        "edad_cliente": [17, 30, 101],
        "puntaje": [-5.0, 50.0, 120.0],
        "puntaje_datacredito": [100, 800, 1000],
        "tendencia_ingresos": ["Estable", "123", "Creciente"],
        "tipo_credito": [4, 9, 7],
    })
    limpio = limpiar_base(df)
    assert limpio["edad_cliente"].isna().tolist() == [True, False, True]
    assert limpio["puntaje"].isna().tolist() == [True, False, True]
    assert limpio["puntaje_datacredito"].isna().tolist() == [True, False, True]
    assert limpio["tendencia_ingresos"].isna().tolist() == [False, True, False]
    assert limpio["tipo_credito"].tolist() == ["4", "9", "7"]


def test_limpiar_base_no_modifica_el_original_y_tolera_columnas_ausentes():
    df = pd.DataFrame({"edad_cliente": [10, 40]})
    limpio = limpiar_base(df)
    assert df["edad_cliente"].tolist() == [10, 40]
    assert limpio["edad_cliente"].isna().tolist() == [True, False]
    assert list(limpiar_base(pd.DataFrame({"otra": [1]})).columns) == ["otra"]


def test_limpiar_base_conserva_los_valores_validos():
    df = pd.DataFrame({"edad_cliente": [18, 100], "puntaje": [0.0, 100.0], "puntaje_datacredito": [150, 999]})
    assert limpiar_base(df).notna().all().all()


def test_imputador_mediana_por_grupo():
    X = pd.DataFrame({
        "grupo": ["A", "A", "A", "B", "B"],
        "valor": [1.0, 3.0, np.nan, 10.0, np.nan],
    })
    imputador = ImputadorMedianaPorGrupo("valor", "grupo").fit(X)
    resultado = imputador.transform(X)
    assert resultado["valor"].tolist() == [1.0, 3.0, 2.0, 10.0, 10.0]
    nuevo = pd.DataFrame({"grupo": ["Z", None], "valor": [np.nan, np.nan]})
    assert imputador.transform(nuevo)["valor"].tolist() == [X["valor"].median()] * 2   # mediana global
    assert X["valor"].isna().sum() == 2                                              # no altera la entrada


def test_preprocesador_deja_todo_numerico_y_sin_nulos(base_sintetica):
    X = _entradas(base_sintetica)
    transformado = construir_preprocesador(X.columns).fit_transform(X)
    assert transformado.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in transformado.dtypes)
    assert any(c.endswith("_na") for c in transformado.columns)          # indicador de dato faltante
    assert any(c.startswith("tipo_laboral_") for c in transformado.columns)  # One-Hot


def test_preprocesador_ajustado_en_train_maneja_categorias_nuevas(base_sintetica):
    X = _entradas(base_sintetica)
    preprocesador = construir_preprocesador(X.columns).fit(X.iloc[:600])
    prueba = X.iloc[600:].copy()
    prueba.loc[prueba.index[0], "tipo_laboral"] = "Otro"
    transformado = preprocesador.transform(prueba)
    assert transformado.isna().sum().sum() == 0
    assert len(transformado) == len(prueba)


def test_winsorizacion_limita_los_valores_extremos(base_sintetica):
    X = _entradas(base_sintetica)
    X.loc[X.index[0], "salario_cliente"] = 22_000_000_000.0
    preprocesador = construir_preprocesador(X.columns).fit(X)
    assert preprocesador.transform(X)["salario_cliente"].max() < 22_000_000_000.0


def test_ejecutar_feature_engineering_genera_el_csv(tmp_path, ruta_csv):
    salida = tmp_path / "transformada.csv"
    resultado = ejecutar_feature_engineering(str(ruta_csv), str(salida))
    assert salida.exists()
    assert "Pago_atiempo" in resultado.columns and "fecha_prestamo" in resultado.columns
    assert pd.read_csv(salida).shape == resultado.shape


def test_cargar_datos(ruta_csv, base_sintetica):
    assert cargar_datos(str(ruta_csv)).shape == base_sintetica.shape
