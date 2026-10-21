"""Pruebas de model_monitoring.py: PSI, pruebas estadisticas, ventanas y reporte."""
import numpy as np
import pandas as pd
import pytest

import model_monitoring as mm


def test_psi_categorico_identico_es_cero_y_distinto_es_alto():
    ref = ["a"] * 50 + ["b"] * 50
    assert mm.psi_categorico(ref, ref) == pytest.approx(0, abs=1e-9)
    assert mm.psi_categorico(ref, ["a"] * 95 + ["b"] * 5) > mm.UMBRAL_PSI_DRIFT


def test_psi_con_series_vacias_devuelve_nan():
    assert np.isnan(mm.psi_categorico([], ["a"]))
    assert np.isnan(mm.psi_numerico([], [1.0]))
    estadistico, p_valor = mm.prueba_chi2([], ["a"])
    assert np.isnan(estadistico) and np.isnan(p_valor)


def test_psi_numerico_detecta_desplazamientos():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    assert mm.psi_numerico(ref, rng.normal(0, 1, 2000)) < mm.UMBRAL_PSI_ALERTA
    assert mm.psi_numerico(ref, rng.normal(3, 1, 2000)) > mm.UMBRAL_PSI_DRIFT


def test_psi_numerico_variable_casi_constante_usa_el_calculo_categorico():
    assert mm.psi_numerico([5.0] * 50, [5.0] * 50) == pytest.approx(0, abs=1e-9)


def test_prueba_chi2():
    estadistico, p_valor = mm.prueba_chi2(["a", "b"] * 50, ["a", "b"] * 50)
    assert estadistico == pytest.approx(0, abs=1e-9) and p_valor > 0.99
    assert mm.prueba_chi2(["a"] * 10, ["a"] * 10) == (0.0, 1.0)        # una sola categoria
    assert mm.prueba_chi2(["a"] * 100, ["b"] * 100)[1] < 0.001         # categorias distintas


@pytest.mark.parametrize("psi, esperado", [
    (0.02, "Estable"), (0.15, "Alerta"), (0.40, "Drift"), (np.nan, "Sin datos"), (None, "Sin datos"),
])
def test_clasificar_psi(psi, esperado):
    assert mm.clasificar_psi(psi) == esperado


def test_es_numerica_continua():
    assert mm.es_numerica_continua(pd.Series(range(50)))
    assert not mm.es_numerica_continua(pd.Series([0, 1, 2] * 20))
    assert not mm.es_numerica_continua(pd.Series(["a", "b"] * 20))


def test_dividir_por_fecha_ordena_y_respeta_la_proporcion():
    df = pd.DataFrame({"fecha_prestamo": ["2025-03-01", "2025-01-01", "2025-02-01", "2025-04-01"] * 5, "x": range(20)})
    ref, act = mm.dividir_por_fecha(df, 0.70)
    assert len(ref) == 14 and len(act) == 6
    assert pd.to_datetime(ref["fecha_prestamo"]).max() <= pd.to_datetime(act["fecha_prestamo"]).min()


def test_dividir_por_fecha_ilegible_conserva_el_orden(capsys):
    df = pd.DataFrame({"fecha_prestamo": ["no es fecha"] * 10, "x": range(10)})
    ref, act = mm.dividir_por_fecha(df, 0.70)
    assert ref["x"].tolist() == list(range(7)) and "Aviso" in capsys.readouterr().out
    assert len(mm.dividir_por_fecha(pd.DataFrame({"x": range(10)}))[0]) == 7   # sin columna de fecha


def test_detectar_drift_marca_la_variable_desplazada(base_sintetica):
    ref = base_sintetica.iloc[:500].reset_index(drop=True)
    act = base_sintetica.iloc[500:].reset_index(drop=True).copy()
    act["capital_prestado"] = act["capital_prestado"] * 5
    tabla = mm.detectar_drift(ref, act, ["capital_prestado", "tipo_laboral", "plazo_meses"])
    assert tabla.iloc[0]["variable"] == "capital_prestado" and tabla.iloc[0]["estado"] == "Drift"
    assert set(tabla["prueba"]) == {"KS", "Chi2"}
    assert tabla["psi"].is_monotonic_decreasing


def test_cargar_base_y_cargar_artefacto(ruta_csv, ruta_pkl, base_sintetica):
    assert mm.cargar_base(ruta_csv).shape == base_sintetica.shape
    assert mm.cargar_artefacto(ruta_pkl)["nombre_modelo"].startswith("XGBoost")


def test_puntaje_riesgo_y_drift_de_predicciones(artefacto_entrenado, base_sintetica):
    puntajes = mm.puntaje_riesgo(artefacto_entrenado, base_sintetica)
    assert len(puntajes) == len(base_sintetica) and ((puntajes >= 0) & (puntajes <= 1)).all()
    ref, act = base_sintetica.iloc[:560], base_sintetica.iloc[560:]
    resumen, s_ref, s_act = mm.drift_predicciones(artefacto_entrenado, ref, act)
    assert resumen["estado"] in {"Estable", "Alerta", "Drift"}
    assert len(s_ref) == 560 and len(s_act) == 240


def test_ejecutar_monitoreo_cubre_todas_las_columnas_de_entrada(artefacto_entrenado, base_sintetica):
    ref, act = mm.dividir_por_fecha(base_sintetica, 0.70)
    tabla, resumen, _, _ = mm.ejecutar_monitoreo(ref, act, artefacto_entrenado)
    assert len(tabla) == len(artefacto_entrenado["columnas_entrada"])
    assert "psi_prediccion" in resumen


def test_main_imprime_el_reporte(monkeypatch, capsys, base_sintetica, artefacto_entrenado):
    monkeypatch.setattr(mm, "cargar_base", lambda: base_sintetica)
    monkeypatch.setattr(mm, "cargar_artefacto", lambda: artefacto_entrenado)
    mm.main()
    salida = capsys.readouterr().out
    assert "MONITOREO DE DATA DRIFT" in salida and "Resumen:" in salida
    assert "PSI de la probabilidad de no pago" in salida


def test_main_avisa_cuando_hay_drift(monkeypatch, capsys, base_sintetica, artefacto_entrenado):
    desplazada = base_sintetica.copy()
    corte = int(len(desplazada) * 0.70)
    desplazada.loc[desplazada.index[corte:], "capital_prestado"] *= 10
    monkeypatch.setattr(mm, "cargar_base", lambda: desplazada)
    monkeypatch.setattr(mm, "cargar_artefacto", lambda: artefacto_entrenado)
    mm.main()
    assert "ATENCION" in capsys.readouterr().out


def test_color_estado_y_deteccion_de_streamlit():
    assert mm._color_estado("Drift") != "" and mm._color_estado("otro") == ""
    assert mm._corriendo_en_streamlit() is False
