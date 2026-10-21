"""Pruebas de la API (model_deploy.py) con un modelo entrenado sobre datos sinteticos."""
import pytest
from fastapi.testclient import TestClient

import model_deploy

CLIENTE_BUENO = {
    "tipo_credito": "4", "capital_prestado": 10_000_000, "plazo_meses": 36, "edad_cliente": 35,
    "tipo_laboral": "Empleado", "salario_cliente": 3_000_000, "total_otros_prestamos": 1_500_000,
    "cuota_pactada": 450_000, "puntaje": 95.227787, "puntaje_datacredito": 780,
    "cant_creditosvigentes": 2, "huella_consulta": 1, "saldo_mora": 0, "saldo_total": 5_000_000,
    "saldo_principal": 4_500_000, "promedio_ingresos_datacredito": 3_200_000, "tendencia_ingresos": "Estable",
}


@pytest.fixture()
def cliente_api(artefacto_entrenado):
    """TestClient sin lifespan: se inyecta el modelo de prueba directamente."""
    model_deploy.estado["artefacto"], model_deploy.estado["error"] = artefacto_entrenado, None
    yield TestClient(model_deploy.app)
    model_deploy.estado["artefacto"], model_deploy.estado["error"] = None, None


def test_raiz_y_health(cliente_api):
    assert cliente_api.get("/").json()["documentacion"] == "/docs"
    salud = cliente_api.get("/health").json()
    assert salud["estado"] == "ok" and salud["modelo_cargado"] is True


def test_info_del_modelo(cliente_api, artefacto_entrenado):
    info = cliente_api.get("/info").json()
    assert info["columnas_entrada"] == artefacto_entrenado["columnas_entrada"]
    assert info["modelo"] == artefacto_entrenado["nombre_modelo"]


def test_predict_distingue_buen_y_mal_puntaje(cliente_api):
    bueno = cliente_api.post("/predict", json=CLIENTE_BUENO).json()
    malo = cliente_api.post("/predict", json={**CLIENTE_BUENO, "puntaje": 20}).json()
    assert bueno["prediccion"] == 1 and bueno["decision"] == "Paga a tiempo"
    assert malo["prediccion"] == 0 and malo["probabilidad_no_pago"] > bueno["probabilidad_no_pago"]
    assert bueno["probabilidad_pago"] + bueno["probabilidad_no_pago"] == pytest.approx(1, abs=1e-5)
    assert bueno["umbral_riesgo"] == model_deploy.UMBRAL_RIESGO


def test_predict_lista_los_campos_no_informados(cliente_api, artefacto_entrenado):
    respuesta = cliente_api.post("/predict", json={})
    assert respuesta.status_code == 200
    faltantes = set(respuesta.json()["campos_no_informados"])
    # tipo_credito se convierte en "Sin_dato" antes de listar los campos faltantes
    assert set(artefacto_entrenado["columnas_entrada"]) - faltantes == {"tipo_credito"}


def test_predict_rechaza_campos_mal_escritos(cliente_api):
    assert cliente_api.post("/predict", json={"puntajee": 50}).status_code == 422


def test_predict_batch_ok_y_limites(cliente_api):
    respuesta = cliente_api.post("/predict_batch", json=[CLIENTE_BUENO, {**CLIENTE_BUENO, "puntaje": 20}])
    assert respuesta.status_code == 200 and len(respuesta.json()) == 2
    assert cliente_api.post("/predict_batch", json=[]).status_code == 400
    exceso = [{}] * (model_deploy.MAX_CLIENTES_POR_LOTE + 1)
    assert cliente_api.post("/predict_batch", json=exceso).status_code == 400


def test_sin_modelo_la_api_responde_503():
    model_deploy.estado["artefacto"], model_deploy.estado["error"] = None, "sin modelo"
    cliente = TestClient(model_deploy.app)
    assert cliente.get("/health").json()["estado"] == "error"
    assert cliente.get("/info").status_code == 503
    assert cliente.post("/predict", json={}).status_code == 503
    assert cliente.post("/predict_batch", json=[{}]).status_code == 503


def test_lifespan_carga_el_modelo_desde_el_pkl(monkeypatch, ruta_pkl):
    monkeypatch.setattr(model_deploy, "RUTA_MODELO", ruta_pkl)
    with TestClient(model_deploy.app) as cliente:
        assert cliente.get("/health").json()["modelo_cargado"] is True
    model_deploy.estado["artefacto"] = None


def test_lifespan_con_pkl_inexistente_informa_el_error(monkeypatch, tmp_path):
    monkeypatch.setattr(model_deploy, "RUTA_MODELO", tmp_path / "no_existe.pkl")
    with TestClient(model_deploy.app) as cliente:
        salud = cliente.get("/health").json()
        assert salud["estado"] == "error" and "FileNotFoundError" in salud["detalle"]


@pytest.mark.parametrize("valor, esperado", [(4, "4"), (4.0, "4"), ("4", "4"), (float("nan"), "Sin_dato"), (99, "99")])
def test_codigo_a_texto_normaliza_tipo_credito(valor, esperado):
    assert model_deploy._codigo_a_texto(valor, ["4", "9"]) == esperado
