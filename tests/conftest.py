"""
Configuracion comun de las pruebas.

Las pruebas NO dependen de la base real ni del .pkl del repositorio: usan una
base sintetica con las mismas columnas que Base_de_datos.csv, asi corren igual
en cualquier maquina y en GitHub Actions.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "mlops_pipeline" / "src"
sys.path.insert(0, str(SRC))

from ft_engineering import COLUMNAS_NO_PREDICTORAS, TARGET, limpiar_base  # noqa: E402
from model_training_evaluation import construir_xgboost  # noqa: E402


def construir_base_sintetica(n: int = 800, semilla: int = 0) -> pd.DataFrame:
    """Base con las columnas del proyecto; `puntaje` separa las clases (como en el dataset real)."""
    rng = np.random.default_rng(semilla)
    y = (rng.random(n) > 0.12).astype(int)
    puntaje = np.where(y == 1, 95.227787, rng.uniform(5, 60, n))
    puntaje[(y == 0) & (rng.random(n) < 0.25)] = -10.0     # negativos, como en el dataset real
    capital = rng.lognormal(14.5, 0.6, n)
    plazo = rng.integers(3, 36, n)

    tendencia = rng.choice(["Creciente", "Estable", "Decreciente"], n).astype(object)
    tendencia[rng.random(n) < 0.27] = np.nan
    tendencia[:3] = ["8315", "0", "1000000"]                 # valores corruptos
    promedio_ingresos = rng.lognormal(14, 0.7, n)
    promedio_ingresos[rng.random(n) < 0.27] = np.nan
    saldo_codeudor = np.zeros(n)
    saldo_codeudor[rng.random(n) < 0.05] = np.nan

    return pd.DataFrame({
        "tipo_credito": rng.choice([4, 9, 7, 6, 10], n),
        "fecha_prestamo": pd.date_range("2025-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M:%S"),
        "capital_prestado": capital,
        "plazo_meses": plazo,
        "edad_cliente": rng.integers(20, 70, n),
        "tipo_laboral": rng.choice(["Empleado", "Independiente"], n, p=[0.63, 0.37]),
        "salario_cliente": rng.lognormal(15, 0.5, n),
        "total_otros_prestamos": rng.lognormal(13.5, 0.8, n),
        "cuota_pactada": capital / plazo,
        "puntaje": puntaje,
        "puntaje_datacredito": rng.normal(780, 60, n).round(),
        "cant_creditosvigentes": rng.poisson(5, n),
        "huella_consulta": rng.poisson(4, n),
        "saldo_mora": np.where(rng.random(n) < 0.02, rng.uniform(100, 5000, n), 0.0),
        "saldo_total": rng.lognormal(9.5, 1.0, n),
        "saldo_principal": rng.lognormal(9.3, 1.0, n),
        "saldo_mora_codeudor": saldo_codeudor,
        "creditos_sectorFinanciero": rng.poisson(2.7, n),
        "creditos_sectorCooperativo": rng.poisson(0.3, n),
        "creditos_sectorReal": rng.poisson(1.3, n),
        "promedio_ingresos_datacredito": promedio_ingresos,
        "tendencia_ingresos": tendencia,
        "Pago_atiempo": y,
    })


@pytest.fixture(scope="session")
def base_sintetica() -> pd.DataFrame:
    return construir_base_sintetica()


@pytest.fixture()
def ruta_csv(tmp_path, base_sintetica) -> Path:
    ruta = tmp_path / "base.csv"
    base_sintetica.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def artefacto_entrenado(base_sintetica) -> dict:
    """Artefacto con la misma forma que guarda model_training_evaluation.py (XGBoost, el modelo ganador del proyecto)."""
    df = limpiar_base(base_sintetica)
    X = df.drop(columns=[c for c in COLUMNAS_NO_PREDICTORAS if c in df.columns])
    y = df[TARGET]
    columnas = list(X.columns)
    pipeline = construir_xgboost(y, columnas).fit(X, y)
    return {
        "pipeline": pipeline,
        "nombre_modelo": "XGBoost (prueba)",
        "columnas_entrada": columnas,
        "metricas_test": {"f1_clase0": 0.5, "modelo": "prueba"},
        "target": TARGET,
        "significado_target": "1 = paga a tiempo, 0 = no paga a tiempo",
    }


@pytest.fixture()
def ruta_pkl(tmp_path, artefacto_entrenado) -> Path:
    ruta = tmp_path / "modelo.pkl"
    with open(ruta, "wb") as f:
        pickle.dump(artefacto_entrenado, f)
    return ruta
