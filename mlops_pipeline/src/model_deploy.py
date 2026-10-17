"""
model_deploy.py
----------------
API (FastAPI) para servir el modelo de riesgo crediticio.

Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autor: Margot Huaman Quispe

La API carga el artefacto guardado por model_training_evaluation.py
(modelo_riesgo_crediticio.pkl), que contiene el pipeline COMPLETO:
preprocesamiento + modelo. Recibe los datos de un cliente nuevo, aplica las
reglas de negocio de ft_engineering.limpiar_base y devuelve la probabilidad
de que NO pague a tiempo.

Endpoints:
    GET  /               datos basicos del servicio
    GET  /health         estado del servicio y del modelo
    GET  /info           modelo cargado, columnas de entrada y metricas de prueba
    POST /predict        prediccion para un cliente
    POST /predict_batch  prediccion para una lista de clientes (maximo 1000)

Uso local (desde mlops_pipeline/src):
    uvicorn model_deploy:app --reload
    Documentacion interactiva: http://127.0.0.1:8000/docs

Notas:
- Los campos que no se informen se tratan como datos faltantes y el pipeline
  los imputa igual que en el entrenamiento; la respuesta lista cuales no se
  informaron, porque la calidad de la prediccion baja si faltan muchos.
- El modelo depende casi por completo de `puntaje` (ver README, seccion 5.1).
  Si `puntaje` no se informa, la prediccion no es confiable.
"""

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import COLUMNAS_CATEGORICAS, limpiar_base  # noqa: E402

# ---------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_MODELO = Path(os.getenv("RUTA_MODELO", RAIZ_PROYECTO / "modelo_riesgo_crediticio.pkl"))
UMBRAL_RIESGO = 0.50          # prob. de no pago desde la cual se clasifica como "no paga"
MAX_CLIENTES_POR_LOTE = 1000

estado = {"artefacto": None, "error": None}


# ---------------------------------------------------------------
# ESQUEMAS DE ENTRADA Y SALIDA
# ---------------------------------------------------------------
class Cliente(BaseModel):
    """Datos de un cliente (las mismas columnas de Base_de_datos.csv, sin la fecha ni el objetivo)."""

    model_config = ConfigDict(
        extra="forbid",   # un nombre de campo mal escrito devuelve error 422 en vez de ignorarse
        json_schema_extra={"example": {
            "tipo_credito": "4",
            "capital_prestado": 10000000,
            "plazo_meses": 36,
            "edad_cliente": 35,
            "tipo_laboral": "Empleado",
            "salario_cliente": 3000000,
            "total_otros_prestamos": 1500000,
            "cuota_pactada": 450000,
            "puntaje": 95.227787,
            "puntaje_datacredito": 780,
            "cant_creditosvigentes": 2,
            "huella_consulta": 1,
            "creditos_sectorFinanciero": 1,
            "creditos_sectorCooperativo": 0,
            "creditos_sectorReal": 1,
            "saldo_mora": 0,
            "saldo_total": 5000000,
            "saldo_principal": 4500000,
            "saldo_mora_codeudor": 0,
            "promedio_ingresos_datacredito": 3200000,
            "tendencia_ingresos": "Estable",
        }},
    )

    tipo_credito: Union[str, int, None] = Field(None, description="Codigo del tipo de credito (categorico).")
    capital_prestado: Optional[float] = None
    plazo_meses: Optional[float] = None
    edad_cliente: Optional[float] = Field(None, description="Valido entre 18 y 100; fuera de rango se imputa.")
    tipo_laboral: Optional[str] = Field(None, description="Por ejemplo: Empleado o Independiente.")
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: Optional[float] = None
    puntaje: Optional[float] = Field(None, description="Valido entre 0 y 100; fuera de rango se imputa. Variable clave del modelo.")
    puntaje_datacredito: Optional[float] = Field(None, description="Valido entre 150 y 999; fuera de rango se imputa.")
    cant_creditosvigentes: Optional[float] = None
    huella_consulta: Optional[float] = None
    creditos_sectorFinanciero: Optional[float] = None
    creditos_sectorCooperativo: Optional[float] = None
    creditos_sectorReal: Optional[float] = None
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = Field(None, description="Estable, Creciente o Decreciente.")


class Prediccion(BaseModel):
    prediccion: int = Field(description="1 = paga a tiempo, 0 = no paga a tiempo")
    decision: str
    probabilidad_no_pago: float
    probabilidad_pago: float
    umbral_riesgo: float
    campos_no_informados: List[str]


# ---------------------------------------------------------------
# LOGICA DE PREDICCION
# ---------------------------------------------------------------
def _categorias_aprendidas(artefacto: dict, variable: str) -> list:
    """Categorias que el One-Hot del pipeline aprendio para `variable` (como texto)."""
    try:
        for _, paso in artefacto["pipeline"].named_steps["preprocesador"].steps:
            if hasattr(paso, "encoder_dict_") and variable in paso.encoder_dict_:
                return [str(c) for c in paso.encoder_dict_[variable]]
    except Exception:
        pass
    return []


def _codigo_a_texto(valor, categorias: list) -> str:
    """
    Convierte el codigo de tipo_credito a texto igual que en el entrenamiento
    (4, 4.0 y "4" deben dar la misma categoria). Un dato faltante pasa a "Sin_dato".
    """
    if pd.isna(valor):
        return "Sin_dato"
    candidatos = []
    try:
        numero = float(valor)
        if numero.is_integer():
            candidatos.append(str(int(numero)))
        candidatos.append(str(numero))
    except (TypeError, ValueError):
        pass
    candidatos.append(str(valor).strip())
    for candidato in candidatos:
        if candidato in categorias:
            return candidato
    return candidatos[0]


def _construir_dataframe(clientes: List[Cliente], artefacto: dict) -> pd.DataFrame:
    columnas = artefacto["columnas_entrada"]
    df = pd.DataFrame([c.model_dump() for c in clientes])
    for col in columnas:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columnas].copy()
    for col in columnas:
        if col in COLUMNAS_CATEGORICAS:
            df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tipo_credito" in df.columns:
        categorias = _categorias_aprendidas(artefacto, "tipo_credito")
        df["tipo_credito"] = [_codigo_a_texto(v, categorias) for v in df["tipo_credito"]]
    return df


def predecir(clientes: List[Cliente]) -> List[Prediccion]:
    artefacto = estado["artefacto"]
    columnas = artefacto["columnas_entrada"]
    df = _construir_dataframe(clientes, artefacto)
    no_informados = [[c for c in columnas if pd.isna(fila[c])] for _, fila in df.iterrows()]

    X = limpiar_base(df)[columnas]
    prob_pago = artefacto["pipeline"].predict_proba(X)[:, 1]

    resultados = []
    for p_pago, faltantes in zip(prob_pago, no_informados):
        p_no_pago = 1.0 - float(p_pago)
        riesgo = p_no_pago >= UMBRAL_RIESGO
        resultados.append(Prediccion(
            prediccion=0 if riesgo else 1,
            decision="No paga a tiempo (riesgo alto)" if riesgo else "Paga a tiempo",
            probabilidad_no_pago=round(p_no_pago, 6),
            probabilidad_pago=round(float(p_pago), 6),
            umbral_riesgo=UMBRAL_RIESGO,
            campos_no_informados=faltantes,
        ))
    return resultados


def _exigir_modelo():
    if estado["artefacto"] is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {estado['error']}")


# ---------------------------------------------------------------
# APLICACION
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open(RUTA_MODELO, "rb") as f:
            estado["artefacto"] = pickle.load(f)
        estado["error"] = None
        print(f"Modelo cargado desde {RUTA_MODELO}: {estado['artefacto']['nombre_modelo']}")
    except Exception as error:  # el servicio arranca igual y /health informa el problema
        estado["artefacto"], estado["error"] = None, f"{type(error).__name__}: {error}"
        print(f"No se pudo cargar el modelo desde {RUTA_MODELO}: {estado['error']}")
    yield


app = FastAPI(
    title="API de riesgo crediticio",
    version="1.3.0",
    description="Devuelve la probabilidad de que un cliente NO pague a tiempo su credito.",
    lifespan=lifespan,
)


@app.get("/")
def raiz():
    return {"servicio": "API de riesgo crediticio", "version": app.version, "documentacion": "/docs"}


@app.get("/health")
def health():
    return {"estado": "ok" if estado["artefacto"] is not None else "error",
            "modelo_cargado": estado["artefacto"] is not None,
            "detalle": estado["error"]}


@app.get("/info")
def info():
    _exigir_modelo()
    artefacto = estado["artefacto"]
    metricas = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in artefacto.get("metricas_test", {}).items()}
    return {"modelo": artefacto["nombre_modelo"],
            "objetivo": artefacto.get("significado_target"),
            "columnas_entrada": artefacto["columnas_entrada"],
            "metricas_test": metricas}


@app.post("/predict", response_model=Prediccion)
def predict(cliente: Cliente):
    _exigir_modelo()
    try:
        return predecir([cliente])[0]
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar el cliente: {error}")


@app.post("/predict_batch", response_model=List[Prediccion])
def predict_batch(clientes: List[Cliente]):
    _exigir_modelo()
    if not clientes:
        raise HTTPException(status_code=400, detail="La lista de clientes esta vacia.")
    if len(clientes) > MAX_CLIENTES_POR_LOTE:
        raise HTTPException(status_code=400, detail=f"Maximo {MAX_CLIENTES_POR_LOTE} clientes por solicitud.")
    try:
        return predecir(clientes)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar el lote: {error}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)