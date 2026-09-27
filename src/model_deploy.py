"""
model_deploy.py
-----------------
Expone el modelo de riesgo crediticio entrenado como una API REST con FastAPI.

Ejecutar:
    uvicorn src.model_deploy:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ft_engineering import clean_tendencia_ingresos, add_derived_features

app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Predice si un nuevo cliente pagará su crédito a tiempo (Pago_atiempo).",
    version="1.0.0",
)

model = joblib.load("modelo_riesgo_crediticio.pkl")


class CreditoInput(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: int
    total_otros_prestamos: int
    cuota_pactada: int
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API de riesgo crediticio activa"}


@app.post("/predict")
def predict(credito: CreditoInput):
    df = pd.DataFrame([credito.dict()])
    df = clean_tendencia_ingresos(df)
    df = add_derived_features(df)
    proba_no_pago = 1 - model.predict_proba(df)[0, 1]
    pred = int(model.predict(df)[0])
    return {
        "pago_a_tiempo_predicho": pred,
        "probabilidad_no_pago": round(float(proba_no_pago), 4),
    }
