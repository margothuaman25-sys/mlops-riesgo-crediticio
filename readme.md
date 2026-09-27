# Despliegue de Proyecto — Modelo de Riesgo Crediticio

**Rol:** Científico/a de Datos Junior Advanced — Equipo de Datos y Analítica
**Objetivo de negocio:** Predecir el comportamiento (riesgo de incumplimiento) de nuevos usuarios de crédito a partir de información histórica, para apoyar decisiones de aprobación de crédito.

## Estructura del proyecto (fija, no modificable)

```
mlops_pipeline/
└── src/
    ├── Cargar_datos.ipynb          # carga y validación inicial del dataset
    ├── comprension_eda.ipynb       # EDA: univariable, bivariable, multivariable
    ├── ft_engineering.py           # ingeniería de características (pipelines sklearn)
    ├── model_training_evaluation.py# entrenamiento y evaluación de modelos supervisados
    ├── model_deploy.py             # API (FastAPI) para servir el modelo
    └── model_monitoring.py         # monitoreo y detección de data drift
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── readme.md
```

## Estrategia de ramas y versiones

- `developer` → desarrollo diario, commits frecuentes
- `certification` → integración/QA antes de pasar a master
- `master` → versión estable, lista para producción

| Versión | Rama de origen | Contenido |
|---|---|---|
| V1.0.0 | developer → certification → master | Estructura de carpetas inicial |
| V1.0.1 | developer → certification → master | Cargar_datos.ipynb + comprension_eda.ipynb |
| V1.1.0 | developer → certification → master | ft_engineering.py + primeros modelos |
| V1.1.1 | developer → certification → master | model_training_evaluation.py (selección del mejor modelo) |

Flujo por avance: trabajar en `developer` → Pull Request a `certification` → validar → Pull Request a `master`.

## Cómo levantar el entorno

```bash
git clone <URL_DE_TU_REPO>
cd mlops_pipeline
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correr la API y el monitoreo

```bash
uvicorn src.model_deploy:app --reload
streamlit run src/model_monitoring.py
```

## Caso de negocio y hallazgos (EDA — Avance #1)

- Dataset: 10,763 créditos históricos, variable objetivo `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).
- **Fuerte desbalance de clases:** 95.25% paga a tiempo vs 4.75% no paga → se prioriza ROC-AUC, PR-AUC, recall y F1 de la clase minoritaria en vez de accuracy.
- **Problema de calidad de datos:** la columna `tendencia_ingresos` mezcla categorías válidas (Estable/Creciente/Decreciente) con valores numéricos corruptos en ~27% de las filas → se recodificó a `tendencia_ingresos_limpia` con la categoría `Desconocido`.
- Nulos en columnas de saldo (`saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`) imputados con 0 (ausencia de deuda registrada).
- Variable derivada candidata para el modelo: `ratio_cuota_salario` (cuota_pactada / salario_cliente).
- Detalle completo del análisis univariable, bivariable y multivariable en `src/comprension_eda.ipynb`.

## Métricas del modelo

_(completar en Avance #2 tras correr model_training_evaluation.py: precisión, recall, F1, ROC-AUC y PR-AUC del modelo ganador — logistic_regression / random_forest / gradient_boosting / xgboost)_
