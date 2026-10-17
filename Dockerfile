# Imagen de la API de riesgo crediticio (Avance #4)
# Construir:  docker build -t riesgo-crediticio-api .
# Ejecutar:   docker run -p 8000:8000 riesgo-crediticio-api
# Probar:     http://localhost:8000/docs
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias: se reutiliza requirements.txt (una sola fuente de versiones),
# quitando lo que la API no necesita (notebooks, tablero Streamlit y graficos).
COPY requirements.txt .
RUN tr -d '\r' < requirements.txt \
    | grep -v -E '^(#|ipykernel|streamlit|matplotlib|seaborn|openpyxl|$)' > requirements-api.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# Codigo y modelo, con la misma estructura de carpetas que el repositorio
COPY mlops_pipeline/src/ft_engineering.py mlops_pipeline/src/model_deploy.py ./mlops_pipeline/src/
COPY modelo_riesgo_crediticio.pkl ./

WORKDIR /app/mlops_pipeline/src

# Se ejecuta sin privilegios de administrador
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, sys, urllib.request; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['modelo_cargado'] else 1)"

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]