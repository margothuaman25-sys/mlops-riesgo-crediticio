# Despliegue de Proyecto — Modelo de Riesgo Crediticio

Proyecto Integrador PIM5 (Data Science, Henry) · Autor: Margot Huaman Quispe

Modelo de aprendizaje automático que, a partir del historial de créditos de una empresa financiera, estima si un cliente **pagará a tiempo** su crédito. El proyecto cubre el ciclo completo: exploración de datos, ingeniería de características, entrenamiento y evaluación, y monitoreo de *data drift* con un tablero en Streamlit. El despliegue incluye una API con FastAPI y una imagen Docker.

---

## 1. Caso de negocio

La empresa necesita anticipar el comportamiento de los nuevos solicitantes para decidir con más información a quién otorgar un crédito.

- **Variable a predecir:** `Pago_atiempo` (1 = paga a tiempo, 0 = no paga a tiempo).
- **Clase de interés: la 0.** Un cliente que no paga representa dinero que no se recupera, y por eso el modelo se evalúa con las métricas de esa clase.
- **Los dos errores tienen costos distintos:**
  - Decir "paga" cuando en realidad **no paga** (falso negativo para el riesgo): se presta dinero que no se recupera.
  - Decir "no paga" cuando en realidad **sí paga** (falso positivo): se rechaza a un buen cliente y se pierde una oportunidad de negocio.

## 2. Datos

| Concepto | Detalle |
|---|---|
| Archivo | `Base_de_datos.csv` (créditos históricos) |
| Tamaño | 10.763 créditos |
| Variable objetivo | `Pago_atiempo`: 95,25 % paga a tiempo y 4,75 % no paga |
| Exclusión | `fecha_prestamo` no se usa para entrenar: es la fecha de originación y no se repite en producción |

El desbalance de clases es fuerte (≈ 95 % / 5 %), por lo que la *accuracy* no es una métrica confiable: un modelo que siempre dijera "paga" tendría ~95 % de accuracy sin detectar un solo caso de riesgo.

## 3. Estructura del proyecto (fija, no modificable)

```
mlops_pipeline/
└── src/
    ├── Cargar_datos.ipynb            # carga y validación inicial del dataset
    ├── comprension_eda.ipynb         # EDA: univariable, bivariable, multivariable
    ├── ft_engineering.py             # ingeniería de características (pipeline de scikit-learn)
    ├── model_training_evaluation.py  # entrenamiento y evaluación de modelos supervisados
    ├── model_deploy.py               # API (FastAPI) para servir el modelo (Avance #4)
    └── model_monitoring.py           # monitoreo y detección de data drift + tablero Streamlit
Base_de_datos.csv
requirements.txt
.gitignore
readme.md
```

Además se versiona `modelo_riesgo_crediticio.pkl` en la raíz: es el modelo entrenado que usa `model_deploy.py`. También se versiona el `Dockerfile` de la API (Avance #4). Los CSV generados localmente (`Base_de_datos_eda.csv`, `Base_de_datos_tipada.csv`) están en `.gitignore`.

## 4. Proceso

### 4.1 Exploración (Avance #1)
`Cargar_datos.ipynb` y `comprension_eda.ipynb` contienen la carga, la validación inicial y el análisis univariable, bivariable y multivariable.

### 4.2 Ingeniería de características (Avance #2) — `ft_engineering.py`
El código se divide en dos partes para **evitar fuga de datos**:

**A) Reglas de negocio (`limpiar_base`)**: operan fila por fila y no aprenden nada de los datos, por lo que son seguras antes de dividir en train y test.
- `edad_cliente` fuera de [18, 100], `puntaje` fuera de [0, 100] y `puntaje_datacredito` fuera de [150, 999] se tratan como nulos (valores imposibles); ver en la sección 5.1 el efecto de esta regla sobre `puntaje`.
- `tendencia_ingresos` solo admite `Estable`, `Creciente` o `Decreciente`: se detectaron ~58 filas con números por un error de mapeo en el origen y se anulan.
- `tipo_credito` llega como código numérico, pero es categórico: se convierte a texto.

**B) Pipeline de scikit-learn con feature-engine (`construir_preprocesador`)**: aprende de los datos, por lo que **se ajusta solo con el conjunto de entrenamiento**.
1. Imputación con la mediana de los valores imposibles (`MeanMedianImputer`).
2. Winsorización al percentil 99 en `salario_cliente`, `total_otros_prestamos` y `cant_creditosvigentes` (`Winsorizer`).
3. Nulos categóricos como categoría `Sin_dato` (`CategoricalImputer`): no se inventa información con la moda, porque la ausencia del dato puede ser informativa.
4. `promedio_ingresos_datacredito` (~27 % de nulos): indicador de dato faltante (`AddMissingIndicator`) + mediana por `tipo_laboral` (transformador propio).
5. Saldos nulos → 0 (`ArbitraryNumberImputer`), porque representan ausencia de deuda registrada.
6. Codificación One-Hot de `tipo_laboral`, `tipo_credito` y `tendencia_ingresos` (`OneHotEncoder`).

### 4.3 Modelamiento y evaluación (Avance #2) — `model_training_evaluation.py`
- Partición estratificada 80/20 (8.610 filas de entrenamiento y 2.153 de prueba, con 102 clientes que no pagan en la prueba). `random_state=42`.
- Dos modelos, ambos dentro de un `Pipeline` que incluye el preprocesamiento:
  - **Regresión Logística** (línea base): escalado + `class_weight='balanced'`.
  - **XGBoost**: `scale_pos_weight` = negativos / positivos (≈ 0,05; la etiqueta 1 es la mayoritaria, por lo que el valor reduce su peso para equilibrar).
- **Selección del modelo:** validación cruzada estratificada de 5 folds sobre el entrenamiento, con el **F1 de la clase 0** como criterio.
- El artefacto guardado (`modelo_riesgo_crediticio.pkl`) contiene el pipeline completo (preprocesamiento + modelo), el listado de columnas de entrada y las métricas.

### 4.4 Despliegue como API (Avance #4) — `model_deploy.py` y `Dockerfile`

API con **FastAPI** que carga el pipeline guardado (`modelo_riesgo_crediticio.pkl`), aplica `limpiar_base` al cliente nuevo y devuelve la probabilidad de que **no pague a tiempo**.

| Endpoint | Función |
|---|---|
| `GET /` | Datos básicos del servicio |
| `GET /health` | Estado del servicio y si el modelo está cargado |
| `GET /info` | Modelo cargado, columnas de entrada y métricas de prueba |
| `POST /predict` | Predicción para un cliente |
| `POST /predict_batch` | Predicción para una lista de clientes (máximo 1.000) |

La respuesta de `/predict` incluye `prediccion` (1 = paga, 0 = no paga), `decision`, `probabilidad_no_pago`, `probabilidad_pago`, `umbral_riesgo` (0,5) y `campos_no_informados`.

Decisiones de diseño:
- Los campos que no se informan se tratan como datos faltantes y el pipeline los imputa igual que en el entrenamiento. La respuesta los lista, porque la calidad de la predicción baja si faltan muchos (en especial `puntaje`).
- Un nombre de campo mal escrito o un valor de tipo incorrecto devuelve un error 422 en lugar de ignorarse.
- `tipo_credito` se normaliza (`4`, `4.0` y `"4"` son la misma categoría) usando las categorías que aprendió el entrenamiento.
- Si el modelo no se puede cargar, el servicio arranca igual: `/health` informa el problema y `/predict` responde 503.

**Imagen Docker (`Dockerfile`, en la raíz):** parte de `python:3.14-slim`, instala las dependencias de `requirements.txt` sin las que la API no necesita (notebooks, Streamlit y gráficos), copia solo `ft_engineering.py`, `model_deploy.py` y el modelo respetando la estructura de carpetas del repositorio, se ejecuta con un usuario sin privilegios e incluye un `HEALTHCHECK` sobre `/health`.

Pruebas realizadas:
- **API local con `uvicorn`:** `/health` responde `estado: ok` y `/predict` con un cliente de ejemplo (puntaje 95,23) devolvió "Paga a tiempo" con una probabilidad de no pago de 0,026.
- **Imagen Docker:** se construyó sin errores y el contenedor arrancó y respondió en `/` (versión 1.3.0) y en `/docs`. La prueba se hizo en GitHub Codespaces, porque el equipo local no tiene habilitada la virtualización que requiere Docker Desktop.

## 5. Resultados

**Validación cruzada (5 folds, clase 0):**

| Modelo | F1 | Precision | Recall |
|---|---|---|---|
| Regresión Logística | 0,600 ± 0,016 | 0,481 | 0,800 |
| **XGBoost** | **0,811 ± 0,024** | 0,864 | 0,765 |

**Conjunto de prueba (clase 0 = no paga):**

| Modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Regresión Logística | 0,500 | 0,794 | 0,614 | 0,813 | 0,947 |
| **XGBoost (modelo final)** | **0,854** | **0,745** | **0,796** | **0,823** | **0,950** |

Matriz de confusión del modelo final (filas = real, columnas = predicción):

| | Pred. no paga | Pred. paga |
|---|---|---|
| **Real: no paga** | 76 | 26 |
| **Real: paga** | 13 | 2.038 |

**Lectura de negocio:** de cada 100 clientes que no pagan, el modelo detecta unos 75; de cada 100 alertas de "no pagará", unas 85 son correctas, y se rechaza a menos de 1 de cada 100 buenos clientes (13 de 2.051).

> El F1 de la clase 1 (paga) es muy alto en ambos modelos (0,975 en Regresión Logística y 0,991 en XGBoost) porque esa clase es el 95 % de los datos. Por eso no se usa como métrica de selección.

**Variables más importantes (XGBoost):** `puntaje` (~58 % de la importancia), `saldo_mora` (~7 %), `creditos_sectorCooperativo`, `puntaje_datacredito` y `huella_consulta`.

### 5.1 Hallazgo: dependencia de `puntaje`

Al revisar por qué `puntaje` concentra tanta importancia se encontró que **separa casi perfectamente las dos clases**:

| Indicador | Pagan a tiempo (10.252) | No pagan (511) |
|---|---|---|
| `puntaje` mínimo | 63,8 | -38,0 |
| `puntaje` máximo | 95,23 | 62,7 |
| Con el valor por defecto 95,227787 | 91,8 % | 0 % |
| Con `puntaje` negativo | 0 | 26,4 % |

- Los rangos **no se solapan**: ningún cliente que no paga supera 62,7 y ninguno de los que pagan baja de 63,8.
- La limpieza de negocio trata los puntajes fuera de [0, 100] como valores imposibles y los reemplaza por la mediana (95,23, el valor de quienes pagan). Por eso el ~26 % de los clientes que no pagan y tienen `puntaje` negativo se confunde con un buen pagador, lo que explica la mayor parte de los casos que el modelo no detecta (26 de 102, un 25,5 %).
- **Sin `puntaje` el modelo casi no predice** (XGBoost, mismo pipeline y misma partición):

| XGBoost, clase 0 (no paga) | Precision | Recall | F1 |
|---|---|---|---|
| Con `puntaje` (modelo final) | 0,854 | 0,745 | 0,796 |
| Sin `puntaje` | 0,128 | 0,206 | 0,158 |

**Interpretación:** un puntaje conocido al momento de otorgar el crédito rara vez separa tan limpiamente a quienes pagan de quienes no. Es probable que se trate de un artefacto de la base sintética del curso o de un valor calculado *después* del resultado del crédito. En un caso real habría que confirmar con el dueño de los datos cuándo y cómo se calcula `puntaje` antes de usarlo; si se calculara después del desembolso, debería excluirse y el modelo tendría muy poca capacidad predictiva con el resto de las variables.

**Decisión:** se mantiene `puntaje` y la regla de negocio de rango [0, 100], porque el foco del proyecto es el flujo de ingeniería y despliegue. No se usa el puntaje negativo como señal, para no incorporar de forma deliberada un patrón que probablemente es una fuga de información.

## 6. Monitoreo de data drift (Avance #3) — `model_monitoring.py`

*Data drift* es cuando los datos que llegan en producción dejan de parecerse a los datos de entrenamiento. Se compara una ventana de **referencia** (70 % más antiguo de la base, por `fecha_prestamo`) contra una ventana **actual** (30 % más reciente, o un CSV nuevo cargado en la app).

- **PSI (Population Stability Index)** por variable, como criterio principal: `< 0,10` estable · `0,10 – 0,25` alerta · `≥ 0,25` drift significativo.
- **Prueba KS** (numéricas) y **chi-cuadrado** (categóricas) como apoyo. Con muchos datos detectan diferencias mínimas como "significativas", por eso no son el criterio único.
- **Drift de las predicciones:** PSI de la probabilidad de no pago y porcentaje de clientes marcados como riesgo.

**Resultado con la base actual** (7.534 filas de referencia y 3.229 actuales): 18 variables estables, 2 en alerta y 1 con drift.

| Variable | PSI | Estado |
|---|---|---|
| `promedio_ingresos_datacredito` | 0,334 | Drift |
| `plazo_meses` | 0,160 | Alerta |
| `total_otros_prestamos` | 0,130 | Alerta |

El PSI de las predicciones es 0,0076 (estable). El porcentaje de clientes marcados como riesgo pasa de 5,32 % a 3,47 %, un cambio que el PSI no resalta porque casi todos los clientes tienen una probabilidad de no pago cercana a 0.

**Uso:**
```bash
python model_monitoring.py            # reporte en la terminal
streamlit run model_monitoring.py     # tablero interactivo en http://localhost:8501
```

## 7. Cómo ejecutar el proyecto

Probado con Python 3.14 en Windows.

```bash
git clone https://github.com/margothuaman25-sys/mlops-riesgo-crediticio.git
cd mlops-riesgo-crediticio
python -m venv venv
venv\Scripts\activate          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
cd mlops_pipeline\src          # macOS / Linux: cd mlops_pipeline/src
```

Ejecución, en orden, desde `mlops_pipeline/src`:

```bash
python model_training_evaluation.py   # entrena, evalúa y guarda modelo_riesgo_crediticio.pkl
python model_monitoring.py            # reporte de data drift
streamlit run model_monitoring.py     # tablero de monitoreo
python ft_engineering.py              # opcional: genera Base_de_datos_tipada.csv solo para inspección
```

API (desde `mlops_pipeline/src`):

```bash
uvicorn model_deploy:app --reload     # documentación interactiva en http://127.0.0.1:8000/docs
```

Docker (desde la raíz del repositorio):

```bash
docker build -t riesgo-crediticio-api .
docker run -d -p 8000:8000 --name api riesgo-crediticio-api
curl http://localhost:8000/health     # en Windows PowerShell: curl.exe http://localhost:8000/health
docker stop api && docker rm api      # detener y eliminar el contenedor
```

Para abrir los notebooks, selecciona el entorno virtual como kernel (`ipykernel` está en `requirements.txt`).

> El entrenamiento no usa `Base_de_datos_tipada.csv`: parte de la base cruda y aplica el pipeline completo, para que el preprocesamiento se ajuste solo con el entrenamiento.

## 8. Estrategia de ramas y versiones

- `developer` → desarrollo diario, commits frecuentes.
- `certification` → integración y QA antes de pasar a `master`.
- `master` → versión estable, lista para producción.

Flujo por avance: trabajar en `developer` → Pull Request a `certification` → validar → Pull Request a `master` → release con su tag.

| Versión | Flujo | Contenido |
|---|---|---|
| V1.0.0 | developer → certification → master | Estructura de carpetas inicial (incluida dentro de V1.0.1; no existe un tag V1.0.0 separado) |
| V1.0.1 | developer → certification → master | `Cargar_datos.ipynb` + `comprension_eda.ipynb` |
| V1.1.0 | developer → certification → master | `ft_engineering.py` + primera versión de `model_training_evaluation.py` |
| V1.1.1 | developer → certification → master | Pipeline sin fuga de datos, métricas por clase 0, validación cruzada y `requirements.txt` con versiones fijas |
| V1.2.0 | developer → certification → master | `model_monitoring.py` (data drift y tablero Streamlit) y README completo |
| V1.3.0 | developer → certification → master | `model_deploy.py` (API con FastAPI), `Dockerfile` y README actualizado |

> **Nota sobre el versionado:** la estructura inicial (V1.0.0) y los notebooks de EDA (V1.0.1) se subieron en un mismo commit, por eso no existe un tag V1.0.0 aparte. En el tag V1.0.1 la carpeta `src/` todavía estaba en la raíz; la estructura final `mlops_pipeline/src/` quedó establecida a partir de V1.1.0.

## 9. Decisiones y limitaciones

- **Desbalance de clases:** se usa `class_weight` / `scale_pos_weight`, sin remuestreo. El profesor del curso confirmó que tratar el desbalance no es obligatorio para la entrega.
- **Umbral de decisión:** se usa el estándar de 0,5. En un entorno real convendría ajustarlo según el costo de cada tipo de error.
- **Muestra pequeña de la clase 0:** el conjunto de prueba tiene solo 102 clientes que no pagan, por lo que las métricas tienen incertidumbre. La validación cruzada de 5 folds ofrece una estimación más estable.
- **Variable `puntaje`:** el modelo depende casi por completo de ella (ver sección 5.1). La base es sintética, según informó el equipo docente, y esa dependencia probablemente es un artefacto de cómo se generó. En un entorno real habría que validar con el dueño de los datos cuándo se calcula `puntaje`, y las métricas del modelo deberían interpretarse con esa salvedad.
- **Monitoreo:** no hay datos de producción reales, así que la ventana "actual" se simula con la parte más reciente de la base. Como el modelo se entrenó con una partición aleatoria de toda la base, el monitoreo compara distribuciones de datos y de predicciones, no el desempeño real.
- **PSI:** resume toda la distribución y puede no resaltar cambios en las colas, como el porcentaje de clientes en riesgo.
- **API:** es una versión de demostración, sin autenticación ni límite de uso, y no registra las predicciones. Para producción harían falta seguridad, registro de solicitudes y monitoreo del servicio.
- **Docker:** la imagen se construyó y se probó en GitHub Codespaces, no con Docker Desktop en el equipo local, que no tiene habilitada la virtualización.
- **Cargar el `.pkl`:** requiere tener `mlops_pipeline/src` en el `PYTHONPATH`, porque el pipeline usa una clase definida en `ft_engineering.py`.

## 10. Próximos pasos

- Automatizar pruebas y validación con CI/CD (por ejemplo, GitHub Actions o SonarCloud): pruebas unitarias de `limpiar_base` y de la API.
- Reentrenar el modelo de forma programada cuando el monitoreo detecte drift.
- Confirmar con el dueño de los datos cuándo se calcula `puntaje` y reevaluar el modelo con esa información.
- Ajustar el umbral de decisión según el costo de cada tipo de error.