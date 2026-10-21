"""
model_monitoring.py
--------------------
Monitoreo y deteccion de DATA DRIFT para el modelo de riesgo crediticio.
 
Proyecto Integrador PIM5 - Despliegue de modelo de riesgo crediticio
Autora: Margot Huaman Quispe
 
Que es el data drift: cuando los datos que llegan en produccion dejan de
parecerse a los datos con los que se entreno el modelo (cambia el perfil de
los clientes, la economia, el proceso de captura de datos...). Si eso pasa,
el modelo puede perder calidad aunque el codigo siga igual.
 
Como se detecta aqui (se compara una ventana de REFERENCIA contra una
ventana ACTUAL):
  - PSI (Population Stability Index) por variable: es el criterio principal.
        PSI < 0.10         -> Estable
        0.10 <= PSI < 0.25 -> Alerta (revisar)
        PSI >= 0.25        -> Drift significativo (accion: investigar/reentrenar)
  - Prueba KS (variables numericas) y chi-cuadrado (variables categoricas).
    Con muchos datos estas pruebas marcan diferencias minimas como
    "significativas", por eso se muestran como apoyo y no como criterio unico.
  - Drift de las PREDICCIONES: PSI de la probabilidad de no pago que entrega
    el modelo y porcentaje de clientes marcados como riesgo.
 
De donde salen los datos:
  - Referencia = la parte mas antigua de la base (por fecha_prestamo).
  - Actual     = la parte mas reciente, o un CSV nuevo que se suba en la app.
 
Uso:
    python model_monitoring.py              -> reporte en la terminal
    streamlit run model_monitoring.py       -> tablero interactivo (Streamlit)
"""
 
import pickle
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import limpiar_base  # noqa: E402
 
# ---------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ_PROYECTO / "Base_de_datos.csv"
RUTA_MODELO = RAIZ_PROYECTO / "modelo_riesgo_crediticio.pkl"
 
COLUMNA_FECHA = "fecha_prestamo"
UMBRAL_PSI_ALERTA = 0.10
UMBRAL_PSI_DRIFT = 0.25
UMBRAL_RIESGO = 0.50          # probabilidad de no pago desde la cual se marca "riesgo"
MAX_VALORES_DISCRETOS = 10    # una numerica con <= 10 valores distintos se trata como categorica
EPSILON = 1e-4                # evita log(0) en el PSI
 
 
# ---------------------------------------------------------------
# 1. CARGA Y VENTANAS DE COMPARACION
# ---------------------------------------------------------------
def cargar_base(ruta=RUTA_DATOS) -> pd.DataFrame:
    """Carga la base cruda."""
    return pd.read_csv(ruta)
 
 
def cargar_artefacto(ruta=RUTA_MODELO) -> dict:
    """Carga el .pkl guardado por model_training_evaluation.py."""
    with open(ruta, "rb") as f:
        return pickle.load(f)
 
 
def dividir_por_fecha(df: pd.DataFrame, frac_referencia: float = 0.70):
    """
    Ordena la base por fecha_prestamo y la parte en dos: el primer
    `frac_referencia` es la referencia (el pasado) y el resto es la
    ventana actual (lo mas reciente). Si la fecha no se puede leer, se
    respeta el orden original de las filas.
    """
    df = df.copy()
    if COLUMNA_FECHA in df.columns:
        fechas = pd.to_datetime(df[COLUMNA_FECHA], errors="coerce")
        if fechas.notna().mean() > 0.90:
            df = df.assign(_fecha=fechas).sort_values("_fecha", kind="stable").drop(columns="_fecha")
        else:
            print(f"Aviso: no se pudo leer {COLUMNA_FECHA} como fecha; se usa el orden original de las filas.")
    corte = int(len(df) * frac_referencia)
    return df.iloc[:corte].reset_index(drop=True), df.iloc[corte:].reset_index(drop=True)
 
 
# ---------------------------------------------------------------
# 2. METRICAS DE DRIFT
# ---------------------------------------------------------------
def _psi_desde_proporciones(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPSILON, None)
    p_act = np.clip(p_act, EPSILON, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))
 
 
def psi_categorico(ref, act) -> float:
    """PSI para variables categoricas (o numericas con pocos valores)."""
    ref, act = pd.Series(ref).dropna(), pd.Series(act).dropna()
    if ref.empty or act.empty:
        return np.nan
    categorias = sorted(set(ref.unique()) | set(act.unique()), key=str)
    p_ref = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    p_act = act.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return _psi_desde_proporciones(p_ref, p_act)
 
 
def psi_numerico(ref, act, n_bins: int = 10) -> float:
    """
    PSI para variables numericas: se crean ~10 tramos con los cuantiles de
    la REFERENCIA y se compara que proporcion de datos cae en cada tramo.
    Los valores fuera del rango de referencia caen en los tramos extremos.
    """
    ref = pd.Series(ref).dropna().astype(float)
    act = pd.Series(act).dropna().astype(float)
    if ref.empty or act.empty:
        return np.nan
    bordes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:                       # variable casi constante
        return psi_categorico(ref, act)
    bordes[0], bordes[-1] = -np.inf, np.inf
    c_ref = np.histogram(ref, bins=bordes)[0]
    c_act = np.histogram(act, bins=bordes)[0]
    return _psi_desde_proporciones(c_ref / c_ref.sum(), c_act / c_act.sum())
 
 
def prueba_chi2(ref, act):
    """Chi-cuadrado de homogeneidad entre las categorias de ambas ventanas."""
    ref = pd.Series(ref).dropna().astype(str)
    act = pd.Series(act).dropna().astype(str)
    if ref.empty or act.empty:
        return np.nan, np.nan
    categorias = sorted(set(ref) | set(act))
    tabla = np.vstack([
        ref.value_counts().reindex(categorias, fill_value=0).to_numpy(),
        act.value_counts().reindex(categorias, fill_value=0).to_numpy(),
    ])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2:
        return 0.0, 1.0
    estadistico, p_valor, *_ = chi2_contingency(tabla)
    return float(estadistico), float(p_valor)
 
 
def clasificar_psi(psi: float) -> str:
    if psi is None or np.isnan(psi):
        return "Sin datos"
    if psi < UMBRAL_PSI_ALERTA:
        return "Estable"
    if psi < UMBRAL_PSI_DRIFT:
        return "Alerta"
    return "Drift"
 
 
# ---------------------------------------------------------------
# 3. DRIFT POR VARIABLE Y DE PREDICCIONES
# ---------------------------------------------------------------
def es_numerica_continua(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie) and serie.nunique() > MAX_VALORES_DISCRETOS
 
 
def detectar_drift(df_ref: pd.DataFrame, df_act: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Devuelve una tabla con el PSI, la prueba estadistica y el estado de cada variable."""
    filas = []
    for col in columnas:
        ref, act = df_ref[col], df_act[col]
        if es_numerica_continua(ref):
            psi = psi_numerico(ref, act)
            r, a = ref.dropna(), act.dropna()
            estadistico, p_valor = ks_2samp(r, a) if len(r) and len(a) else (np.nan, np.nan)
            prueba, tipo = "KS", "numerica"
        else:
            psi = psi_categorico(ref, act)
            estadistico, p_valor = prueba_chi2(ref, act)
            prueba, tipo = "Chi2", "categorica"
        filas.append({
            "variable": col,
            "tipo": tipo,
            "psi": psi,
            "estado": clasificar_psi(psi),
            "prueba": prueba,
            "estadistico": estadistico,
            "p_valor": p_valor,
            "nulos_ref_%": ref.isna().mean() * 100,
            "nulos_actual_%": act.isna().mean() * 100,
        })
    tabla = pd.DataFrame(filas)
    return tabla.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
 
 
def puntaje_riesgo(artefacto: dict, df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de NO pagar a tiempo (clase 0) que entrega el modelo."""
    X = limpiar_base(df)[artefacto["columnas_entrada"]]
    return 1 - artefacto["pipeline"].predict_proba(X)[:, 1]
 
 
def drift_predicciones(artefacto: dict, df_ref: pd.DataFrame, df_act: pd.DataFrame):
    """Compara lo que predice el modelo en ambas ventanas. Devuelve (resumen, score_ref, score_act)."""
    s_ref, s_act = puntaje_riesgo(artefacto, df_ref), puntaje_riesgo(artefacto, df_act)
    psi = psi_numerico(s_ref, s_act)
    resumen = {
        "psi_prediccion": psi,
        "estado": clasificar_psi(psi),
        "prob_no_pago_media_ref": float(np.mean(s_ref)),
        "prob_no_pago_media_actual": float(np.mean(s_act)),
        "pct_riesgo_ref": float(np.mean(s_ref >= UMBRAL_RIESGO) * 100),
        "pct_riesgo_actual": float(np.mean(s_act >= UMBRAL_RIESGO) * 100),
    }
    return resumen, s_ref, s_act
 
 
def ejecutar_monitoreo(df_ref: pd.DataFrame, df_act: pd.DataFrame, artefacto: dict):
    """Corre el monitoreo completo y devuelve (tabla de drift, resumen de predicciones, scores)."""
    columnas = [c for c in artefacto["columnas_entrada"] if c in df_ref.columns and c in df_act.columns]
    ref, act = limpiar_base(df_ref), limpiar_base(df_act)
    tabla = detectar_drift(ref, act, columnas)
    resumen, s_ref, s_act = drift_predicciones(artefacto, df_ref, df_act)
    return tabla, resumen, s_ref, s_act
 
 
# ---------------------------------------------------------------
# 4. REPORTE EN TERMINAL
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("MONITOREO DE DATA DRIFT")
    print("=" * 60)
    df = cargar_base()
    artefacto = cargar_artefacto()
    df_ref, df_act = dividir_por_fecha(df, 0.70)
    print(f"Referencia: {len(df_ref)} filas | Actual: {len(df_act)} filas")
 
    tabla, resumen, _, _ = ejecutar_monitoreo(df_ref, df_act, artefacto)
    pd.set_option("display.width", 200)
    print("\nDrift por variable (ordenado por PSI):")
    print(tabla[["variable", "tipo", "psi", "estado", "prueba", "p_valor"]].round(4).to_string(index=False))
 
    print("\nDrift de las predicciones del modelo:")
    print(f"  PSI de la probabilidad de no pago: {resumen['psi_prediccion']:.4f} ({resumen['estado']})")
    print(f"  % marcados como riesgo: referencia {resumen['pct_riesgo_ref']:.2f}% -> actual {resumen['pct_riesgo_actual']:.2f}%")
 
    conteo = tabla["estado"].value_counts()
    print("\nResumen:", ", ".join(f"{k}: {v}" for k, v in conteo.items()))
    if conteo.get("Drift", 0) > 0:
        print("ATENCION: hay variables con drift significativo. Investigar y evaluar un reentrenamiento.")
    elif conteo.get("Alerta", 0) > 0:
        print("Hay variables en alerta: monitorear de cerca.")
    else:
        print("Sin drift relevante: el modelo puede seguir operando con normalidad.")
 
 
# ---------------------------------------------------------------
# 5. APP DE STREAMLIT
# ---------------------------------------------------------------
def _color_estado(valor: str) -> str:
    colores = {"Estable": "background-color: #d4edda; color: #155724",
               "Alerta": "background-color: #fff3cd; color: #856404",
               "Drift": "background-color: #f8d7da; color: #721c24"}
    return colores.get(valor, "")
 
 
def app():
    import matplotlib.pyplot as plt
    import streamlit as st
 
    st.set_page_config(page_title="Monitoreo - Riesgo crediticio", layout="wide")
    st.title("Monitoreo de data drift - Modelo de riesgo crediticio")
    st.caption("Compara los datos recientes contra los datos de referencia con los que se entreno el modelo.")
 
    @st.cache_resource
    def _artefacto():
        return cargar_artefacto()
 
    @st.cache_data
    def _base():
        return cargar_base()
 
    try:
        artefacto, base = _artefacto(), _base()
    except FileNotFoundError as error:
        st.error(f"No se encontro un archivo necesario: {error}. Revisa Base_de_datos.csv y modelo_riesgo_crediticio.pkl en la raiz del proyecto.")
        st.stop()
 
    # ---- Panel lateral ----
    st.sidebar.header("Configuracion")
    fuente = st.sidebar.radio("Datos a monitorear", ["Ventana mas reciente de la base", "Subir un CSV nuevo"])
    if fuente == "Ventana mas reciente de la base":
        frac = st.sidebar.slider("% de la base usado como referencia (lo mas antiguo)", 50, 90, 70, 5) / 100
        df_ref, df_act = dividir_por_fecha(base, frac)
    else:
        archivo = st.sidebar.file_uploader("CSV con las mismas columnas que Base_de_datos.csv", type="csv")
        if archivo is None:
            st.info("Sube un CSV en el panel lateral para compararlo contra la base de referencia.")
            st.stop()
        df_act = pd.read_csv(archivo)
        faltantes = [c for c in artefacto["columnas_entrada"] if c not in df_act.columns]
        if faltantes:
            st.error(f"Al CSV le faltan estas columnas: {faltantes}")
            st.stop()
        df_ref = base
 
    tabla, resumen, s_ref, s_act = ejecutar_monitoreo(df_ref, df_act, artefacto)
 
    # ---- Resumen ----
    conteo = tabla["estado"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables estables", int(conteo.get("Estable", 0)))
    c2.metric("En alerta", int(conteo.get("Alerta", 0)))
    c3.metric("Con drift", int(conteo.get("Drift", 0)))
    c4.metric("Filas: referencia / actual", f"{len(df_ref)} / {len(df_act)}")
 
    if conteo.get("Drift", 0) > 0:
        st.error("Hay variables con drift significativo: investigar y evaluar un reentrenamiento.")
    elif conteo.get("Alerta", 0) > 0:
        st.warning("Hay variables en alerta: conviene monitorearlas de cerca.")
    else:
        st.success("Sin drift relevante: el modelo puede seguir operando con normalidad.")
 
    # ---- Tabla y grafico ----
    st.subheader("Drift por variable")
    estilo = tabla.style
    estilo = (estilo.map if hasattr(estilo, "map") else estilo.applymap)(_color_estado, subset=["estado"])
    st.dataframe(estilo.format({"psi": "{:.4f}", "estadistico": "{:.3f}", "p_valor": "{:.4f}",
                                "nulos_ref_%": "{:.2f}", "nulos_actual_%": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.bar_chart(tabla.set_index("variable")["psi"])
    with st.expander("Como leer el PSI"):
        st.markdown(
            f"- **PSI < {UMBRAL_PSI_ALERTA}**: estable.\n"
            f"- **{UMBRAL_PSI_ALERTA} a {UMBRAL_PSI_DRIFT}**: alerta, conviene revisar.\n"
            f"- **PSI >= {UMBRAL_PSI_DRIFT}**: drift significativo.\n\n"
            "Las pruebas KS y chi-cuadrado apoyan el analisis, pero con muchos datos detectan "
            "diferencias minimas, por eso el criterio principal es el PSI."
        )
 
    # ---- Detalle de una variable ----
    st.subheader("Detalle de una variable")
    variable = st.selectbox("Variable", tabla["variable"].tolist())
    ref_v, act_v = limpiar_base(df_ref)[variable], limpiar_base(df_act)[variable]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if es_numerica_continua(ref_v):
        ax.hist(ref_v.dropna(), bins=30, density=True, alpha=0.5, label="Referencia")
        ax.hist(act_v.dropna(), bins=30, density=True, alpha=0.5, label="Actual")
        ax.set_ylabel("Densidad")
    else:
        prop = pd.DataFrame({
            "Referencia": ref_v.astype(str).value_counts(normalize=True),
            "Actual": act_v.astype(str).value_counts(normalize=True),
        }).fillna(0)
        prop.plot.bar(ax=ax, alpha=0.8)
        ax.set_ylabel("Proporcion")
        ax.tick_params(axis="x", rotation=0)
    ax.set_title(variable)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
 
    # ---- Drift de predicciones ----
    st.subheader("Drift de las predicciones del modelo")
    p1, p2, p3 = st.columns(3)
    p1.metric("PSI de la probabilidad de no pago", f"{resumen['psi_prediccion']:.4f}", resumen["estado"], delta_color="off")
    p2.metric("% marcados como riesgo (referencia)", f"{resumen['pct_riesgo_ref']:.2f}%")
    p3.metric("% marcados como riesgo (actual)", f"{resumen['pct_riesgo_actual']:.2f}%")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.hist(s_ref, bins=30, density=True, alpha=0.5, label="Referencia")
    ax2.hist(s_act, bins=30, density=True, alpha=0.5, label="Actual")
    ax2.set_xlabel("Probabilidad de no pagar a tiempo")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)
 
 
def _corriendo_en_streamlit() -> bool:
    """True solo si el archivo se ejecuta con `streamlit run` (evita avisos al usar `python`)."""
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            contexto = get_script_run_ctx(suppress_warning=True)
        except TypeError:
            contexto = get_script_run_ctx()
        return contexto is not None
    except Exception:
        return False
 
 
if _corriendo_en_streamlit():
    app()
elif __name__ == "__main__":
    main()
 