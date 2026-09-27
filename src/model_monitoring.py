"""
model_monitoring.py
---------------------
App en Streamlit para monitorear el modelo en producción y detectar data drift
comparando la distribución de datos de referencia (entrenamiento) vs. datos nuevos.

Ejecutar:
    streamlit run src/model_monitoring.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(page_title="Monitoreo — Riesgo Crediticio", layout="wide")
st.title("📊 Monitoreo del Modelo de Riesgo Crediticio")

st.markdown("""
Compara la distribución de los datos de **referencia** (entrenamiento) contra
los datos **nuevos** (producción) para detectar *data drift* por variable,
usando el test de Kolmogorov-Smirnov.
""")

ref_file = st.file_uploader("Dataset de referencia (entrenamiento)", type="csv")
new_file = st.file_uploader("Dataset nuevo (producción)", type="csv")

if ref_file and new_file:
    df_ref = pd.read_csv(ref_file)
    df_new = pd.read_csv(new_file)

    num_cols = df_ref.select_dtypes(include=["int64", "float64"]).columns.tolist()
    drift_results = []

    for col in num_cols:
        if col in df_new.columns:
            stat, p_value = ks_2samp(df_ref[col].dropna(), df_new[col].dropna())
            drift_results.append({
                "variable": col,
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detectado": p_value < 0.05,
            })

    drift_df = pd.DataFrame(drift_results)
    st.subheader("Resultado del test de drift (KS-test)")
    st.dataframe(drift_df)

    variable = st.selectbox("Ver distribución de:", num_cols)
    fig, ax = plt.subplots()
    ax.hist(df_ref[variable].dropna(), bins=30, alpha=0.5, label="Referencia")
    ax.hist(df_new[variable].dropna(), bins=30, alpha=0.5, label="Nuevo")
    ax.legend()
    ax.set_title(f"Distribución: {variable}")
    st.pyplot(fig)
else:
    st.info("Sube ambos datasets (referencia y nuevo) para iniciar el análisis.")
