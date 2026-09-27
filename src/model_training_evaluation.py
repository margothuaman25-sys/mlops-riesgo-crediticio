"""
model_training_evaluation.py
-----------------------------
Entrena y compara modelos supervisados para predecir Pago_atiempo (riesgo crediticio).
El dataset está desbalanceado (~95% paga a tiempo / ~5% no paga), por lo que se usa
class_weight='balanced' y se evalúa con ROC-AUC, PR-AUC, recall y F1 de la clase minoritaria,
NO con accuracy.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report,
)

from ft_engineering import (
    load_data, clean_tendencia_ingresos, filtrar_outliers_por_rango, winsorizar,
    add_derived_features, build_preprocessing_pipeline, TARGET_COL,
)

RANDOM_STATE = 42


def get_candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            scale_pos_weight=19,  # ~95/5 desbalance observado en el EDA
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(metrics)
    return metrics


def main():
    df = load_data()
    df = clean_tendencia_ingresos(df)
    df = filtrar_outliers_por_rango(df)
    df = winsorizar(df, ["salario_cliente", "total_otros_prestamos", "cant_creditosvigentes"])
    df = add_derived_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    best_pipeline, best_score, best_name = None, -1, None

    for name, model in get_candidate_models().items():
        preprocessor = build_preprocessing_pipeline()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        # PR-AUC es más informativo que ROC-AUC con este desbalance tan fuerte
        if metrics["pr_auc"] > best_score:
            best_pipeline, best_score, best_name = pipeline, metrics["pr_auc"], name

    print(f"\nModelo ganador: {best_name} (PR-AUC={best_score:.4f})")
    joblib.dump(best_pipeline, "modelo_riesgo_crediticio.pkl")
    pd.DataFrame(results).to_csv("resultados_modelos.csv", index=False)


if __name__ == "__main__":
    main()
