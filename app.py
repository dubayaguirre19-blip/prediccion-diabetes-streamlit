import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Predicción académica de diabetes", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
PIPELINE_PATH = BASE / "modelo_pipeline.joblib"
METRICS_PATH = BASE / "metricas_modelo.json"
DATA_PATH = BASE / "diabetes_limpio.csv"
TEST_PATH = BASE / "registro_prueba.csv"
EXPECTED_PATH = BASE / "resultado_registro_prueba.json"

st.title("Aplicación académica de predicción de diabetes")
st.write(
    "Proyecto de Minería de Datos: integración de un modelo de clasificación "
    "con una interfaz interactiva."
)

with st.expander("Objetivo, datos y alcance", expanded=True):
    st.write(
        "El objetivo es mostrar cómo un modelo de aprendizaje automático puede "
        "clasificar registros del conjunto Pima Indians Diabetes según las variables "
        "incluidas en el dataset."
    )
    st.write(
        "Fuente de datos: Pima Indians Diabetes Database, distribuida en formato CSV "
        "en el repositorio público de Jason Brownlee."
    )
    st.write("Integrantes: completar con los nombres de todos los integrantes del grupo.")
    st.warning(
        "Este sitio es un prototipo académico. No diagnostica diabetes, no sustituye "
        "una consulta médica y sus resultados no deben utilizarse para tomar decisiones clínicas."
    )

@st.cache_resource
def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "No se encuentra modelo_pipeline.joblib. Ejecuta todas las celdas del "
            "notebook actualizado y sube el archivo generado al repositorio."
        )
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("No se encuentra metricas_modelo.json. Genera los archivos desde el notebook.")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

try:
    model = load_pipeline()
    info = load_metrics()
except Exception as exc:
    st.error(str(exc))
    st.stop()

features = info.get("variables", [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
])

st.header("Exploración de los datos")
data = load_data()
if data is None:
    st.info("Para mostrar las gráficas, ejecuta el notebook y agrega diabetes_limpio.csv al repositorio.")
else:
    c1, c2 = st.columns(2)
    with c1:
        counts = data["Outcome"].map({0: "Sin diabetes (0)", 1: "Con diabetes (1)"}).value_counts().rename_axis("Clase").reset_index(name="Registros")
        fig = px.bar(counts, x="Clase", y="Registros", title="Distribución de la variable objetivo")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(data, x="Glucose", color="Outcome", nbins=25,
                           title="Distribución de glucosa por clase",
                           labels={"Outcome": "Clase", "Glucose": "Glucosa"})
        st.plotly_chart(fig, use_container_width=True)
    fig = px.box(data, x="Outcome", y="BMI", color="Outcome",
                 title="Distribución del IMC según la clase",
                 labels={"Outcome": "Clase (0 = sin diabetes, 1 = con diabetes)", "BMI": "IMC"})
    st.plotly_chart(fig, use_container_width=True)

st.header("Consultar una predicción")
st.caption("Introduce valores numéricos. Los campos representan las variables del conjunto de datos.")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucosa (mg/dL)", min_value=1.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Presión arterial diastólica (mmHg)", min_value=1.0, max_value=200.0, value=70.0)
        skin = st.number_input("Espesor del pliegue cutáneo (mm)", min_value=0.0, max_value=100.0, value=20.0)
    with col2:
        insulin = st.number_input("Insulina (μU/mL)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("Índice de masa corporal (kg/m²)", min_value=1.0, max_value=100.0, value=30.0)
        pedigree = st.number_input("Función de pedigrí de diabetes", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Edad (años)", min_value=18, max_value=120, value=30, step=1)
    submitted = st.form_submit_button("Realizar predicción", type="primary")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }], columns=features)

    # En el notebook, los ceros de estas mediciones se convierten en faltantes
    # y el imputador del pipeline los completa con la mediana aprendida.
    columnas_cero_faltante = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    row[columnas_cero_faltante] = row[columnas_cero_faltante].replace(0, np.nan)

    if (row[["Pregnancies", "Age"]] < 0).any().any():
        st.error("Revisa los valores ingresados.")
    else:
        prediction = int(model.predict(row)[0])
        st.subheader("Resultado del modelo")
        if prediction == 1:
            st.warning("El modelo clasifica este registro en la clase 1.")
        else:
            st.success("El modelo clasifica este registro en la clase 0.")

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0][1])
            st.metric("Probabilidad estimada para la clase 1", f"{probability:.1%}")
        st.caption("La clase estimada es una salida estadística del modelo, no un diagnóstico médico.")

st.header("Evaluación del modelo")
st.write(f"**Modelo seleccionado:** {info.get('modelo', 'No especificado')}")
metrics = info.get("metricas_test", {})
metric_labels = [
    ("accuracy", "Accuracy"), ("precision", "Precision"),
    ("recall", "Recall"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC")
]
cols = st.columns(len(metric_labels))
for col, (key, label) in zip(cols, metric_labels):
    if key in metrics:
        col.metric(label, f"{metrics[key]:.3f}")

cm = info.get("matriz_confusion")
if cm:
    cm_df = pd.DataFrame(cm, index=["Real: sin diabetes", "Real: con diabetes"],
                         columns=["Predicho: sin diabetes", "Predicho: con diabetes"])
    st.write("Matriz de confusión")
    st.dataframe(cm_df, use_container_width=True)

st.subheader("Validación cruzada")
cv = info.get("validacion_cruzada", {})
if cv:
    cv_df = pd.DataFrame(cv).T
    st.dataframe(cv_df.style.format("{:.3f}"), use_container_width=True)

st.subheader("Limitaciones")
st.markdown(
    "- El conjunto de datos tiene 768 registros y no representa necesariamente a todas las poblaciones.\n"
    "- Existe desbalance entre las clases.\n"
    "- Algunas mediciones contienen ceros tratados como valores faltantes e imputados dentro del pipeline.\n"
    "- El rendimiento depende de la calidad y representatividad de los datos.\n"
    "- La predicción no equivale a una evaluación médica."
)

with st.expander("Comprobación con un registro conocido del conjunto de prueba"):
    if TEST_PATH.exists() and EXPECTED_PATH.exists():
        test_row = pd.read_csv(TEST_PATH)
        with open(EXPECTED_PATH, encoding="utf-8") as f:
            expected = json.load(f)["prediccion_esperada"]
        actual = int(model.predict(test_row[features])[0])
        st.write(f"Predicción guardada desde el notebook: **{expected}**")
        st.write(f"Predicción obtenida en Streamlit: **{actual}**")
        if actual == expected:
            st.success("La predicción coincide con la obtenida en el notebook.")
        else:
            st.error("La predicción no coincide. Verifica que el pipeline y los archivos pertenezcan a la misma ejecución.")
    else:
        st.info("Agrega registro_prueba.csv y resultado_registro_prueba.json, generados por el notebook, para ejecutar esta comprobación.")
