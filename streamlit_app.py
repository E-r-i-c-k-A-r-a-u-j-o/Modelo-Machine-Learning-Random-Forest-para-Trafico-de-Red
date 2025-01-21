import streamlit as st
from joblib import load
import numpy as np

# Cargar modelo y escalador
modelo = load("random_forest_model.pkl")
scaler = load("random_forest_scaler.pkl")

# Título de la aplicación
st.title("Modelo de Machine Learning - Random Forest")

# Ingreso de datos
st.header("Introduce los datos para predicción")
features = st.text_input("Características (separadas por coma)", "1.2, 3.4, 5.6, 7.8")

if st.button("Predecir"):
    try:
        # Procesar entrada
        input_features = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        scaled_features = scaler.transform(input_features)
        prediction = modelo.predict(scaled_features)
        probability = modelo.predict_proba(scaled_features)

        # Mostrar resultados
        st.success(f"Predicción: {int(prediction[0])}")
        st.info(f"Probabilidades: {probability.tolist()}")
    except Exception as e:
        st.error(f"Error al procesar la predicción: {str(e)}")
