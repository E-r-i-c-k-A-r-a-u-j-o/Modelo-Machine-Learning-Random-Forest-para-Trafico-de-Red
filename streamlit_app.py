import streamlit as st
from joblib import load
import numpy as np

# Cargar modelo y escalador
modelo = load("random_forest_model.pkl")
scaler = load("random_forest_scaler.pkl")

# Título de la aplicación
st.title("GINA PRECIOSA HERMOSA este es mi Modelo de Machine Learning para predecir el Tráfico de Red")

# Ingreso de datos
st.header("Introduce los datos a predecir")
features = st.text_input("Características (separadas por coma)", "1.2, 3.4, 5.6, 7.8")

if st.button("Predecir"):
    try:
        # Procesar entrada
        input_features = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        scaled_features = scaler.transform(input_features)
        prediction = modelo.predict(scaled_features)

        # Interpretar la predicción
        if int(prediction[0]) == 1:
            resultado = "El tráfico de red es normal"
        else:
            resultado = "El tráfico de red es un ataque"

        # Mostrar resultados
        st.success(f"Resultado: {resultado}")
    except Exception as e:
        st.error(f"Intenta de nuevo: {str(e)}")

