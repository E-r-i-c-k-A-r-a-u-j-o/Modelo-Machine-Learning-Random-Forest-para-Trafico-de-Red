from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load

# Inicializa la aplicación FastAPI
app = FastAPI()

# Carga el modelo y el escalador
try:
    modelo = load('random_forest_model.pkl')  # Asegúrate de que el archivo está en el directorio raíz del proyecto
    scaler = load('random_forest_scaler.pkl')  # Asegúrate de que el archivo está en el directorio raíz del proyecto
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo o el escalador: {str(e)}")

# Clase para definir la estructura de los datos de entrada
class InputData(BaseModel):
    features: list[float]  # Lista de características para la predicción

# Ruta raíz de prueba
@app.get("/")
def home():
    return {"mensaje": "API funcionando correctamente"}

# Endpoint para predecir
@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convierte la entrada a un array de NumPy
        input_features = np.array(data.features).reshape(1, -1)

        # Escala los datos
        scaled_features = scaler.transform(input_features)

        # Realiza la predicción
        prediction = modelo.predict(scaled_features)
        probability = modelo.predict_proba(scaled_features)

        # Devuelve los resultados
        return {
            "prediction": int(prediction[0]),  # Resultado de la predicción
            "probability": probability.tolist()  # Probabilidades asociadas
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")
