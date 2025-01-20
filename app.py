from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

# Cargar el modelo y el escalador
modelo = joblib.load("random_forest_model.pkl")
escalador = joblib.load("random_forest_scaler.pkl")

# Definir un modelo de entrada
class DatosEntrada(BaseModel):
    input: list

# Ruta principal para verificar que la API esté funcionando
@app.get("/")
def read_root():
    return {"mensaje": "¡API funcionando correctamente!"}

# Ruta para realizar predicciones
@app.post("/predict")
def predict(datos: DatosEntrada):
    # Convertir los datos a un formato adecuado
    entrada = np.array(datos.input).reshape(1, -1)
    entrada_escalada = escalador.transform(entrada)

    # Realizar la predicción
    resultado = modelo.predict(entrada_escalada)
    return {"prediccion": resultado.tolist()}
