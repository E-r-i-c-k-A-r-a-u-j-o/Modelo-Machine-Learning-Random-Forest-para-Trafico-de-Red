from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from joblib import load
from fastapi.templating import Jinja2Templates

# Inicializa FastAPI
app = FastAPI()

# Configura Jinja2 para renderizar plantillas HTML
templates = Jinja2Templates(directory="templates")

# Cargar modelo y escalador
modelo = load("random_forest_model.pkl")
scaler = load("random_forest_scaler.pkl")

class InputData(BaseModel):
    features: list[float]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
def predict(data: InputData):
    input_features = np.array(data.features).reshape(1, -1)
    scaled_features = scaler.transform(input_features)
    prediction = modelo.predict(scaled_features)
    probability = modelo.predict_proba(scaled_features)
    return {"prediction": int(prediction[0]), "probability": probability.tolist()}

