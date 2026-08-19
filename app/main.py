from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# =========================================================
# PATHS
# =========================================================

# Student_performance/
BASE_DIR = Path(__file__).resolve().parent.parent

# Student_performance/model.pkl
MODEL_PATH = BASE_DIR / "Data_analaysis" / "model.pkl"


# Student_performance/app/static/templates/
TEMPLATE_DIR = Path(__file__).resolve().parent / "static" / "templates"


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Student Performance Predictor",
    description="AI-powered mathematics score prediction",
    version="1.0.0"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# =========================================================
# INPUT MODEL
# =========================================================

class StudentData(BaseModel):

    gender: str

    race_ethnicity: str

    parental_level_of_education: str

    lunch: str

    test_preparation_course: str

    reading_score: float

    writing_score: float


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(data: StudentData):

    input_data = pd.DataFrame([{

        "gender": data.gender,

        "race_ethnicity":
            data.race_ethnicity,

        "parental_level_of_education":
            data.parental_level_of_education,

        "lunch":
            data.lunch,

        "test_preparation_course":
            data.test_preparation_course,

        "reading_score":
            data.reading_score,

        "writing_score":
            data.writing_score

    }])


    prediction = model.predict(input_data)

    predicted_score = float(prediction[0])


    # Keep prediction within 0-100

    predicted_score = max(
        0,
        min(100, predicted_score)
    )


    return {
        "predicted_math_score": round(
            predicted_score,
            2
        )
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }
