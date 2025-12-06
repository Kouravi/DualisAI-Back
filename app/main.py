from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router

app = FastAPI(
    title="AI Prediction API",
    description="API para predicción de edad/género humano y detección de perros/gatos 🧠",
    version="2.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(predict_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de predicción IA 🎯"}