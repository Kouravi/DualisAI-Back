from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.inference import predict_image
# from app.services.database import insert_prediction, get_all_predictions
from datetime import datetime
import io
from PIL import Image

router = APIRouter(prefix="/predict", tags=["Predicción"])

@router.post("/")

async def predict_endpoint(
    tipo_modelo: str = Form(..., description="Tipo de modelo: humano o mascota"),
    file: UploadFile = File(...)
):
    try:
        if tipo_modelo not in ["humano", "mascota"]:
            raise HTTPException(status_code=400, detail="El tipo de modelo debe ser 'humano' o 'mascota'.")

        if file.content_type is not None and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Predicción según modelo
        resultado = predict_image(image, tipo_modelo)

        # # Preparar datos para guardar en la base de datos
        # prediction_data = {
        #     "filename": file.filename,
        #     "tipo_modelo": tipo_modelo,
        #     "resultado": resultado,
        #     # Use current UTC datetime instance then isoformat()
        #     "timestamp": datetime.utcnow().isoformat()
        # }

        # await insert_prediction(prediction_data)

        return {"success": True, "prediccion": resultado}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# """Listar todas las predicciones guardadas en MongoDB"""
# @router.get("/all")
# async def get_all():
   
#     registros = await get_all_predictions()
#     return {"total": len(registros), "predicciones": registros}

@router.get("/all")
async def get_all():
    """
    Antes devolvía todos los registros de MongoDB.
    Si ya no usas historial, puedes:
    - devolver lista vacía (mejor para deploy)
    """
    return {"total": 0, "predicciones": []}