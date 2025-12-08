import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # sigue siendo útil para el humano
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from app.neural_network.messages import get_random_message
from app.neural_network.model_defs import build_dog_cat_model

MODEL_HUMANO_PATH = os.getenv(
    "MODEL_HUMANO_PATH",
    "app/neural_network/models/utkface_model_best.keras"
)
PETS_WEIGHTS_PATH = os.getenv(
    "MODEL_MASCOTA_WEIGHTS",
    "app/neural_network/models/dog-cat-mnv2.weights.h5"
)

# Humano: si este también te da guerra, podemos hacer lo mismo de pesos luego
model_humano = load_model(MODEL_HUMANO_PATH, compile=False)

# Mascota: reconstruye arquitectura y carga SOLO pesos
model_mascota = build_dog_cat_model(img_size=224)
model_mascota.load_weights(PETS_WEIGHTS_PATH)

MAX_AGE = int(os.getenv("MAX_AGE", "116"))

def preprocess_image(image: Image.Image, size=128):
    """Preprocesamiento genérico"""
    image = image.convert("RGB")
    image = image.resize((size, size))
    img_array = img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_image(image: Image.Image, tipo_modelo: str):
    """Predice según el tipo de modelo indicado"""

    if tipo_modelo == "humano":
        img_pre = preprocess_image(image, size=128)
        age_pred, gender_pred = model_humano.predict(img_pre)

        # Edad estimada en años
        edad_real = age_pred[0][0] * MAX_AGE
        genero_pred = "Observo un Hombre" if gender_pred[0][0] < 0.5 else "Observo una Mujer"

        edad_min = max(0, edad_real - 10)
        edad_max = max(0, edad_real - 2)

        mensaje = get_random_message(genero_pred)

        return {
            "tipo": "Humano",
            "genero": genero_pred,
            "edad_estimacion": f"De edad entre {edad_min:.0f} y {edad_max:.0f} años",
            "mensaje": mensaje
        }

    elif tipo_modelo == "mascota":
        img_pre = preprocess_image(image, size=224)
        pred = model_mascota.predict(img_pre)

        # pred[0][0] ~ 0 → gato, ~1 → perro
        es_perro = pred[0][0] >= 0.5
        etiqueta = "Observo un Perro 🐶" if es_perro else "Observo un Gato 🐱"
        probabilidad = float(pred[0][0]) if es_perro else float(1 - pred[0][0])

        mensaje = get_random_message(etiqueta)

        return {
            "tipo": "Mascota",
            "especie": etiqueta,
            "confianza": f"{probabilidad:.2%}",
            "mensaje": mensaje
        }

    else:
        return {
            "error": "Tipo de modelo no soportado. Usa 'humano' o 'mascota'."
        }