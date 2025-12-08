import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ================================
# 1. Cargar modelo entrenado
# ================================
model_path = "app/neural_network/models/utkface_model_best.keras"
model = load_model(model_path)
img_path = "app/neural_network/imagenes_pt/mujer5.jpg"  # <- pon aquí la ruta a tu imagen

# ================================
# 2. Función para predecir una imagen
# ================================
def procesar_imagen(img_path, img_size=128):  
    # Preprocesar la imagen
    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # batch de 1

    return img_array, img

# ================================
# 3. Ejemplo de uso
# ================================


def predecir_imagen(model, img_path, max_age=116):

    imagen_preprocesada, img_original = procesar_imagen(img_path)

    # Hacer predicción
    age_pred, gender_pred = model.predict(imagen_preprocesada)

    # Reconstruir edad real
    edad_real = age_pred[0][0] * max_age
    genero_pred = "Hombre" if gender_pred[0][0] < 0.5 else "Mujer"

    # Calcular rango de edad estimado
    edad_min = max(0, edad_real - 10)  # no permitir valores negativos
    edad_max = max(0, edad_real - 5)

    # Mostrar imagen con etiqueta y probabilidad
    plt.imshow(img_original)
    plt.title(f"Género: {genero_pred} - Edad: entre {edad_min:.0f} y {edad_max:.0f} años")
    plt.axis("off")
    plt.show()

    return edad_min, edad_max, genero_pred

# Ruta y pruebas

edad_min, edad_max, genero = predecir_imagen(model, img_path)
print(f"Género: {genero} - Edad estimada: entre {edad_min:.0f} y {edad_max:.0f} años")