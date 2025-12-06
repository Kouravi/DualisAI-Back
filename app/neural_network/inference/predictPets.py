import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array


# Cargar modelo entrenado

modelo = tf.keras.models.load_model("neural_network/models/dog-cat-mnv2.keras")
ruta_imagen = "neural_network/imagenes_pt/perro5.jpg"

# Función para cargar y preparar imagen

def preparar_imagen(ruta_imagen, img_size=224):
    """
    Carga una imagen y la prepara para el modelo entrenado.
    - ruta_imagen: Ruta a la imagen (str)
    - img_size: Tamaño al que se redimensiona la imagen
    """
    img = load_img(ruta_imagen, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalización MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)  # Batch de 1 imagen
    return img_array, img


# Función para predecir y mostrar resultado

def predecir_imagen(modelo, ruta_imagen):

    imagen_preprocesada, img_original = preparar_imagen(ruta_imagen)

    # Predicción: valores cercanos a 0 -> gato, valores cercanos a 1 -> perro
    prediccion = modelo.predict(imagen_preprocesada)
    etiqueta = "Perro 🐶" if prediccion[0][0] >= 0.5 else "Gato 🐱"
    probabilidad = prediccion[0][0] if etiqueta == "Perro 🐶" else 1 - prediccion[0][0]

    # Mostrar imagen con etiqueta y probabilidad
    plt.imshow(img_original)
    plt.title(f"Predicción: {etiqueta} ({probabilidad:.2%})")
    plt.axis("off")
    plt.show()

    return etiqueta, probabilidad

# Ruta y pruebas

resultado, prob = predecir_imagen(modelo, ruta_imagen)
print(f"Resultado: {resultado} - Confianza: {prob:.2%}")