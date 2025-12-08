'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Parámetros base

TAMANO_IMG = 224   # MobileNetV2 funciona bien con 224x224
TAMANO_LOTE = 32
EPOCAS = 15
model_path = "neural_network/models/dog-cat-mnv2.keras"

# Cargar dataset

carpeta_base = "neural_network/data/cats_and_dogs_filtered"
carpeta_entrenamiento = os.path.join(carpeta_base, "train")
carpeta_validacion = os.path.join(carpeta_base, "validation")

train_ds = tf.keras.utils.image_dataset_from_directory(
    carpeta_entrenamiento,
    image_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=TAMANO_LOTE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    carpeta_validacion,
    image_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=TAMANO_LOTE,
    label_mode="int"
)


# Preprocesamiento

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Normalización automática de MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))


# Modelo con MobileNetV2
# Cargamos MobileNetV2 sin la última capa, con pesos preentrenados en ImageNet

base_model = tf.keras.applications.MobileNetV2(input_shape=(TAMANO_IMG, TAMANO_IMG, 3),
                                               include_top=False,
                                               weights="imagenet")
base_model.trainable = False  # Congelamos capas para entrenamiento inicial

# Añadimos capas finales
modelo = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binaria: gato o perro
])


# Compilación

modelo.compile(optimizer="adam",
               loss="binary_crossentropy",
               metrics=["accuracy"])

# Entrenamiento

history = modelo.fit(train_ds,
                     validation_data=val_ds,
                     epochs=EPOCAS)

# Fine-tuning opcional
# Descongelamos capas finales para afinar el modelo

base_model.trainable = True
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Learning rate bajo
               loss="binary_crossentropy",
               metrics=["accuracy"])

history_fine = modelo.fit(train_ds,
                          validation_data=val_ds,
                          epochs=5)


# Guardar modelo

modelo.save(model_path)
print(f"Modelo guardado en {model_path}")

# Gráficas

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epocas_total = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epocas_total, acc, label='Entrenamiento')
plt.plot(epocas_total, val_acc, label='Validación')
plt.legend()
plt.title('Precisión')

plt.subplot(1, 2, 2)
plt.plot(epocas_total, loss, label='Entrenamiento')
plt.plot(epocas_total, val_loss, label='Validación')
plt.legend()
plt.title('Pérdida')
plt.show()
'''