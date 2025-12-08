# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.regularizers import l2

# # ================================
# # 1. Configuración
# # ================================
# data_dir = "app/neural_network/data/UTKFace"  # Ajusta ruta a tu dataset
# img_size = 128
# batch_size = 64
# epochs = 50
# model_path = "app/neural_network/models/utkface_model_best.keras"

# # ================================
# # 2. Cargar imágenes y etiquetas
# # ================================
# imagenes, edades, generos = [], [], []

# for file in os.listdir(data_dir):
#     if file.endswith(".jpg"):
#         try:
#             age, gender, _, _ = file.split("_")
#             age = int(age)
#             gender = int(gender)

#             img_path = os.path.join(data_dir, file)
#             img = load_img(img_path, target_size=(img_size, img_size))
#             img_array = img_to_array(img)
#             img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

#             imagenes.append(img_array)
#             edades.append(age)
#             generos.append(gender)
#         except:
#             continue

# imagenes = np.array(imagenes, dtype="float32")
# edades = np.array(edades, dtype="float32")
# generos = np.array(generos, dtype="float32")

# # Normalizamos la edad para mejorar estabilidad en entrenamiento
# max_age = edades.max()
# edades_norm = edades / max_age  # Escalamos 0-1

# # ================================
# # 3. Train-test split
# # ================================
# X_train, X_test, y_age_train, y_age_test, y_gen_train, y_gen_test = train_test_split(
#     imagenes, edades_norm, generos, test_size=0.2, random_state=42
# )

# # ================================
# # 4. Data augmentation
# # ================================

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.2),    # Aumentado a 20%
#     tf.keras.layers.RandomZoom(0.2),        # Aumentado a 20%
#     tf.keras.layers.RandomBrightness(factor=0.15), # Aumentado factor
#     tf.keras.layers.RandomContrast(0.1)     # ¡Añadido!
# ])

# # ================================
# # 5. Modelo base + Fine-tuning gradual
# # ================================
# base_model = MobileNetV2(
#     input_shape=(img_size, img_size, 3),
#     include_top=False,
#     weights="imagenet"
# )
# base_model.trainable = False  # Lo desbloqueamos luego

# inputs = tf.keras.Input(shape=(img_size, img_size, 3))
# x = data_augmentation(inputs)
# x = base_model(x, training=False)

# # Nuevo Dropout antes del pooling
# x = Dropout(0.2)(x) # Nuevo Dropout en características convolucionales
# x = GlobalAveragePooling2D()(x)
# x = BatchNormalization()(x)
# x = Dropout(0.4)(x) # Dropout reducido (antes era 0.5)

# age_output = Dense(1, activation="linear", name="age_output",
#                    kernel_regularizer=l2(0.001))(x) # <-- Regularización
# gender_output = Dense(1, activation="sigmoid", name="gender_output",
#                       kernel_regularizer=l2(0.001))(x) # <-- Regularización

# model = Model(inputs, [age_output, gender_output])

# # ================================
# # 6. Compilación inicial con Loss Weights
# # ================================
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss={"age_output": "mse", "gender_output": "binary_crossentropy"},
#     metrics={"age_output": "mae", "gender_output": "accuracy"},
#     loss_weights={"age_output": 2.0, "gender_output": 1.0} # Peso doble para la edad
# )

# # ================================
# # 7. Callbacks
# # ================================
# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
#     ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
# ]

# # ================================
# # 8. Entrenamiento inicial
# # ================================
# print("Entrenando capas superiores...")
# history = model.fit(
#     X_train, {"age_output": y_age_train, "gender_output": y_gen_train},
#     validation_data=(X_test, {"age_output": y_age_test, "gender_output": y_gen_test}),
#     batch_size=batch_size,
#     epochs=10,
#     callbacks=callbacks
# )

# # ================================
# # 9. Fine-tuning: desbloqueamos el modelo base
# # ================================

# print("Desbloqueando últimas capas del modelo base para fine-tuning...")

# # Desbloquea las últimas 30 capas (ejemplo, se puede experimentar)
# for layer in base_model.layers[-30:]:
#     if not isinstance(layer, BatchNormalization):
#         layer.trainable = True


# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss={"age_output": "mse", "gender_output": "binary_crossentropy"},
#     metrics={"age_output": "mae", "gender_output": "accuracy"},
#     loss_weights={"age_output": 2.0, "gender_output": 1.0} # Mantener pesos
# )

# history_finetune = model.fit(
#     X_train, {"age_output": y_age_train, "gender_output": y_gen_train},
#     validation_data=(X_test, {"age_output": y_age_test, "gender_output": y_gen_test}),
#     batch_size=batch_size,
#     epochs=epochs,
#     callbacks=callbacks
# )

# # ================================
# # 10. Evaluación final
# # ================================
# loss, age_loss, gender_loss, age_mae, gender_acc = model.evaluate(
#     X_test, {"age_output": y_age_test, "gender_output": y_gen_test}
# )

# # Reconstruimos la edad a valores reales
# age_mae_real = age_mae * max_age  

# print(f"MAE Edad: {age_mae_real:.2f}")
# print(f"Accuracy Género: {gender_acc:.2%}")

# # Guardamos modelo final
# model.save(model_path)
# print(f"Modelo guardado en {model_path}")