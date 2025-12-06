# DualisAI-Back
🧠 Descripción del funcionamiento del Backend

El backend está desarrollado con FastAPI y expone una API REST responsable de procesar las imágenes enviadas desde el frontend, ejecutar la predicción con modelos de Inteligencia Artificial y almacenar resultados en una base de datos.

Su funcionamiento se divide en tres etapas principales:

1️⃣ Recepción de imágenes desde el frontend

El servidor recibe una imagen vía HTTP mediante una petición POST.
FastAPI la valida, la convierte a un formato procesable y la envía al servicio de predicción.

2️⃣ Inferencia con modelos de Deep Learning

El backend utiliza dos modelos especializados en TensorFlow/Keras:

📌 Modelo 1: Clasificación humana → Predice:

Sexo (Hombre / Mujer)

Rango de edad estimado

📌 Modelo 2: Clasificación animal → Detecta:

Si es Perro 🐶 o Gato 🐱

Probabilidad de predicción

El sistema también genera un mensaje aleatorio asociado al resultado detectado.

Todo el proceso se ejecuta dentro del servicio inference.py.

3️⃣ Almacenamiento y gestión de resultados

Cada predicción se guarda en MongoDB, incluyendo:

✔ Imagen procesada (o su referencia)
✔ Tipo de sujeto detectado
✔ Predicciones del modelo
✔ Confianza o rango de edad
✔ Mensaje generado
✔ Timestamp

Esto permite que el frontend pueda consultar un historial temporal mientras la sesión esté activa.

📌 Resumen operativo
Frontend (imagen) →
API FastAPI (procesa) →
Modelos IA predicen →
Se genera mensaje →
Se guarda en MongoDB →
Resultado devuelto como JSON →
Frontend visualiza

🛠Tecnologías principales del Backend
Componente	Tecnología
Framework API	FastAPI
IA / Inferencia	TensorFlow + Keras
Base de datos	MongoDB (Motor async)
Servidor ASGI	Uvicorn
Manejo de imágenes	Pillow