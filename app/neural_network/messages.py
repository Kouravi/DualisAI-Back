import random  # 👈 para elegir mensajes aleatorios

# ============================
# Mensajes por categoría
# ============================

MENSAJES_HOMBRE = [
    "Persona de mente brillante!",
    "Su valor es admirable!",
    "Parece muy ingenioso!",
    "Su presencia irradia calma!",
    "Ser inspirador es su don!",
    "Parece ser un caballero!"
]

MENSAJES_MUJER = [
    "Que sonrisa tan iluminada!",
    "Su empatía es única!",
    "Es increíblemente fuerte!",
    "De un estilo es impecable!",
    "Brilla con luz propia!",
    "Qué energía tan bella!",
    "Simplemente cautivadora!"
]

MENSAJES_PERRO = [
    "Eres el mejor amigo del ser humano!",
    "Tu cola es pura alegría!",
    "El más leal!",
    "Eres un héroe peludo!",
    "Eres tan noble como un rey!",
    "¡Qué adorable mirada tienes!",
    "Eres un perro perfecto!"
]

MENSAJES_GATO = [
    "Tus ronroneos son magia!",
    "Eres un cazador elegante!",
    "Ojos de ensueño!",
    "Eres puro misterio!",
    "Maestro de la siesta!",
    "Tu pelaje es un tesoro!",
    "Simplemente majestuoso!"
]

# Mapa de categoría → lista de mensajes
MENSAJES_POR_TIPO = {
    "Observo un Hombre": MENSAJES_HOMBRE,
    "Observo una Mujer": MENSAJES_MUJER,
    "Observo un Perro 🐶": MENSAJES_PERRO,
    "Observo un Gato 🐱": MENSAJES_GATO,
}

def get_random_message(tipo: str) -> str:
    """Devuelve un mensaje aleatorio según el tipo (Hombre, Mujer, Perro 🐶, Gato 🐱)."""
    mensajes = MENSAJES_POR_TIPO.get(tipo)
    if not mensajes:
        # Mensaje por defecto en caso de que falte el tipo
        return "¡Qué genial te ves hoy!"
    return random.choice(mensajes)