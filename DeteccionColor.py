import cv2
import numpy as np
import time
import pyttsx3

from ReconocimientoFacial import procesar_imagen

# Último tiempo en el que se detectó un color
ultimo_tiempo_color = time.time()

# Define los umbrales de color para cada color
umbrales_de_color = {
    "verde": ((35, 100, 100), (85, 255, 255)),
    "amarillo": ((20, 100, 100), (35, 255, 255)),
    # "negro": ((0, 0, 0), (180, 255, 30)),
    # "blanco": ((0, 0, 200), (180, 30, 255)),
    # "naranja": ((5, 100, 100), (20, 255, 255)),
    "rojo": ((0, 100, 100), (5, 255, 255)),
    # "morado": ((125, 100, 100), (140, 255, 255)),
    "azul": ((100, 100, 20), (125, 255, 255)),
}

# Inicializa el motor de texto a voz
engine = pyttsx3.init()

def calcular_centro_camara(frame):
    height, width, _ = frame.shape
    return width // 2, height // 2


# Función para detectar color y rostro
def detectar_color(frame):
    global ultimo_tiempo_color

    tiempo_actual = time.time()
    if tiempo_actual - ultimo_tiempo_color < 3:
        return None

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_detectado = None
    for color, (umbral_bajo, umbral_alto) in umbrales_de_color.items():
        mask = cv2.inRange(frameHSV, umbral_bajo, umbral_alto)
        area = cv2.countNonZero(mask)
        if area > 3000:
            color_detectado = color
            ultimo_tiempo_color = tiempo_actual
            break

    # # Detección facial
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(
    #     gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return color_detectado

def reproducir_audio(texto):
    try:
        engine.say(texto)
        engine.runAndWait()
    except Exception as Error:
        print(Error)
    
