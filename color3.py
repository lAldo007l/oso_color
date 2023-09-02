import cv2
import numpy as np
import pyttsx3
import time
import threading

# Define los umbrales de color para cada color
umbrales_de_color = {
    "verde": ((35, 100, 100), (85, 255, 255)),
    "amarillo": ((20, 100, 100), (35, 255, 255)),
    "naranja": ((5, 100, 100), (20, 255, 255)),
    "rojo": ((0, 100, 100), (5, 255, 255)),
    "morado": ((125, 100, 100), (140, 255, 255)),
    "azul": ((100, 100, 20), (125, 255, 255)),
}

# Inicializa el motor de texto a voz
engine = pyttsx3.init()

# Variable para rastrear el estado del anuncio del color
color_anunciado = ""

# Importa el clasificador de detección facial de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño y posición de la región de reconocimiento
region_size = 150  # Tamaño en píxeles del cuadrado
center_x = None  # Coordenada x del centro de la cámara
center_y = None  # Coordenada y del centro de la cámara

# Último tiempo en el que se detectó un color
ultimo_tiempo_color = time.time()

# Calcula el centro de la cámara
def calcular_centro_camara(frame):
    height, width, _ = frame.shape
    return width // 2, height // 2
    

# Función para detectar color y rostro
def detectar_color_y_rostro(frame):
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
    
    # Detección facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return color_detectado

# Configuración de la cámara
cap = cv2.VideoCapture(0)

def reproducir_audio(texto):
    engine.say(texto)
    engine.runAndWait()
    
while True:
    ret, frame = cap.read()
    
    if ret:
        if center_x is None or center_y is None:
            center_x, center_y = calcular_centro_camara(frame)
        
        # Dibuja la región de reconocimiento en el centro
        x1 = center_x - region_size // 2
        y1 = center_y - region_size // 2
        x2 = x1 + region_size
        y2 = y1 + region_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        color_detectado = detectar_color_y_rostro(frame)
        
        if(color_detectado is not None):
            print(color_detectado)
            # Pronuncia el color detectado
            # engine.say("El color detectado es: {}".format(color_detectado))
            # engine.runAndWait()
            audio_thread = threading.Thread(target=reproducir_audio, args=('El color detectado es:' + color_detectado,))
            audio_thread.start()
            color_anunciado = color_detectado

        cv2.imshow('frame', frame)
        
        # Captura durante 5 segundos enfocando objetos en la región de reconocimiento
        if cv2.waitKey(1) & 0xFF == ord('s'):
            time.sleep(10)
    
    else:
        break

cap.release()
cv2.destroyAllWindows()
