import threading
import cv2
import time
import os

from DeteccionColor import procesar_imagen, calcular_centro_camara, detectar_color, reproducir_audio

# Variable para rastrear el estado del anuncio del color
color_anunciado = ""

# Importa el clasificador de detección facial de OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño y posición de la región de reconocimiento
region_size = 150  # Tamaño en píxeles del cuadrado
center_x = None  # Coordenada x del centro de la cámara
center_y = None  # Coordenada y del centro de la cámara

# Calcula el centro de la cámara

# Configuración de la cámara
cap = cv2.VideoCapture(0)

ids_detectados = procesar_imagen()
os.remove('imagen.jpg')
print(ids_detectados)
# if ids_detectados != 0:

if ids_detectados != 0:
    audio_thread = threading.Thread(target=reproducir_audio, args=('Bienvenido, hora de aventura...',))
    audio_thread.start()
while True:
    # Ejemplo de uso:
    ret, frame = cap.read()
    if ret and ids_detectados != 0:
        if center_x is None or center_y is None:
            center_x, center_y = calcular_centro_camara(frame)

        # Dibuja la región de reconocimiento en el centro
        x1 = center_x - region_size // 2
        y1 = center_y - region_size // 2
        x2 = x1 + region_size
        y2 = y1 + region_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        color_detectado = detectar_color(frame)

        if (color_detectado is not None):
            print(color_detectado)
            # Pronuncia el color detectado
            # engine.say("El color detectado es: {}".format(color_detectado))
            # engine.runAndWait()
            audio_thread = threading.Thread(target=reproducir_audio, args=('El color detectado es: ' + color_detectado,))
            audio_thread.start()
            color_anunciado = color_detectado

        cv2.imshow('frame', frame)

        # Captura durante 5 segundos enfocando objetos en la región de reconocimiento
        if cv2.waitKey(1) & 0xFF == ord('s'):
            time.sleep(100)

    else:
        break

cap.release()
cv2.destroyAllWindows()
