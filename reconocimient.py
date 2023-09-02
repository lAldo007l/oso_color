import dlib
import cv2
import numpy as np

humbral_similitud = 50

detector_rostro = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "Modelos/shape_predictor_68_face_landmarks.dat")

rostros_conocidos = {}


def detect_faces(image):
    imagen_grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector_rostro(imagen_grises, 0)
    detected_faces = []

    for (i, rect) in enumerate(rects):
        shape = predictor(imagen_grises, rect)
        shape = face_utils.shape_to_np(shape)
        detected_faces.append(shape)

    return detected_faces


def register_face(detected_face):
    promedio = np.mean(detected_face, axis=0, dtype=int)

    rostro_conocido = False
    id_actual = -1

    for id_rostro, datos_rostro in rostros_conocidos.items():
        puntos_rostro, pos_promedio = datos_rostro
        distancia = np.linalg.norm(pos_promedio - promedio)

        if distancia < humbral_similitud:
            id_actual = id_rostro
            rostro_conocido = True
            break

    if not rostro_conocido:
        id_actual = max(rostros_conocidos.keys(), default=0) + 1
        rostros_conocidos[id_actual] = (detected_face, promedio)

    return id_actual


def procesar_imagen(ruta_imagen):
    image = cv2.imread(ruta_imagen)
    detected_faces = detect_faces(image)
    ids_detectados = []

    for detected_face in detected_faces:
        id_actual = register_face(detected_face)
        ids_detectados.append(id_actual)

        shape = detected_face
        (x, y, w, h) = cv2.boundingRect(np.array([shape]))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(image, f'ID: {id_actual}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return ids_detectados

# Para usar esta base de reconocimiento facial, puedes llamar a la función `procesar_imagen(ruta_imagen)`
# con la ruta de la imagen que deseas procesar. Devolverá una lista de IDs de rostros detectados.


# Ejemplo de uso:
ruta_imagen_ejemplo = "foto.jpeg"  # Ruta a la imagen que deseas procesar
ids_detectados = procesar_imagen(ruta_imagen_ejemplo)
print("IDs detectados:", ids_detectados)
