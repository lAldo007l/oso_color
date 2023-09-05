import dlib
import cv2
from imutils import face_utils

humbral_similitud = 200

detector_rostro = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

rostros_conocidos = {}


def take_photo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        exit()
        
    ret, frame = cap.read()

    if not ret:
        print("No se pudo capturar un cuadro de la cámara.")
        exit()

    archivo_salida = "imagen.jpg"
    cv2.imwrite(archivo_salida, frame)


    cap.release()
    cv2.destroyAllWindows()

    print(f"Imagen guardada como '{archivo_salida}'.")


def detect_faces(image):
    imagen_grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector_rostro(imagen_grises, 0)
    detected_faces = []

    for (i, rect) in enumerate(rects):
        shape = predictor(imagen_grises, rect)
        shape = face_utils.shape_to_np(shape)
        detected_faces.append(shape)

    return detected_faces


# def base64ToImage(base64_string):
#     image_bytes = base64.b64decode(base64_string)
#     image = Image.open(BytesIO(image_bytes))
#     image.save("foto.jpeg")


def CrearImagen():
    # base64ToImage(base64_string)
    image = cv2.imread("imagen.jpg")
    detected_faces = detect_faces(image)
    return detected_faces


def procesar_imagen():
    take_photo()
    detected_faces = CrearImagen()
    cantidadRostros = (len(detected_faces))
    return cantidadRostros