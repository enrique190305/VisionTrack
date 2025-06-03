import cv2
import face_recognition
import os
import sys
import numpy as np
from datetime import datetime

CAPTURE_THRESHOLD = 70.0
CAPTURE_DIR = "capturas"
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.65
CAMERA_INDEX = 0
FRAME_RESIZE_SCALE = 0.25

os.makedirs(CAPTURE_DIR, exist_ok=True)


def load_known_faces(known_faces_dir):
    if not os.path.exists(known_faces_dir):
        print(f"❌ Carpeta no encontrada: '{known_faces_dir}'")
        return [], []

    encodings, names = [], []
    print(f"📁 Cargando rostros desde: {known_faces_dir}")
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(path)
                faces = face_recognition.face_encodings(image)
                if faces:
                    encodings.append(faces[0])
                    names.append(os.path.splitext(filename)[0])
                    print(f"✅ {filename} → {names[-1]}")
            except Exception as e:
                print(f"❌ Error en {filename}: {e}")
    print(f"📊 Total rostros cargados: {len(encodings)}\n")
    return encodings, names


def initialize_camera(index=0):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print("❌ No se pudo acceder a la cámara")
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


def recognize_faces_safe(frame, known_encodings, known_names, tolerance):
    small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []

    for loc, enc in zip(locations, encodings):
        top, right, bottom, left = [int(v / FRAME_RESIZE_SCALE) for v in [loc[0], loc[1], loc[2], loc[3]]]

        name = "Desconocido"
        confidence = 0.0

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_match = np.argmin(distances)
            confidence = max(0, (1 - distances[best_match]) * 100)

            if confidence >= 50.0:
                name = known_names[best_match]

        results.append({
            'location': (top, right, bottom, left),
            'name': name,
            'confidence': confidence
        })

    return results


def draw_results(frame, results):
    for result in results:
        top, right, bottom, left = result['location']
        name = result['name']
        confidence = result['confidence']
        label = f"{name} ({confidence:.1f}%)" if name != "Desconocido" else "Desconocido"

        color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
        thickness = 5 if confidence >= CAPTURE_THRESHOLD else 3

        cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), color, thickness)
        cv2.rectangle(frame, (left - 10, bottom + 10), (right + 10, bottom + 45), color, cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 8, bottom + 35), font, 0.8, (255, 255, 255), 2)


def save_capture(name, live_frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{name}_{timestamp}.jpg"
    combined_path = os.path.join(CAPTURE_DIR, output_filename)

    known_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if not os.path.exists(known_image_path):
        known_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.png")
        if not os.path.exists(known_image_path):
            print(f"⚠️ Imagen de referencia no encontrada para {name}")
            return

    known_image = cv2.imread(known_image_path)
    if known_image is None:
        print(f"❌ Error al leer {known_image_path}")
        return

    height = 300
    known_resized = cv2.resize(known_image, (int(known_image.shape[1] * height / known_image.shape[0]), height))
    live_resized = cv2.resize(live_frame, (int(live_frame.shape[1] * height / live_frame.shape[0]), height))

    min_width = min(known_resized.shape[1], live_resized.shape[1])
    known_resized = known_resized[:, :min_width]
    live_resized = live_resized[:, :min_width]

    combined = np.hstack((known_resized, live_resized))

    cv2.imwrite(combined_path, combined)
    print(f"📸 Captura combinada guardada: {combined_path}")


def main():
    print("🔍 Iniciando sistema de reconocimiento facial")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    if not known_encodings:
        print("⚠️ No hay rostros conocidos cargados.")
        cont = input("¿Continuar de todas formas? (s/N): ").strip().lower()
        if cont != 's':
            return

    cam = initialize_camera(CAMERA_INDEX)
    if not cam:
        return

    print("🎥 Cámara activa. Presiona 'q' para salir.")
    process_frame = True
    already_captured = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara")
            break

        if process_frame:
            results = recognize_faces_safe(frame, known_encodings, known_names, TOLERANCE)
            draw_results(frame, results)

            for res in results:
                name = res['name']
                confidence = res['confidence']
                if name != "Desconocido" and confidence >= CAPTURE_THRESHOLD:
                    if name not in already_captured:
                        save_capture(name, frame)
                        already_captured.add(name)

        process_frame = not process_frame  

        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("🧹 Finalizado y recursos liberados")


if __name__ == "__main__":
    try:
        import face_recognition
        import cv2
        import numpy as np
    except ImportError:
        print("❌ Faltan dependencias. Ejecuta: pip install opencv-python face-recognition numpy")
        sys.exit(1)

    main()
