import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import threading
import queue

CAPTURE_THRESHOLD = 60.0
CAPTURE_DIR = "capturas"
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.60
CAMERA_INDEX = 0
FRAME_RESIZE_SCALE = 0.4

known_encodings = []
known_names = []
frame_queue = queue.Queue(maxsize=10)

os.makedirs(CAPTURE_DIR, exist_ok=True)

def align_face(image):
    """Alinea el rostro basándose en la posición de los ojos"""
    try:
        landmarks_list = face_recognition.face_landmarks(image)
        if not landmarks_list:
            return image

        landmarks = landmarks_list[0]
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return image

        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)

        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        eyes_center = tuple(map(int, np.mean([left_center, right_center], axis=0)))
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

        return aligned
    except Exception:
        return image


def estimate_eye_color(image, eye_points):
    """Estima el color de ojos basándose en los landmarks"""
    try:
        if not eye_points:
            return "desconocido"
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        eye_array = np.array(eye_points, dtype=np.int32)
        cv2.fillPoly(mask, [eye_array], 255)
        
        mean_color = cv2.mean(image, mask=mask)[:3]
        b, g, r = [int(c) for c in mean_color]

        if b > 90 and g > 90 and r < 80:
            return "azules"
        elif g > 80 and r > 70:
            return "verdes"
        elif r > 80 and g > 60:
            return "avellana"
        else:
            return "cafés"
    except Exception:
        return "desconocido"


def load_known_faces(known_faces_dir=None):
    """Carga y procesa las imágenes de rostros conocidos"""
    global known_encodings, known_names
    
    if known_faces_dir is None:
        known_faces_dir = KNOWN_FACES_DIR
    
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
                image = align_face(image)
                faces = face_recognition.face_encodings(image)
                
                if faces:
                    encodings.append(faces[0])
                    names.append(os.path.splitext(filename)[0])
                    print(f"✅ {filename} → {names[-1]}")
                else:
                    print(f"⚠️ No se detectó rostro en: {filename}")
            except Exception as e:
                print(f"❌ Error en {filename}: {e}")
    
    print(f"📊 Total rostros cargados: {len(encodings)}\n")

    known_encodings = encodings
    known_names = names
    
    return encodings, names


def initialize_camera(index=None):
    """Inicializa la cámara con configuraciones optimizadas"""
    if index is None:
        index = CAMERA_INDEX
        
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print("❌ No se pudo acceder a la cámara")
        return None

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✅ Cámara inicializada correctamente")
    return cam


def recognize_faces_safe(frame, known_encodings_param=None, known_names_param=None, tolerance=None):
    """
    Versión segura del reconocimiento facial (mantiene compatibilidad con código existente)
    Compatible con la función original recognize_faces_safe
    """
    global known_encodings, known_names

    if known_encodings_param is None:
        known_encodings_param = known_encodings
    if known_names_param is None:
        known_names_param = known_names
    if tolerance is None:
        tolerance = TOLERANCE
    
    try:
        small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        landmarks_list = face_recognition.face_landmarks(rgb, locations)
        
        results = []
        
        for i, (loc, enc) in enumerate(zip(locations, encodings)):
            top, right, bottom, left = [int(v / FRAME_RESIZE_SCALE) for v in [loc[0], loc[1], loc[2], loc[3]]]
            
            name = "Desconocido"
            confidence = 0.0
            eye_color = "desconocido"
            landmarks = {}

            if known_encodings_param:
                distances = face_recognition.face_distance(known_encodings_param, enc)
                best_match = np.argmin(distances)
                confidence = max(0, (1 - distances[best_match]) * 100)
                
                if confidence >= 50.0:
                    name = known_names_param[best_match]

            if i < len(landmarks_list):
                landmarks = landmarks_list[i]
                if 'left_eye' in landmarks:
                    scaled_eye = [(int(p[0] / FRAME_RESIZE_SCALE), int(p[1] / FRAME_RESIZE_SCALE)) 
                                 for p in landmarks['left_eye']]
                    eye_color = estimate_eye_color(frame, scaled_eye)
            
            results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence,
                'landmarks': landmarks,
                'eye_color': eye_color
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error en reconocimiento: {e}")
        return []


def recognize_faces_threaded(frame, known_encodings_param=None, known_names_param=None, tolerance=None):
    """Alias para compatibilidad con versión threaded"""
    return recognize_faces_safe(frame, known_encodings_param, known_names_param, tolerance)


def draw_results(frame, results):
    """Dibuja los resultados del reconocimiento en el frame"""
    if not results:
        return
        
    for result in results:
        top, right, bottom, left = result['location']
        name = result['name']
        confidence = result['confidence']
        eye_color = result.get('eye_color', 'desconocido')
        landmarks = result.get('landmarks', {})

        if name != "Desconocido":
            label = f"{name} ({confidence:.1f}%)"
            if eye_color != "desconocido":
                label += f" - Ojos: {eye_color}"
        else:
            label = "Desconocido"

        color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
        thickness = 4 if confidence >= CAPTURE_THRESHOLD else 2

        cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]

        cv2.rectangle(frame, (left - 10, bottom + 10), 
                     (left + text_size[0] + 20, bottom + 45), color, cv2.FILLED)

        cv2.putText(frame, label, (left + 5, bottom + 35), 
                   font, font_scale, (255, 255, 255), text_thickness)

        if landmarks:
            for part_name, points in landmarks.items():
                color_landmarks = (200, 200, 200) if part_name in ['left_eye', 'right_eye'] else (150, 150, 150)
                for (x, y) in points:
                    scaled_x = int(x / FRAME_RESIZE_SCALE)
                    scaled_y = int(y / FRAME_RESIZE_SCALE)
                    cv2.circle(frame, (scaled_x, scaled_y), 1, color_landmarks, -1)


def save_capture(name, live_frame, capture_dir=None):
    """Guarda una captura combinada con la imagen de referencia"""
    if capture_dir is None:
        capture_dir = CAPTURE_DIR
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{name}_{timestamp}.jpg"
    combined_path = os.path.join(capture_dir, output_filename)

    known_image_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        temp_path = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
        if os.path.exists(temp_path):
            known_image_path = temp_path
            break

    if not known_image_path:
        print(f"⚠️ Imagen de referencia no encontrada para {name}")
        return False

    try:
        known_image = cv2.imread(known_image_path)
        if known_image is None:
            print(f"❌ Error al leer {known_image_path}")
            return False

        height = 300
        known_resized = cv2.resize(known_image, 
                                 (int(known_image.shape[1] * height / known_image.shape[0]), height))
        live_resized = cv2.resize(live_frame, 
                                (int(live_frame.shape[1] * height / live_frame.shape[0]), height))

        min_width = min(known_resized.shape[1], live_resized.shape[1])
        known_resized = known_resized[:, :min_width]
        live_resized = live_resized[:, :min_width]

        combined = np.hstack((known_resized, live_resized))

        label = f"Coincidencia: {name}"
        cv2.putText(combined, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(combined_path, combined)
        print(f"📸 Captura combinada guardada: {combined_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar captura: {e}")
        return False


def get_known_faces_info():
    """Retorna información sobre los rostros conocidos cargados"""
    global known_encodings, known_names
    return {
        'count': len(known_encodings),
        'names': known_names.copy()
    }


def set_capture_threshold(threshold):
    """Establece el umbral de captura"""
    global CAPTURE_THRESHOLD
    CAPTURE_THRESHOLD = max(0, min(100, threshold))
    return CAPTURE_THRESHOLD


def get_capture_threshold():
    """Obtiene el umbral de captura actual"""
    return CAPTURE_THRESHOLD


def process_frame_for_web(frame):
    """Procesa un frame para uso en aplicaciones web"""
    results = recognize_faces_safe(frame)
    draw_results(frame, results)
    return frame, results


def encode_frame_to_base64(frame):
    """Codifica un frame a base64 para transmisión web"""
    import base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64


def configure_for_flask(config_dict):
    """Configura el módulo con un diccionario de configuración"""
    global CAPTURE_THRESHOLD, CAPTURE_DIR, KNOWN_FACES_DIR, TOLERANCE, CAMERA_INDEX, FRAME_RESIZE_SCALE
    
    CAPTURE_THRESHOLD = config_dict.get('CAPTURE_THRESHOLD', CAPTURE_THRESHOLD)
    CAPTURE_DIR = config_dict.get('CAPTURE_DIR', CAPTURE_DIR)
    KNOWN_FACES_DIR = config_dict.get('KNOWN_FACES_DIR', KNOWN_FACES_DIR)
    TOLERANCE = config_dict.get('TOLERANCE', TOLERANCE)
    CAMERA_INDEX = config_dict.get('CAMERA_INDEX', CAMERA_INDEX)
    FRAME_RESIZE_SCALE = config_dict.get('FRAME_RESIZE_SCALE', FRAME_RESIZE_SCALE)

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

if __name__ == "__main__":
    print("🔍 Módulo de reconocimiento facial cargado")
    print("📁 Cargando rostros conocidos...")
    load_known_faces()
    print(f"✅ Módulo listo - {len(known_encodings)} rostros cargados")