from flask import Blueprint, render_template, Response, jsonify
import cv2
import os
from . import facial_bp
from .reco_facial import load_known_faces, recognize_faces_safe, draw_results, initialize_camera

# Definir la ruta absoluta a known_faces
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'known_faces')
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'captures')

# Asegurar que las carpetas existan
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Variables globales
video_capture = initialize_camera()
known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Procesar el frame
        results = recognize_faces_safe(frame, known_face_encodings, known_face_names, 0.65)
        draw_results(frame, results)
        
        # Convertir el frame para streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@facial_bp.route('/')
def facial_index():
    return render_template('facial/index.html')

@facial_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@facial_bp.route('/status')
def get_status():
    return jsonify({
        'known_faces': len(known_face_names),
        'faces_loaded': known_face_names
    })