from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import face_recognition
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
SIMILARITY_THRESHOLD = 80.0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_video_file(file):
    """Validate uploaded video file."""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, "File is valid"

def cleanup_old_files():
    """Remove old video files to save space."""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                # Remove files older than 1 hour
                if os.path.getmtime(filepath) < time.time() - 3600:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filename}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def save_file(file, filename):
    """Save uploaded file with security measures."""
    try:
        # Clean up old files before saving new ones
        cleanup_old_files()
        
        # Secure the filename
        secure_name = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        logger.info(f"File saved successfully: {filename} (original: {secure_name})")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file {filename}: {str(e)}")
        raise

def get_mean_face_encoding(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()

    encodings = []
    for frame in frames:
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locations)
        encodings.extend(face_encs)
    
    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        return mean_encoding
    return None

@app.route('/upload_user_video', methods=['POST'])
def upload_user_video():
    file = request.files.get('file')
    if file:
        save_file(file, 'user_video.mp4')
        return jsonify({'message': 'User video uploaded successfully'}), 200
    return jsonify({'error': 'No video file uploaded'}), 400

@app.route('/upload_document_video', methods=['POST'])
def upload_document_video():
    file = request.files.get('file')
    if file:
        save_file(file, 'document_video.mp4')
        return jsonify({'message': 'Document video uploaded successfully'}), 200
    return jsonify({'error': 'No video file uploaded'}), 400

@app.route('/verify_faces', methods=['POST'])
def verify_faces():
    user_video_path = os.path.join(UPLOAD_FOLDER, 'user_video.mp4')
    document_video_path = os.path.join(UPLOAD_FOLDER, 'document_video.mp4')
    
    if not (os.path.exists(user_video_path) and os.path.exists(document_video_path)):
        return jsonify({'error': 'Both videos must be uploaded'}), 400

    user_mean_encoding = get_mean_face_encoding(user_video_path)
    document_mean_encoding = get_mean_face_encoding(document_video_path)

    if user_mean_encoding is not None and document_mean_encoding is not None:
        distance = np.linalg.norm(user_mean_encoding - document_mean_encoding)
        similarity = (1 - distance) * 100
        is_verified = similarity >= 80.0
        return jsonify({'verified': is_verified}), 200
    
    return jsonify({'error': 'Could not encode faces in one or both videos'}), 400

if __name__ == '__main__':
    app.run(debug=True)

