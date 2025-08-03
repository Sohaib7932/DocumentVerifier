from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import random

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

def mock_face_verification(video_path):
    """Mock face verification function for testing."""
    try:
        # Check if video file exists and is readable
        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()
        
        if frame_count > 0:
            logger.info(f"Video processed: {video_path} with {frame_count} frames")
            # Return a mock encoding (random array for simulation)
            return np.random.rand(128)
        else:
            logger.warning(f"No frames found in video: {video_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None

@app.route('/upload_user_video', methods=['POST'])
def upload_user_video():
    """Upload user video endpoint."""
    try:
        logger.info("Received request to upload user video")
        file = request.files.get('file')
        
        is_valid, message = validate_video_file(file)
        if not is_valid:
            logger.warning(f"Invalid user video file: {message}")
            return jsonify({'error': message}), 400
        
        filepath = save_file(file, 'user_video.mp4')
        logger.info(f"User video uploaded successfully: {filepath}")
        return jsonify({'message': 'User video uploaded successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error uploading user video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload_document_video', methods=['POST'])
def upload_document_video():
    """Upload document video endpoint."""
    try:
        logger.info("Received request to upload document video")
        file = request.files.get('file')
        
        is_valid, message = validate_video_file(file)
        if not is_valid:
            logger.warning(f"Invalid document video file: {message}")
            return jsonify({'error': message}), 400
        
        filepath = save_file(file, 'document_video.mp4')
        logger.info(f"Document video uploaded successfully: {filepath}")
        return jsonify({'message': 'Document video uploaded successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error uploading document video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/verify_faces', methods=['POST'])
def verify_faces():
    """Verify faces from both uploaded videos."""
    try:
        logger.info("Received request to verify faces")
        user_video_path = os.path.join(UPLOAD_FOLDER, 'user_video.mp4')
        document_video_path = os.path.join(UPLOAD_FOLDER, 'document_video.mp4')
        
        if not (os.path.exists(user_video_path) and os.path.exists(document_video_path)):
            logger.warning("Missing video files for verification")
            return jsonify({'error': 'Both videos must be uploaded'}), 400

        logger.info("Processing user video for face verification")
        user_encoding = mock_face_verification(user_video_path)
        
        logger.info("Processing document video for face verification")
        document_encoding = mock_face_verification(document_video_path)

        if user_encoding is not None and document_encoding is not None:
            # Calculate similarity (mock calculation)
            distance = np.linalg.norm(user_encoding - document_encoding)
            similarity = max(0, (1 - distance/2) * 100)  # Normalized similarity
            is_verified = similarity >= SIMILARITY_THRESHOLD
            
            logger.info(f"Face verification completed: similarity={similarity:.2f}%, verified={is_verified}")
            return jsonify({
                'verified': is_verified,
                'similarity': round(similarity, 2)
            }), 200
        else:
            logger.error("Could not process faces in one or both videos")
            return jsonify({'error': 'Could not encode faces in one or both videos'}), 400
        
    except Exception as e:
        logger.error(f"Error during face verification: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)
