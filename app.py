from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
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
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
SIMILARITY_THRESHOLD = 70.0  # Lowered threshold for more realistic face matching

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_image_file(filename):
    """Check if the uploaded file has an allowed image extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    """Check if the uploaded file has an allowed video extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def validate_video_file(file):
    """Validate uploaded video file."""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_video_file(file.filename):
        return False, f"File type not allowed. Supported formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    
    return True, "File is valid"

def validate_image_file(file):
    """Validate uploaded image file."""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_image_file(file.filename):
        return False, f"File type not allowed. Supported formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
    
    return True, "File is valid"

def cleanup_old_files():
    """Remove old files to save space."""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.jpg', '.jpeg', '.png')):
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

def extract_face_features_from_video(video_path):
    """Extract face features from video frames."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video: {video_path} with {frame_count} frames")
        
        face_features = []
        frame_number = 0
        faces_found = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for increased accuracy
            if frame_number % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.05, 6)
                
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]

                    # Enhance face detail
                    face_enhanced = cv2.equalizeHist(face_roi)

                    # Resize to standard size for comparison
                    face_resized = cv2.resize(face_enhanced, (100, 100))

                    # Resize face to HOG-compatible size
                    face_hog_size = cv2.resize(face_enhanced, (64, 64))
                    
                    # Use Histogram of Oriented Gradients (HOG) for feature extraction
                    winSize = (64, 64)
                    blockSize = (16, 16)
                    blockStride = (8, 8)
                    cellSize = (8, 8)
                    nbins = 9
                    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
                    hist = hog.compute(face_hog_size)
                    hist = np.ravel(hist)

                    face_features.append(hist)
                    faces_found += 1

            frame_number += 1
        
        cap.release()
        
        logger.info(f"Found {faces_found} faces in video")
        
        if face_features:
            # Calculate mean feature vector
            mean_features = np.mean(face_features, axis=0)
            return mean_features
        else:
            logger.warning(f"No faces found in video: {video_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None

def extract_face_features_from_image(image_path):
    """Extract face features from image."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info(f"Image loaded successfully, shape: {gray.shape}")
        
        # Detect faces with multiple scale factors for better detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            # Try with different parameters
            faces = face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(20, 20))
            
        logger.info(f"Found {len(faces)} faces in image")
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for comparison
            face_resized = cv2.resize(face_roi, (100, 100))

            # Equalize histogram for better feature contrast
            face_enhanced = cv2.equalizeHist(face_resized)
            
            # Resize face to HOG-compatible size
            face_hog_size = cv2.resize(face_enhanced, (64, 64))
            
            # Use Histogram of Oriented Gradients (HOG) for feature extraction
            winSize = (64, 64)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (8, 8)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            hist = hog.compute(face_hog_size)
            hist = np.ravel(hist)
            
            logger.info(f"Generated face features for largest face (size: {w}x{h})")
            return hist
        else:
            logger.warning(f"No faces found in image: {image_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def calculate_similarity(features1, features2):
    """Calculate similarity between two feature vectors using multiple methods."""
    try:
        # Convert to float32 for OpenCV functions
        f1 = features1.astype(np.float32)
        f2 = features2.astype(np.float32)
        
        # Method 1: Histogram correlation
        correlation = cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
        corr_similarity = max(0, correlation * 100)
        
        # Method 2: Chi-Square (lower is better, so invert)
        chi_square = cv2.compareHist(f1, f2, cv2.HISTCMP_CHISQR)
        chi_similarity = max(0, (1 / (1 + chi_square)) * 100)
        
        # Method 3: Intersection (normalized)
        intersection = cv2.compareHist(f1, f2, cv2.HISTCMP_INTERSECT)
        intersect_similarity = min(100, intersection)  # Cap at 100%
        
        # Method 4: Cosine similarity
        dot_product = np.dot(f1, f2)
        norm_a = np.linalg.norm(f1)
        norm_b = np.linalg.norm(f2)
        cosine_sim = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
        cosine_similarity = max(0, cosine_sim * 100)
        
        # Use primarily correlation and cosine similarity as they are more reliable
        combined_similarity = (
            corr_similarity * 0.6 +
            cosine_similarity * 0.4
        )
        
        logger.info(f"Similarity methods - Correlation: {corr_similarity:.2f}%, Chi-Square: {chi_similarity:.2f}%, Intersection: {intersect_similarity:.2f}%, Cosine: {cosine_similarity:.2f}%")
        
        # Convert to Python float for JSON serialization
        return float(combined_similarity)
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

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

@app.route('/upload_document_image', methods=['POST'])
def upload_document_image():
    """Upload document image endpoint."""
    try:
        logger.info("Received request to upload document image")
        file = request.files.get('file')
        
        is_valid, message = validate_image_file(file)
        if not is_valid:
            logger.warning(f"Invalid document image file: {message}")
            return jsonify({'error': message}), 400
        
        filepath = save_file(file, 'document_image.jpg')
        logger.info(f"Document image uploaded successfully: {filepath}")
        return jsonify({'message': 'Document image uploaded successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error uploading document image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/verify_faces', methods=['POST'])
def verify_faces():
    """Verify faces from user video and document image."""
    try:
        logger.info("Received request to verify faces")
        user_video_path = os.path.join(UPLOAD_FOLDER, 'user_video.mp4')
        document_image_path = os.path.join(UPLOAD_FOLDER, 'document_image.jpg')
        
        if not (os.path.exists(user_video_path) and os.path.exists(document_image_path)):
            logger.warning("Missing user video or document image for verification")
            return jsonify({'error': 'Both user video and document image must be uploaded'}), 400

        logger.info("Processing user video for face verification")
        user_features = extract_face_features_from_video(user_video_path)
        
        logger.info("Processing document image for face verification")
        document_features = extract_face_features_from_image(document_image_path)

        if user_features is not None and document_features is not None:
            # Calculate similarity
            similarity = calculate_similarity(user_features, document_features)
            is_verified = similarity >= SIMILARITY_THRESHOLD
            
            logger.info(f"Face verification completed: similarity={similarity:.2f}%, verified={is_verified}")
            return jsonify({
                'verified': is_verified,
                'similarity': round(similarity, 2)
            }), 200
        else:
            logger.error("Could not extract features from user video or document image")
            return jsonify({'error': 'Could not extract face features from user video or document image'}), 400
        
    except Exception as e:
        logger.error(f"Error during face verification: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application with OpenCV face verification")
    app.run(debug=True, host='0.0.0.0', port=5000)
