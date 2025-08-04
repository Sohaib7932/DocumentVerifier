# Face Verification Flask API

A Flask-based REST API for face verification using image files. This application compares faces from two image sources (user selfie image and document/ID image) to verify identity.

## Features

- **Upload User Image**: Accept and store user selfie image
- **Upload Document Image**: Accept and store ID/passport image with user's face
- **Face Verification**: Compare faces from both images with 80% similarity threshold

## Tech Stack

- **Flask**: Web framework for REST API
- **OpenCV**: Video processing and frame extraction
- **face_recognition**: Face detection and encoding (uses dlib)
- **NumPy**: Numerical operations for face comparison

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Note**: On Windows, you might need to install Visual Studio Build Tools for dlib compilation.

## Usage

1. **Start the Flask server**:
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

2. **API Endpoints**:

### Upload User Image
```http
POST /upload_user_image
Content-Type: multipart/form-data

Form data:
- file: image file (user selfie)
```

**Response**:
```json
{
  "message": "User image uploaded successfully"
}
```

### Upload Document Image
```http
POST /upload_document_image
Content-Type: multipart/form-data

Form data:
- file: image file (ID/passport with face)
```

**Response**:
```json
{
  "message": "Document image uploaded successfully"
}
```

### Verify Faces
```http
POST /verify_faces
```

**Response**:
```json
{
  "verified": true
}
```
or
```json
{
  "verified": false
}
```

## How It Works

1. **Face Detection**: Both images are analyzed to detect faces using face_recognition library
2. **Face Encoding**: Detected faces are converted to 128-dimensional encodings
3. **Comparison**: Compare the encodings using Euclidean distance
4. **Verification**: Return `true` if similarity ≥ 80%, otherwise `false`

## Testing

You can test the API using tools like:
- **Postman**: For GUI-based testing
- **cURL**: For command-line testing
- **Python requests**: For programmatic testing

### Example with cURL:

```bash
# Upload user image
curl -X POST -F "file=@user_selfie.jpg" http://localhost:5000/upload_user_image

# Upload document image
curl -X POST -F "file=@id_document.jpg" http://localhost:5000/upload_document_image

# Verify faces
curl -X POST http://localhost:5000/verify_faces
```

## File Structure

```
DocumentVerifier/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── uploads/           # Directory for uploaded images (created automatically)
    ├── user_image.jpg     # Latest user image
    └── document_image.jpg # Latest document image
```

## Notes

- Images are temporarily stored in the `uploads/` directory
- Each new upload overwrites the previous file
- Supported image formats: JPEG, PNG
- The application runs in debug mode for development
