# Face Verification Flask API

A Flask-based REST API for face verification using video files. This application compares faces from two video sources (user selfie video and document/ID video) to verify identity.

## Features

- **Upload User Video**: Accept and store user selfie video
- **Upload Document Video**: Accept and store ID/passport video with user's face
- **Face Verification**: Compare faces from both videos with 80% similarity threshold

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

### Upload User Video
```http
POST /upload_user_video
Content-Type: multipart/form-data

Form data:
- file: video file (user selfie)
```

**Response**:
```json
{
  "message": "User video uploaded successfully"
}
```

### Upload Document Video
```http
POST /upload_document_video
Content-Type: multipart/form-data

Form data:
- file: video file (ID/passport with face)
```

**Response**:
```json
{
  "message": "Document video uploaded successfully"
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

1. **Frame Extraction**: Both videos are processed to extract all frames
2. **Face Detection**: Each frame is analyzed to detect faces using face_recognition library
3. **Face Encoding**: Detected faces are converted to 128-dimensional encodings
4. **Mean Encoding**: Calculate average encoding across all frames for each video
5. **Comparison**: Compare the mean encodings using Euclidean distance
6. **Verification**: Return `true` if similarity ≥ 80%, otherwise `false`

## Testing

You can test the API using tools like:
- **Postman**: For GUI-based testing
- **cURL**: For command-line testing
- **Python requests**: For programmatic testing

### Example with cURL:

```bash
# Upload user video
curl -X POST -F "file=@user_selfie.mp4" http://localhost:5000/upload_user_video

# Upload document video
curl -X POST -F "file=@id_document.mp4" http://localhost:5000/upload_document_video

# Verify faces
curl -X POST http://localhost:5000/verify_faces
```

## File Structure

```
DocumentVerifier/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── uploads/           # Directory for uploaded videos (created automatically)
    ├── user_video.mp4     # Latest user video
    └── document_video.mp4 # Latest document video
```

## Notes

- Videos are temporarily stored in the `uploads/` directory
- Each new upload overwrites the previous file
- Supported video formats: MP4, AVI, MOV (depends on OpenCV support)
- The application runs in debug mode for development
