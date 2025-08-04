import requests
import os
import time
from tkinter import filedialog, messagebox
import tkinter as tk

BASE_URL = "http://localhost:5000"

def select_video_file(title):
    """Open file explorer to select a video file."""
    # Create root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Define video file types
    file_types = [
        ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv'),
        ('MP4 files', '*.mp4'),
        ('AVI files', '*.avi'),
        ('MOV files', '*.mov'),
        ('MKV files', '*.mkv'),
        ('WMV files', '*.wmv'),
        ('All files', '*.*')
    ]
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=file_types
    )
    
    root.destroy()
    return file_path

def select_image_file(title):
    """Open file explorer to select an image file."""
    # Create root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Define image file types
    file_types = [
        ('Image files', '*.jpg *.jpeg *.png'),
        ('JPEG files', '*.jpg *.jpeg'),
        ('PNG files', '*.png'),
        ('All files', '*.*')
    ]
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=file_types
    )
    
    root.destroy()
    return file_path

def check_server_health():
    """Check if the Flask server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and healthy")
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def upload_file(endpoint, file_path):
    """Upload a video file to the specified endpoint."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📤 Uploading {file_path} to {endpoint}...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(f"{BASE_URL}/{endpoint}", files={'file': f}, timeout=30)
            
        if response.status_code == 200:
            print(f"✅ Upload successful: {response.json()['message']}")
            return True
        else:
            print(f"❌ Upload failed: {response.json()}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error during upload: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return False

def verify_faces():
    """Call the face verification endpoint."""
    print("🔍 Starting face verification...")
    try:
        response = requests.post(f"{BASE_URL}/verify_faces", timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            verified = result.get('verified', False)
            similarity = result.get('similarity', 'N/A')
            
            if verified:
                print(f"✅ VERIFICATION SUCCESSFUL!")
                print(f"   Similarity: {similarity}%")
            else:
                print(f"❌ VERIFICATION FAILED")
                print(f"   Similarity: {similarity}%")
            
            return result
        else:
            error_msg = response.json().get('error', 'Unknown error')
            print(f"❌ Verification failed: {error_msg}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error during verification: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return None

def main():
    print("🚀 Document Verifier Client")
    print("=" * 40)
    
    # Check server health
    if not check_server_health():
        print("\n❌ Please start the Flask server first by running:")
        print("   python app_simple.py")
        return
    
    print("\n📹 Video Upload Process")
    print("-" * 25)
    
    # Get user video
    while True:
        print("\n📁 Please select the USER video file...")
        print("Opening file explorer...")
        user_video_path = select_video_file("Select User Video File")
        
        if not user_video_path:
            print("❌ No file selected. Please try again.")
            continue
            
        print(f"Selected file: {os.path.basename(user_video_path)}")
        if upload_file("upload_user_video", user_video_path):
            break
    
    # Get document image
    while True:
        print("\n📁 Please select the DOCUMENT image file...")
        print("Opening file explorer...")
        document_image_path = select_image_file("Select Document Image File")
        
        if not document_image_path:
            print("❌ No file selected. Please try again.")
            continue
            
        print(f"Selected file: {os.path.basename(document_image_path)}")
        if upload_file("upload_document_image", document_image_path):
            break
    
    # Verify faces
    print("\n🔍 Face Verification")
    print("-" * 20)
    result = verify_faces()
    
    # Show logging information
    print("\n📋 Logging Information")
    print("-" * 22)
    print("📄 Check 'app.log' file for detailed server logs")
    
    if os.path.exists('app.log'):
        print("\n📄 Recent log entries:")
        try:
            with open('app.log', 'r') as f:
                lines = f.readlines()
                # Show last 10 lines
                for line in lines[-10:]:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Could not read log file: {e}")
    
    print("\n✨ Process completed!")

if __name__ == "__main__":
    main()

