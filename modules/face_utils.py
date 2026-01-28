"""
Face Recognition & Gender Classification Module
================================================

BUOC 1: TIM KHUON MAT (Face Detection)
- Uu tien: face_recognition (dlib HOG) - Chinh xac nhat
- Du phong: DNN SSD ResNet-10
- Cuoi cung: Haar Cascade

BUOC 2: TAO VECTOR 128 SO (Face Encoding)
- Su dung: dlib ResNet-34 (da train tren 3 trieu anh)
- Output: Vector 128 chieu - "van tay" khuon mat

BUOC 3: SO SANH VA NHAN DANG
- Khoang cach Euclidean < 0.6 = Cung nguoi
- Do tin cay = (1 - khoang cach) x 100%

BUOC 4: NHAN DANG GIOI TINH
- Uu tien: DeepFace (VGG-Face CNN) - ~97% accuracy
- Du phong: Caffe model
"""

import cv2
import numpy as np
import os

# ===============================
# Paths & Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model paths
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
HAAR_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
DNN_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt.txt")
DNN_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ===============================
# Load Libraries
# ===============================
USE_FACE_RECOGNITION = False
face_recognition = None

try:
    import face_recognition as fr
    face_recognition = fr
    USE_FACE_RECOGNITION = True
    print("[OK] face_recognition (dlib 128-d) loaded - 99.38% accuracy")
except ImportError as e:
    print(f"[WARN] face_recognition not available: {e}")
    print("   -> Using DNN SSD as primary method")

USE_DEEPFACE = False
DeepFace = None

try:
    from deepface import DeepFace as DF
    DeepFace = DF
    USE_DEEPFACE = True
    print("[OK] DeepFace CNN loaded - Gender accuracy ~97%")
except ImportError as e:
    print(f"[WARN] DeepFace not available: {e}")
    print("   -> Using Caffe model for gender")

# ===============================
# Global Models (lazy loading)
# ===============================
_gender_net = None
_face_cascade = None
_dnn_net = None


def _to_float_list(arr):
    """Convert numpy array to Python list of floats"""
    if arr is None:
        return []
    try:
        if hasattr(arr, 'tolist'):
            return [float(x) for x in arr.tolist()]
        if hasattr(arr, 'flatten'):
            arr = arr.flatten()
        return [float(x) for x in arr]
    except Exception:
        return []


def _check_model_exists(path, name):
    """Check if model file exists"""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [OK] {name}: {size_mb:.2f} MB")
        return True
    else:
        print(f"  [MISSING] {name}: {path}")
        return False


# ===============================
# Gender Classification
# ===============================

def _load_gender_model():
    """Load Caffe gender model"""
    global _gender_net
    if _gender_net is None:
        if os.path.exists(GENDER_PROTO) and os.path.exists(GENDER_MODEL):
            try:
                _gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
                print("[OK] Caffe Gender Model loaded")
            except Exception as e:
                print(f"[ERROR] Cannot load Caffe gender model: {e}")
                _gender_net = None
    return _gender_net


def _predict_gender_deepface(face_img):
    """
    Gender classification using DeepFace CNN
    - Model: VGG-Face
    - Accuracy: ~97%
    """
    if DeepFace is None:
        return _predict_gender_caffe(face_img)
    
    try:
        if face_img.shape[0] < 48 or face_img.shape[1] < 48:
            face_img = cv2.resize(face_img, (100, 100))
        
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        result = DeepFace.analyze(
            face_rgb,
            actions=['gender'],
            enforce_detection=False,
            silent=True,
            detector_backend='skip'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        gender_dict = result.get('gender', {})
        
        if isinstance(gender_dict, dict):
            woman_score = gender_dict.get('Woman', 0)
            man_score = gender_dict.get('Man', 0)
            
            if woman_score > man_score:
                return "Nu", woman_score / 100.0
            else:
                return "Nam", man_score / 100.0
        else:
            gender = "Nu" if str(gender_dict).lower() == "woman" else "Nam"
            return gender, 0.85
            
    except Exception as e:
        print(f"[WARN] DeepFace gender error: {e}")
        return _predict_gender_caffe(face_img)


def _predict_gender_caffe(face_img):
    """
    Gender classification using Caffe model
    - Input: 227x227 BGR image
    - Output: [Male, Female] probabilities
    """
    net = _load_gender_model()
    if net is None:
        return "Unknown", 0.0
    
    try:
        if face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return "Unknown", 0.0
        
        face = cv2.resize(face_img, (227, 227))
        
        blob = cv2.dnn.blobFromImage(
            face, 
            scalefactor=1.0, 
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
            crop=False
        )
        
        net.setInput(blob)
        predictions = net.forward()
        
        male_prob = float(predictions[0][0])
        female_prob = float(predictions[0][1])
        
        if male_prob > female_prob and male_prob > 0.55:
            return "Nam", male_prob
        elif female_prob > male_prob and female_prob > 0.55:
            return "Nu", female_prob
        else:
            return "Unknown", max(male_prob, female_prob)
            
    except Exception as e:
        print(f"[WARN] Caffe gender error: {e}")
        return "Unknown", 0.0


def predict_gender(face_img):
    """
    Predict gender from face image.
    Returns: (gender_label, confidence)
    """
    if face_img is None:
        return "Unknown", 0.0
    
    if not hasattr(face_img, 'shape') or face_img.size == 0:
        return "Unknown", 0.0
    
    if USE_DEEPFACE:
        return _predict_gender_deepface(face_img)
    return _predict_gender_caffe(face_img)


# ===============================
# Face Detection
# ===============================

def _load_haar():
    """Load Haar Cascade classifier"""
    global _face_cascade
    if _face_cascade is None:
        if os.path.exists(HAAR_PATH):
            _face_cascade = cv2.CascadeClassifier(HAAR_PATH)
            if _face_cascade.empty():
                print("[ERROR] Haar Cascade is empty!")
                _face_cascade = None
    return _face_cascade


def _load_dnn():
    """Load DNN SSD face detector"""
    global _dnn_net
    if _dnn_net is None:
        if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
            try:
                _dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                print("[OK] DNN SSD Face Detector loaded")
            except Exception as e:
                print(f"[ERROR] Cannot load DNN: {e}")
                _dnn_net = None
    return _dnn_net


def _detect_haar(image):
    """Detect faces using Haar Cascade"""
    cascade = _load_haar()
    if cascade is None:
        return []
    
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.equalizeHist(gray)
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            return [tuple(map(int, f)) for f in faces]
        return []
        
    except Exception as e:
        print(f"[WARN] Haar detect error: {e}")
        return []


def _detect_dnn(image, conf_thresh=0.5):
    """Detect faces using DNN SSD (ResNet-10)"""
    net = _load_dnn()
    if net is None:
        return _detect_haar(image)
    
    try:
        h, w = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > conf_thresh:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 + 20 and y2 > y1 + 20:
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
        
    except Exception as e:
        print(f"[WARN] DNN detect error: {e}")
        return _detect_haar(image)


def _detect_face_recognition(image):
    """Detect faces using face_recognition (dlib HOG)"""
    if not USE_FACE_RECOGNITION or face_recognition is None:
        return _detect_dnn(image)
    
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        
        faces = []
        for (top, right, bottom, left) in locations:
            x = left
            y = top
            w = right - left
            h = bottom - top
            faces.append((x, y, w, h))
        
        return faces
        
    except Exception as e:
        print(f"[WARN] face_recognition detect error: {e}")
        return _detect_dnn(image)


def detect_faces(image, method="auto"):
    """
    Detect faces in image.
    Returns: list of (x, y, w, h) tuples
    """
    if image is None or not hasattr(image, 'shape'):
        return []
    
    if image.size == 0:
        return []
    
    if method == "auto":
        if USE_FACE_RECOGNITION:
            faces = _detect_face_recognition(image)
            if faces:
                return faces
        
        faces = _detect_dnn(image)
        if faces:
            return faces
        
        return _detect_haar(image)
        
    elif method == "face_recognition":
        return _detect_face_recognition(image)
    elif method == "dnn":
        return _detect_dnn(image)
    else:
        return _detect_haar(image)


# ===============================
# Face Encoding
# ===============================

def _encode_dlib(face_img):
    """Create face encoding using dlib (128-dimensional vector)"""
    if not USE_FACE_RECOGNITION or face_recognition is None:
        return _encode_simple(face_img)
    
    try:
        min_size = 80
        if face_img.shape[0] < min_size or face_img.shape[1] < min_size:
            scale = max(min_size / face_img.shape[0], min_size / face_img.shape[1])
            new_h = int(face_img.shape[0] * scale)
            new_w = int(face_img.shape[1] * scale)
            face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        
        if not locations:
            h, w = rgb.shape[:2]
            locations = [(0, w, h, 0)]
        
        encodings = face_recognition.face_encodings(rgb, locations, num_jitters=1)
        
        if encodings and len(encodings) > 0:
            return _to_float_list(encodings[0])
        
        return _encode_simple(face_img)
        
    except Exception as e:
        print(f"[WARN] encode_dlib error: {e}")
        return _encode_simple(face_img)


def _encode_simple(face_img):
    """Simple encoding (fallback)"""
    try:
        face = cv2.resize(face_img, (64, 64))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        arr = gray.flatten().astype(np.float32) / 255.0
        
        if len(arr) > 128:
            indices = np.linspace(0, len(arr) - 1, 128, dtype=int)
            arr = arr[indices]
        
        return _to_float_list(arr)
        
    except Exception as e:
        print(f"[WARN] encode_simple error: {e}")
        return []


def encode_face(face_img):
    """
    Create face encoding from face image.
    Returns: list of 128 floats
    """
    if face_img is None:
        return []
    
    if not hasattr(face_img, 'shape') or face_img.size == 0:
        return []
    
    if USE_FACE_RECOGNITION:
        return _encode_dlib(face_img)
    return _encode_simple(face_img)


# ===============================
# Face Matching
# ===============================

def compare_faces(known_enc, test_enc, threshold=0.6):
    """
    Compare two faces using Euclidean distance.
    Returns: (is_match, distance)
    """
    if not known_enc or not test_enc:
        return False, 1.0
    
    if len(known_enc) != len(test_enc):
        return False, 1.0
    
    try:
        known = np.array(known_enc, dtype=np.float64)
        test = np.array(test_enc, dtype=np.float64)
        
        distance = float(np.linalg.norm(known - test))
        is_match = distance < threshold
        
        return is_match, distance
        
    except Exception as e:
        print(f"[WARN] compare_faces error: {e}")
        return False, 1.0


def find_best_match(test_encoding, known_faces, threshold=0.6):
    """
    Find best matching employee for a face.
    Returns: (matched_employee, confidence)
    """
    if not test_encoding:
        return None, 0.0
    
    if not known_faces:
        return None, 0.0
    
    best_match = None
    best_distance = float('inf')
    
    for employee in known_faces:
        encoding = employee.get('encoding', [])
        
        if not encoding:
            continue
        if len(encoding) != len(test_encoding):
            continue
        
        is_match, distance = compare_faces(encoding, test_encoding, threshold)
        
        if is_match and distance < best_distance:
            best_distance = distance
            best_match = employee
    
    if best_match is not None:
        if best_distance <= 0.2:
            confidence = 0.99 - (best_distance * 0.2)
        elif best_distance <= 0.4:
            confidence = 0.95 - ((best_distance - 0.2) * 0.75)
        elif best_distance <= 0.5:
            confidence = 0.80 - ((best_distance - 0.4) * 1.5)
        else:
            confidence = 0.65 - ((best_distance - 0.5) * 1.5)
        
        confidence = max(0.50, min(0.99, confidence))
        
        return best_match, confidence
    
    return None, 0.0


# ===============================
# Utility Functions
# ===============================

def check_models():
    """Check all model files"""
    print("\n" + "="*50)
    print("CHECKING MODEL FILES")
    print("="*50)
    
    print(f"\nModel Directory: {MODEL_DIR}")
    
    models = {
        "Haar Cascade": HAAR_PATH,
        "DNN Proto": DNN_PROTO,
        "DNN Model": DNN_MODEL,
        "Gender Proto": GENDER_PROTO,
        "Gender Model": GENDER_MODEL,
    }
    
    all_ok = True
    for name, path in models.items():
        if not _check_model_exists(path, name):
            all_ok = False
    
    print("\n" + "="*50)
    
    if all_ok:
        print("[OK] ALL MODELS READY!")
    else:
        print("[ERROR] SOME MODELS MISSING!")
    
    print("="*50 + "\n")
    
    return all_ok


def get_system_info():
    """Return face recognition system info"""
    return {
        "face_recognition_available": USE_FACE_RECOGNITION,
        "deepface_available": USE_DEEPFACE,
        "encoding_method": "dlib 128-d" if USE_FACE_RECOGNITION else "simple histogram",
        "gender_method": "DeepFace CNN" if USE_DEEPFACE else "Caffe model",
        "detection_method": "face_recognition" if USE_FACE_RECOGNITION else "DNN SSD",
    }


if __name__ == "__main__":
    check_models()
    print(get_system_info())
