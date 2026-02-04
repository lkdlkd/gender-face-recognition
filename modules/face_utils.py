"""
Face Recognition & Gender Classification Module
================================================

BƯỚC 1: TÌM KHUÔN MẶT (Face Detection)
- Ưu tiên: face_recognition (dlib HOG) - Chính xác nhất
- Dự phòng: DNN SSD ResNet-10
- Cuối cùng: Haar Cascade

BƯỚC 2: TẠO VECTOR 128 SỐ (Face Encoding)
- Sử dụng: dlib ResNet-34 (đã train trên 3 triệu ảnh)
- Output: Vector 128 chiều - "vân tay" khuôn mặt

BƯỚC 3: SO SÁNH VÀ NHẬN DẠNG
- Khoảng cách Euclidean < 0.6 = Cùng người
- Độ tin cậy = (1 - khoảng cách) x 100%

BƯỚC 4: NHẬN DẠNG GIỚI TÍNH
- Ưu tiên: DeepFace (VGG-Face CNN) - ~97% accuracy
- Dự phòng: Caffe model

BƯỚC 5: XÁC THỰC VÀ KIỂM TRA (Validation)
- Kiểm tra tính hợp lệ của ảnh đầu vào
- Kiểm tra chất lượng ảnh (độ sáng, tương phản, độ mờ)
- Xác thực encoding vector (128 chiều, giá trị hợp lệ)
- Ngưỡng tin cậy tối thiểu cho tất cả operations
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Any, Union

# ===============================
# Paths & Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ===============================
# Validation Configuration
# ===============================
# Kích thước ảnh tối thiểu
MIN_IMAGE_SIZE = 20
MIN_FACE_SIZE = 30

# Ngưỡng tin cậy
MIN_DETECTION_CONFIDENCE = 0.5
MIN_GENDER_CONFIDENCE = 0.55
MIN_MATCH_CONFIDENCE = 0.50

# Encoding validation
EXPECTED_ENCODING_LENGTH = 128
ENCODING_VALUE_MIN = -1.0
ENCODING_VALUE_MAX = 1.0

# Image quality thresholds
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 225
MIN_CONTRAST = 20
MIN_SHARPNESS = 50

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
_model_load_errors = {}


# ===============================
# Validation Functions
# ===============================

def _validate_image(img: Any) -> Tuple[bool, str]:
    """
    Xác thực tính hợp lệ của ảnh đầu vào.
    
    Kiểm tra:
    - Ảnh không None
    - Có thuộc tính shape hợp lệ
    - Kích thước tối thiểu
    - Không chứa giá trị NaN/Inf
    
    Returns: (is_valid, error_message)
    """
    if img is None:
        return False, "Image is None"
    
    if not hasattr(img, 'shape'):
        return False, "Invalid image format - no shape attribute"
    
    if not hasattr(img, 'size') or img.size == 0:
        return False, "Image is empty"
    
    if len(img.shape) < 2:
        return False, f"Invalid image dimensions: {len(img.shape)}D, expected 2D or 3D"
    
    h, w = img.shape[:2]
    
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        return False, f"Image too small: {w}x{h}, minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}"
    
    # Kiểm tra giá trị hợp lệ
    if not np.isfinite(img).all():
        return False, "Image contains invalid values (NaN or Inf)"
    
    return True, "OK"


def _validate_face_region(x: int, y: int, w: int, h: int, 
                          img_width: int, img_height: int) -> Tuple[bool, str]:
    """
    Xác thực vùng khuôn mặt có hợp lệ không.
    
    Returns: (is_valid, error_message)
    """
    # Kiểm tra kích thước tối thiểu
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False, f"Face too small: {w}x{h}, minimum: {MIN_FACE_SIZE}x{MIN_FACE_SIZE}"
    
    # Kiem tra bounds
    if x < 0 or y < 0:
        return False, f"Invalid face position: ({x}, {y})"
    
    if x + w > img_width or y + h > img_height:
        return False, f"Face region exceeds image bounds"
    
    # Kiem tra ti le aspect ratio (khuon mat thuong gan vuong)
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False, f"Unusual aspect ratio: {aspect_ratio:.2f}"
    
    return True, "OK"


def _validate_encoding(encoding: List[float]) -> Tuple[bool, str]:
    """
    Xác thực face encoding vector.
    
    Kiểm tra:
    - Không rỗng
    - Đúng độ dài (128 cho dlib)
    - Giá trị trong khoảng hợp lệ
    - Không chứa NaN/Inf
    
    Returns: (is_valid, error_message)
    """
    if not encoding:
        return False, "Encoding is empty"
    
    if not isinstance(encoding, (list, np.ndarray)):
        return False, f"Invalid encoding type: {type(encoding)}"
    
    # Chuyển sang numpy array để kiểm tra
    try:
        arr = np.array(encoding, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return False, f"Cannot convert encoding to array: {e}"
    
    # Kiểm tra độ dài
    if len(arr) != EXPECTED_ENCODING_LENGTH:
        # Cho phép fallback encoding (128 elements từ simple method)
        if len(arr) != 128:
            return False, f"Invalid encoding length: {len(arr)}, expected {EXPECTED_ENCODING_LENGTH}"
    
    # Kiểm tra NaN/Inf
    if not np.isfinite(arr).all():
        return False, "Encoding contains NaN or Inf values"
    
    # Kiểm tra khoảng giá trị (dlib encoding thường trong [-0.5, 0.5])
    if np.abs(arr).max() > 2.0:
        print(f"[WARN] Encoding values may be out of normal range: max={np.abs(arr).max():.4f}")
    
    return True, "OK"


def _check_image_quality(img: np.ndarray) -> Tuple[float, List[str]]:
    """
    Kiểm tra chất lượng ảnh khuôn mặt.
    
    Returns: (quality_score 0-1, list of issues)
    """
    issues = []
    quality_score = 1.0
    
    # Chuyển sang grayscale nếu cần
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1. Kiem tra do sang
    mean_brightness = np.mean(gray)
    if mean_brightness < MIN_BRIGHTNESS:
        issues.append(f"Too dark (brightness: {mean_brightness:.0f})")
        quality_score *= 0.7
    elif mean_brightness > MAX_BRIGHTNESS:
        issues.append(f"Too bright (brightness: {mean_brightness:.0f})")
        quality_score *= 0.7
    
    # 2. Kiem tra do tuong phan
    std_contrast = np.std(gray)
    if std_contrast < MIN_CONTRAST:
        issues.append(f"Low contrast (std: {std_contrast:.1f})")
        quality_score *= 0.8
    
    # 3. Kiem tra do net (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_SHARPNESS:
        issues.append(f"May be blurry (sharpness: {laplacian_var:.1f})")
        quality_score *= 0.85
    
    # 4. Kiem tra histogram distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_std = np.std(hist)
    if hist_std < 100:
        issues.append("Poor histogram distribution")
        quality_score *= 0.9
    
    return max(0.0, min(1.0, quality_score)), issues


def _sanitize_face_region(x: int, y: int, w: int, h: int, 
                          img_width: int, img_height: int, 
                          padding: float = 0.0) -> Tuple[int, int, int, int]:
    """
    Điều chỉnh vùng khuôn mặt để đảm bảo hợp lệ.
    
    Args:
        padding: Tỉ lệ mở rộng vùng mặt (0.1 = 10%)
    
    Returns: (x, y, w, h) đã điều chỉnh
    """
    # Áp dụng padding
    if padding > 0:
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x = x - pad_w
        y = y - pad_h
        w = w + 2 * pad_w
        h = h + 2 * pad_h
    
    # Clamp vao bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    # Dam bao kich thuoc toi thieu
    w = max(MIN_FACE_SIZE, w)
    h = max(MIN_FACE_SIZE, h)
    
    return x, y, w, h


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
                return "Nữ", woman_score / 100.0
            else:
                return "Nam", man_score / 100.0
        else:
            gender = "Nữ" if str(gender_dict).lower() == "woman" else "Nam"
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
            return "Nữ", female_prob
        else:
            return "Unknown", max(male_prob, female_prob)
            
    except Exception as e:
        print(f"[WARN] Caffe gender error: {e}")
        return "Unknown", 0.0


def predict_gender(face_img, return_quality_info: bool = False):
    """
    Predict gender from face image with full validation.
    
    Args:
        face_img: BGR face image
        return_quality_info: If True, return additional quality info
    
    Returns: 
        (gender_label, confidence) 
        or (gender_label, confidence, quality_info) if return_quality_info=True
    """
    default_result = ("Unknown", 0.0)
    
    # === Validation ===
    is_valid, error_msg = _validate_image(face_img)
    if not is_valid:
        print(f"[WARN] Gender prediction - invalid input: {error_msg}")
        if return_quality_info:
            return default_result + ({"error": error_msg},)
        return default_result
    
    # === Quality check ===
    quality_score, quality_issues = _check_image_quality(face_img)
    quality_info = {
        "score": quality_score,
        "issues": quality_issues
    }
    
    if quality_score < 0.5:
        print(f"[WARN] Low image quality for gender: {quality_issues}")
    
    # === Prediction ===
    if USE_DEEPFACE:
        gender, confidence = _predict_gender_deepface(face_img)
    else:
        gender, confidence = _predict_gender_caffe(face_img)
    
    # Điều chỉnh confidence theo chất lượng ảnh
    adjusted_confidence = confidence * quality_score
    
    # Kiểm tra ngưỡng tin cậy
    if adjusted_confidence < MIN_GENDER_CONFIDENCE:
        gender = "Unknown"
    
    quality_info["adjusted_confidence"] = adjusted_confidence
    quality_info["raw_confidence"] = confidence
    
    if return_quality_info:
        return gender, adjusted_confidence, quality_info
    return gender, adjusted_confidence


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


def detect_faces(image, method: str = "auto", 
                 return_confidence: bool = False,
                 min_confidence: float = None) -> List:
    """
    Detect faces in image with validation.
    
    Args:
        image: BGR image
        method: "auto", "face_recognition", "dnn", or "haar"
        return_confidence: If True, return confidence scores (DNN only)
        min_confidence: Override minimum confidence threshold
    
    Returns: 
        list of (x, y, w, h) tuples
        or list of (x, y, w, h, confidence) if return_confidence=True
    """
    # === Validation ===
    is_valid, error_msg = _validate_image(image)
    if not is_valid:
        print(f"[WARN] Face detection - invalid input: {error_msg}")
        return []
    
    conf_thresh = min_confidence if min_confidence is not None else MIN_DETECTION_CONFIDENCE
    
    # === Detection ===
    if method == "auto":
        if USE_FACE_RECOGNITION:
            faces = _detect_face_recognition(image)
            if faces:
                # Validate detected faces
                validated = []
                h, w = image.shape[:2]
                for face in faces:
                    is_valid, _ = _validate_face_region(face[0], face[1], face[2], face[3], w, h)
                    if is_valid:
                        validated.append(face)
                if validated:
                    return validated
        
        faces = _detect_dnn(image, conf_thresh)
        if faces:
            return faces
        
        return _detect_haar(image)
        
    elif method == "face_recognition":
        return _detect_face_recognition(image)
    elif method == "dnn":
        return _detect_dnn(image, conf_thresh)
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


def encode_face(face_img, validate: bool = True) -> List[float]:
    """
    Create face encoding from face image with validation.
    
    Args:
        face_img: BGR face image
        validate: Whether to validate the result
    
    Returns: list of 128 floats (empty list if failed)
    """
    # === Input validation ===
    is_valid, error_msg = _validate_image(face_img)
    if not is_valid:
        print(f"[WARN] Face encoding - invalid input: {error_msg}")
        return []
    
    # === Quality check ===
    quality_score, quality_issues = _check_image_quality(face_img)
    if quality_score < 0.4:
        print(f"[WARN] Low quality image for encoding: {quality_issues}")
    
    # === Encoding ===
    if USE_FACE_RECOGNITION:
        encoding = _encode_dlib(face_img)
    else:
        encoding = _encode_simple(face_img)
    
    # === Output validation ===
    if validate and encoding:
        is_valid, error_msg = _validate_encoding(encoding)
        if not is_valid:
            print(f"[WARN] Invalid encoding generated: {error_msg}")
            return []
    
    return encoding


# ===============================
# Face Matching
# ===============================

def compare_faces(known_enc, test_enc, threshold: float = 0.6, 
                  validate: bool = True) -> Tuple[bool, float]:
    """
    Compare two faces using Euclidean distance with validation.
    
    Args:
        known_enc: Known face encoding
        test_enc: Test face encoding
        threshold: Match threshold (lower = stricter)
        validate: Whether to validate encodings
    
    Returns: (is_match, distance)
    """
    # === Validation ===
    if validate:
        is_valid, error_msg = _validate_encoding(known_enc)
        if not is_valid:
            print(f"[WARN] Invalid known encoding: {error_msg}")
            return False, 1.0
        
        is_valid, error_msg = _validate_encoding(test_enc)
        if not is_valid:
            print(f"[WARN] Invalid test encoding: {error_msg}")
            return False, 1.0
    else:
        if not known_enc or not test_enc:
            return False, 1.0
        
        if len(known_enc) != len(test_enc):
            return False, 1.0
    
    try:
        known = np.array(known_enc, dtype=np.float64)
        test = np.array(test_enc, dtype=np.float64)
        
        # Tính khoảng cách Euclidean
        distance = float(np.linalg.norm(known - test))
        
        # Kiểm tra giá trị hợp lệ
        if not np.isfinite(distance):
            print("[WARN] Distance calculation resulted in invalid value")
            return False, 1.0
        
        is_match = distance < threshold
        
        return is_match, distance
        
    except Exception as e:
        print(f"[WARN] compare_faces error: {e}")
        return False, 1.0


def find_best_match(test_encoding: List[float], 
                    known_faces: List[Dict], 
                    threshold: float = 0.6,
                    min_confidence: float = None) -> Tuple[Optional[Dict], float]:
    """
    Find best matching employee for a face with validation.
    
    Args:
        test_encoding: Face encoding to match
        known_faces: List of employee dicts with 'encoding' key
        threshold: Match threshold
        min_confidence: Minimum confidence to return a match
    
    Returns: (matched_employee or None, confidence)
    """
    # === Validation ===
    is_valid, error_msg = _validate_encoding(test_encoding)
    if not is_valid:
        print(f"[WARN] Invalid test encoding: {error_msg}")
        return None, 0.0
    
    if not known_faces:
        return None, 0.0
    
    min_conf = min_confidence if min_confidence is not None else MIN_MATCH_CONFIDENCE
    
    best_match = None
    best_distance = float('inf')
    valid_comparisons = 0
    
    for employee in known_faces:
        encoding = employee.get('encoding', [])
        
        if not encoding:
            continue
        
        # Validate stored encoding
        is_valid, _ = _validate_encoding(encoding)
        if not is_valid:
            print(f"[WARN] Invalid stored encoding for employee: {employee.get('name', 'Unknown')}")
            continue
        
        if len(encoding) != len(test_encoding):
            continue
        
        valid_comparisons += 1
        is_match, distance = compare_faces(encoding, test_encoding, threshold, validate=False)
        
        if is_match and distance < best_distance:
            best_distance = distance
            best_match = employee
    
    if valid_comparisons == 0:
        print("[WARN] No valid encodings found in known_faces")
        return None, 0.0
    
    if best_match is not None:
        # Tính confidence từ distance
        if best_distance <= 0.2:
            confidence = 0.99 - (best_distance * 0.2)
        elif best_distance <= 0.4:
            confidence = 0.95 - ((best_distance - 0.2) * 0.75)
        elif best_distance <= 0.5:
            confidence = 0.80 - ((best_distance - 0.4) * 1.5)
        else:
            confidence = 0.65 - ((best_distance - 0.5) * 1.5)
        
        confidence = max(min_conf, min(0.99, confidence))
        
        # Kiểm tra ngưỡng tin cậy tối thiểu
        if confidence < min_conf:
            return None, confidence
        
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


def get_system_info() -> Dict[str, Any]:
    """Return face recognition system info"""
    return {
        "face_recognition_available": USE_FACE_RECOGNITION,
        "deepface_available": USE_DEEPFACE,
        "encoding_method": "dlib 128-d" if USE_FACE_RECOGNITION else "simple histogram",
        "gender_method": "DeepFace CNN" if USE_DEEPFACE else "Caffe model",
        "detection_method": "face_recognition" if USE_FACE_RECOGNITION else "DNN SSD",
        "model_load_errors": _model_load_errors,
        "config": {
            "min_image_size": MIN_IMAGE_SIZE,
            "min_face_size": MIN_FACE_SIZE,
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
            "min_gender_confidence": MIN_GENDER_CONFIDENCE,
            "min_match_confidence": MIN_MATCH_CONFIDENCE,
            "expected_encoding_length": EXPECTED_ENCODING_LENGTH,
        }
    }


# ===============================
# Advanced Utility Functions
# ===============================

def extract_face(image, x: int, y: int, w: int, h: int, 
                 padding: float = 0.1) -> Optional[np.ndarray]:
    """
    Extract face region from image with validation and padding.
    
    Args:
        image: Source BGR image
        x, y, w, h: Face bounding box
        padding: Padding ratio around face (0.1 = 10%)
    
    Returns: Cropped face image or None if invalid
    """
    # Validate input image
    is_valid, error_msg = _validate_image(image)
    if not is_valid:
        print(f"[WARN] extract_face - invalid image: {error_msg}")
        return None
    
    img_h, img_w = image.shape[:2]
    
    # Sanitize face region
    x, y, w, h = _sanitize_face_region(x, y, w, h, img_w, img_h, padding)
    
    # Validate face region
    is_valid, error_msg = _validate_face_region(x, y, w, h, img_w, img_h)
    if not is_valid:
        print(f"[WARN] extract_face - invalid region: {error_msg}")
        return None
    
    # Extract face
    try:
        face = image[y:y+h, x:x+w].copy()
        return face
    except Exception as e:
        print(f"[ERROR] extract_face failed: {e}")
        return None


def process_face_complete(image, face_box: Tuple[int, int, int, int],
                          do_gender: bool = True,
                          do_encoding: bool = True) -> Dict[str, Any]:
    """
    Complete face processing pipeline with all validations.
    
    Args:
        image: Source BGR image
        face_box: (x, y, w, h) face bounding box
        do_gender: Whether to predict gender
        do_encoding: Whether to create encoding
    
    Returns: Dict with all results and quality info
    """
    result = {
        "success": False,
        "face_valid": False,
        "quality": {},
        "gender": None,
        "gender_confidence": 0.0,
        "encoding": [],
        "errors": []
    }
    
    # Extract face
    x, y, w, h = face_box
    face = extract_face(image, x, y, w, h, padding=0.1)
    
    if face is None:
        result["errors"].append("Failed to extract face")
        return result
    
    result["face_valid"] = True
    
    # Quality check
    quality_score, quality_issues = _check_image_quality(face)
    result["quality"] = {
        "score": quality_score,
        "issues": quality_issues
    }
    
    # Gender prediction
    if do_gender:
        gender, confidence, quality_info = predict_gender(face, return_quality_info=True)
        result["gender"] = gender
        result["gender_confidence"] = confidence
        result["quality"]["gender_info"] = quality_info
    
    # Face encoding
    if do_encoding:
        encoding = encode_face(face, validate=True)
        if encoding:
            result["encoding"] = encoding
        else:
            result["errors"].append("Failed to create encoding")
    
    result["success"] = result["face_valid"] and (not do_encoding or bool(result["encoding"]))
    
    return result


def validate_face_for_registration(face_img) -> Tuple[bool, List[str], Dict]:
    """
    Validate a face image is suitable for registration.
    
    Stricter validation for registration to ensure high quality encodings.
    
    Returns: (is_valid, list of issues, quality_info)
    """
    issues = []
    
    # Basic validation
    is_valid, error_msg = _validate_image(face_img)
    if not is_valid:
        return False, [error_msg], {}
    
    # Size check (stricter for registration)
    h, w = face_img.shape[:2]
    if w < 80 or h < 80:
        issues.append(f"Face too small for registration: {w}x{h}, minimum 80x80")
    
    # Quality check (stricter thresholds)
    quality_score, quality_issues = _check_image_quality(face_img)
    
    if quality_score < 0.6:
        issues.append(f"Image quality too low: {quality_score:.0%}")
    
    issues.extend(quality_issues)
    
    # Try encoding
    encoding = encode_face(face_img, validate=True)
    if not encoding:
        issues.append("Failed to generate face encoding")
    
    quality_info = {
        "score": quality_score,
        "issues": quality_issues,
        "encoding_success": bool(encoding),
        "encoding_length": len(encoding) if encoding else 0
    }
    
    is_valid = len(issues) == 0
    
    return is_valid, issues, quality_info


if __name__ == "__main__":
    check_models()
    print(get_system_info())
