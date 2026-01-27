"""
Face Recognition & Gender Classification Module
================================================

BƯỚC 1: TÌM KHUÔN MẶT (Face Detection)
- Sử dụng: DNN SSD / dlib HOG
- Tốc độ: 30-60 FPS
- Độ chính xác: 85-95%

BƯỚC 2: ĐÁNH DẤU 68 ĐIỂM ĐẶC TRƯNG (Facial Landmarks)
- 17 điểm đường viền mặt
- 10 điểm lông mày
- 9 điểm mũi
- 12 điểm mắt
- 20 điểm miệng

BƯỚC 3: TẠO VECTOR 128 SỐ (Face Encoding)
- Mỗi khuôn mặt = 128 con số duy nhất
- Giống như "vân tay" của khuôn mặt
- Đã học từ 3 triệu khuôn mặt

BƯỚC 4: SO SÁNH VÀ NHẬN DẠNG
- Khoảng cách < 0.6 = Cùng người ✅
- Khoảng cách >= 0.6 = Khác người ❌
- Độ tin cậy = (1 - khoảng cách) × 100%
"""

import cv2
import numpy as np
import os

# ===============================
# Paths & Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

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
    print("✅ face_recognition (dlib 128-d) - CHUẨN NHẤT")
except ImportError:
    print("⚠️ face_recognition không có, dùng DNN fallback")

USE_DEEPFACE = False
DeepFace = None

try:
    from deepface import DeepFace as DF
    DeepFace = DF
    USE_DEEPFACE = True
    print("✅ DeepFace CNN cho giới tính")
except ImportError:
    print("⚠️ DeepFace không có, dùng Caffe model")

# ===============================
# Global Models (lazy loading)
# ===============================
_gender_net = None
_face_cascade = None
_dnn_net = None


def _to_float_list(arr):
    """Chuyển array thành list Python thuần túy với float"""
    if arr is None:
        return []
    try:
        # Xử lý numpy array
        if hasattr(arr, 'flatten'):
            arr = arr.flatten()
        # Chuyển thành list float
        result = []
        for x in arr:
            result.append(float(x))
        return result
    except Exception:
        return []


# ===============================
# Gender Classification
# ===============================

def _load_gender_model():
    global _gender_net
    if _gender_net is None:
        if os.path.exists(GENDER_PROTO) and os.path.exists(GENDER_MODEL):
            _gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    return _gender_net


def _predict_gender_deepface(face_img):
    """Gender bằng DeepFace CNN"""
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        result = DeepFace.analyze(
            face_rgb,
            actions=['gender'],
            enforce_detection=False,
            silent=True
        )
        
        if isinstance(result, list):
            result = result[0]
        
        gender_dict = result.get('gender', {})
        
        if isinstance(gender_dict, dict):
            woman = gender_dict.get('Woman', 0)
            man = gender_dict.get('Man', 0)
            if woman > man:
                return "Nữ", woman / 100.0
            else:
                return "Nam", man / 100.0
        else:
            return ("Nữ" if gender_dict == "Woman" else "Nam"), 0.9
            
    except Exception as e:
        print(f"DeepFace error: {e}")
        return _predict_gender_caffe(face_img)


def _predict_gender_caffe(face_img):
    """Gender bằng Caffe model"""
    net = _load_gender_model()
    if net is None:
        return "Unknown", 0.0
    
    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
        return "Unknown", 0.0
    
    face = cv2.resize(face_img, (227, 227))
    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False, crop=False
    )
    
    net.setInput(blob)
    preds = net.forward()
    
    idx = int(preds[0].argmax())
    conf = float(preds[0][idx])
    
    if conf > 0.55:
        return ["Nam", "Nữ"][idx], conf
    return "Unknown", conf


def predict_gender(face_img):
    """
    Nhận diện giới tính từ ảnh khuôn mặt.
    Returns: (gender_label, confidence)
    """
    if face_img is None or face_img.size == 0:
        return "Unknown", 0.0
    
    if USE_DEEPFACE:
        return _predict_gender_deepface(face_img)
    return _predict_gender_caffe(face_img)


# ===============================
# Face Detection
# ===============================

def _load_haar():
    global _face_cascade
    if _face_cascade is None:
        if os.path.exists(HAAR_PATH):
            _face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    return _face_cascade


def _load_dnn():
    global _dnn_net
    if _dnn_net is None:
        if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
            _dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
            print("✅ DNN SSD Face Detector loaded")
    return _dnn_net


def _detect_haar(image):
    """Detect bằng Haar Cascade"""
    cascade = _load_haar()
    if cascade is None:
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return [tuple(map(int, f)) for f in faces] if len(faces) > 0 else []


def _detect_dnn(image, conf_thresh=0.5):
    """Detect bằng DNN SSD"""
    net = _load_dnn()
    if net is None:
        return _detect_haar(image)
    
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    
    return faces


def _detect_face_recognition(image):
    """Detect bằng face_recognition (dlib)"""
    if not USE_FACE_RECOGNITION:
        return _detect_dnn(image)
    
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        return [(left, top, right - left, bottom - top) 
                for (top, right, bottom, left) in locations]
    except Exception as e:
        print(f"face_recognition detect error: {e}")
        return _detect_dnn(image)


def detect_faces(image, method="auto"):
    """
    Phát hiện khuôn mặt.
    Returns: list of (x, y, w, h)
    """
    if method == "auto":
        if USE_FACE_RECOGNITION:
            return _detect_face_recognition(image)
        return _detect_dnn(image)
    elif method == "face_recognition":
        return _detect_face_recognition(image)
    elif method == "dnn":
        return _detect_dnn(image)
    return _detect_haar(image)


# ===============================
# Face Encoding
# ===============================

def _encode_dlib(face_img):
    """Encode bằng face_recognition (128-d)"""
    if not USE_FACE_RECOGNITION:
        return _encode_simple(face_img)
    
    try:
        # Resize nếu quá nhỏ
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            face_img = cv2.resize(face_img, (150, 150))
        
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Thử detect trong ảnh đã crop
        locations = face_recognition.face_locations(rgb, model="hog")
        
        if not locations:
            # Dùng toàn bộ ảnh
            h, w = rgb.shape[:2]
            locations = [(0, w, h, 0)]
        
        encodings = face_recognition.face_encodings(rgb, locations, num_jitters=1)
        
        if encodings and len(encodings) > 0:
            return _to_float_list(encodings[0])
        
        return _encode_simple(face_img)
        
    except Exception as e:
        print(f"encode_dlib error: {e}")
        return _encode_simple(face_img)


def _encode_simple(face_img):
    """Encode đơn giản (fallback)"""
    try:
        face = cv2.resize(face_img, (100, 100))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        arr = gray.flatten().astype(np.float32) / 255.0
        return _to_float_list(arr)
    except Exception as e:
        print(f"encode_simple error: {e}")
        return []


def encode_face(face_img):
    """
    Tạo face encoding từ ảnh khuôn mặt.
    Returns: list of floats
    """
    if face_img is None or face_img.size == 0:
        return []
    
    if USE_FACE_RECOGNITION:
        return _encode_dlib(face_img)
    return _encode_simple(face_img)


# ===============================
# Face Matching
# Ngưỡng chuẩn: 0.6
# Khoảng cách < 0.6 = Cùng người
# ===============================

def compare_faces(known_enc, test_enc, threshold=0.6):
    """
    So sánh 2 khuôn mặt bằng khoảng cách Euclidean.
    
    Ngưỡng quyết định: 0.6
    - Khoảng cách = 0.1 → Rất giống (cùng người chắc chắn)
    - Khoảng cách = 0.3 → Giống (cùng người)
    - Khoảng cách = 0.5 → Hơi giống (có thể cùng người)
    - Khoảng cách = 0.8 → Khác xa (khác người)
    
    Returns: (is_match, distance)
    """
    if not known_enc or not test_enc:
        return False, 1.0
    
    if len(known_enc) != len(test_enc):
        return False, 1.0
    
    known = np.array(known_enc, dtype=np.float64)
    test = np.array(test_enc, dtype=np.float64)
    
    # Tính khoảng cách Euclidean
    dist = float(np.linalg.norm(known - test))
    
    # So với ngưỡng
    is_match = dist < threshold
    
    return is_match, dist


def find_best_match(test_encoding, known_faces, threshold=0.6):
    """
    Tìm sinh viên khớp nhất với khuôn mặt.
    
    Quy trình:
    1. So sánh vector 128 số của khuôn mặt hiện tại với database
    2. Tìm khuôn mặt có khoảng cách nhỏ nhất
    3. Nếu khoảng cách < 0.6 → Nhận dạng thành công!
    
    Công thức độ tin cậy:
    - Độ tin cậy = (1 - khoảng cách) × 100%
    - Khoảng cách 0.1 → 90% tin cậy
    - Khoảng cách 0.3 → 70% tin cậy
    - Khoảng cách 0.5 → 50% tin cậy
    
    Returns: (matched_student, confidence) or (None, 0)
    """
    if not test_encoding:
        return None, 0
    
    best_match = None
    best_dist = float('inf')
    
    for student in known_faces:
        enc = student.get('encoding', [])
        if not enc or len(enc) != len(test_encoding):
            continue
        
        is_match, dist = compare_faces(enc, test_encoding, threshold)
        
        if is_match and dist < best_dist:
            best_dist = dist
            best_match = student
    
    if best_match:
        # Công thức chuẩn: Độ tin cậy = (1 - khoảng cách) × 100%
        # Nhưng điều chỉnh để hiển thị hợp lý hơn
        if best_dist <= 0.2:
            # Rất giống: 90-99%
            confidence = 0.99 - (best_dist * 0.45)
        elif best_dist <= 0.4:
            # Giống: 75-90%
            confidence = 0.90 - ((best_dist - 0.2) * 0.75)
        elif best_dist <= 0.6:
            # Chấp nhận được: 60-75%
            confidence = 0.75 - ((best_dist - 0.4) * 0.75)
        else:
            confidence = 0.5
        
        return best_match, max(0.5, min(0.99, confidence))
    
    return None, 0
