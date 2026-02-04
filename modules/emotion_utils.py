"""
Emotion Recognition Module
==========================

BƯỚC 1: TIỀN XỬ LÝ ẢNH (Image Preprocessing)
- Kiểm tra tính hợp lệ của ảnh đầu vào
- Chuyển đổi grayscale và cân bằng histogram
- Resize về kích thước chuẩn 64x64

BƯỚC 2: NHẬN DẠNG CẢM XÚC (Emotion Detection)
- Sử dụng model Mini-XCEPTION (train trên FER2013)
- Output: 7 cảm xúc cơ bản
- Accuracy: ~66% trên FER2013 dataset

BƯỚC 3: XÁC THỰC KẾT QUẢ (Result Validation)
- Ngưỡng tin cậy tối thiểu: 0.25
- Kiểm tra chất lượng ảnh trước khi xử lý
- Fallback an toàn khi lỗi
"""

import cv2
import numpy as np
import os

# ===============================
# Configuration
# ===============================

# Ngưỡng tin cậy tối thiểu để chấp nhận kết quả
MIN_CONFIDENCE_THRESHOLD = 0.25

# Kích thước ảnh tối thiểu để xử lý
MIN_FACE_SIZE = 20

# Kích thước chuẩn cho model
MODEL_INPUT_SIZE = (64, 64)

# ===============================
# Load Emotion Model
# ===============================

try:
    from tensorflow import keras
    EMOTION_AVAILABLE = True
    print("[OK] TensorFlow loaded - Emotion detection enabled")
except ImportError:
    EMOTION_AVAILABLE = False
    print("[WARN] TensorFlow not available - emotion detection disabled")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "fer2013_mini_XCEPTION.102-0.66.hdf5")

emotion_model = None
_model_load_error = None

EMOTION_LABELS = [
    "Tức giận", "Ghê tởm", "Sợ hãi",
    "Vui vẻ", "Buồn bã", "Ngạc nhiên", "Bình thường"
]

EMOTION_EMOJIS = {
    "Tức giận": "Angry",
    "Ghê tởm": "Disgust", 
    "Sợ hãi": "Fear",
    "Vui vẻ": "Happy",
    "Buồn bã": "Sad",
    "Ngạc nhiên": "Surprise",
    "Bình thường": "Neutral"
}

# Default fallback values
DEFAULT_EMOTION = "Bình thường"
DEFAULT_EMOJI = "Neutral"
DEFAULT_CONFIDENCE = 0.0


# ===============================
# Validation Functions
# ===============================

def _validate_image(img):
    """
    Xác thực tính hợp lệ của ảnh đầu vào.
    Returns: (is_valid, error_message)
    """
    if img is None:
        return False, "Image is None"
    
    if not hasattr(img, 'shape'):
        return False, "Invalid image format - no shape attribute"
    
    if not hasattr(img, 'size') or img.size == 0:
        return False, "Image is empty"
    
    if len(img.shape) < 2:
        return False, f"Invalid image dimensions: {img.shape}"
    
    h, w = img.shape[:2]
    
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False, f"Image too small: {w}x{h}, minimum: {MIN_FACE_SIZE}x{MIN_FACE_SIZE}"
    
    if not np.isfinite(img).all():
        return False, "Image contains invalid values (NaN or Inf)"
    
    return True, "OK"


def _validate_predictions(preds):
    """
    Xác thực kết quả dự đoán.
    Returns: (is_valid, error_message)
    """
    if preds is None:
        return False, "Predictions is None"
    
    if not hasattr(preds, 'shape') or len(preds.shape) < 2:
        return False, "Invalid predictions format"
    
    if preds.shape[1] != len(EMOTION_LABELS):
        return False, f"Expected {len(EMOTION_LABELS)} classes, got {preds.shape[1]}"
    
    if not np.isfinite(preds).all():
        return False, "Predictions contain invalid values"
    
    # Kiểm tra tổng xác suất (cho softmax output)
    prob_sum = np.sum(preds[0])
    if not (0.99 <= prob_sum <= 1.01):
        print(f"[WARN] Probability sum: {prob_sum:.4f} (expected ~1.0)")
    
    return True, "OK"


def _check_image_quality(gray_img):
    """
    Kiểm tra chất lượng ảnh (độ tương phản, độ sáng).
    Returns: (quality_score, issues_list)
    """
    issues = []
    quality_score = 1.0
    
    # Kiểm tra độ sáng trung bình
    mean_brightness = np.mean(gray_img)
    if mean_brightness < 30:
        issues.append("Image too dark")
        quality_score *= 0.7
    elif mean_brightness > 225:
        issues.append("Image too bright")
        quality_score *= 0.7
    
    # Kiểm tra độ tương phản (standard deviation)
    std_contrast = np.std(gray_img)
    if std_contrast < 20:
        issues.append("Low contrast")
        quality_score *= 0.8
    
    # Kiểm tra độ mờ (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    if laplacian_var < 50:
        issues.append("Image may be blurry")
        quality_score *= 0.85
    
    return quality_score, issues


# ===============================
# Preprocessing Functions
# ===============================

def _preprocess_face(face_img):
    """
    Tiền xử lý ảnh khuôn mặt trước khi đưa vào model.
    Returns: preprocessed image array or None if failed
    """
    try:
        # Chuyển sang grayscale
        if len(face_img.shape) == 3:
            if face_img.shape[2] == 4:  # BGRA
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        elif len(face_img.shape) == 2:
            gray = face_img.copy()
        else:
            return None
        
        # Cân bằng histogram để cải thiện độ tương phản
        gray = cv2.equalizeHist(gray)
        
        # Resize về kích thước chuẩn
        gray = cv2.resize(gray, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        
        # Normalize ve [0, 1]
        gray = gray.astype("float32") / 255.0
        
        # Reshape cho model: (1, 64, 64, 1)
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)
        
        return gray
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None


# ===============================
# Model Loading
# ===============================

def load_emotion_model():
    """
    Load emotion recognition model với xác thực đầy đủ.
    Returns: loaded model or None
    """
    global emotion_model, _model_load_error
    
    if emotion_model is not None:
        return emotion_model
    
    if not EMOTION_AVAILABLE:
        _model_load_error = "TensorFlow not available"
        return None
    
    if not os.path.exists(EMOTION_MODEL_PATH):
        _model_load_error = f"Model file not found: {EMOTION_MODEL_PATH}"
        print(f"[ERROR] {_model_load_error}")
        return None
    
    # Kiểm tra kích thước file
    file_size = os.path.getsize(EMOTION_MODEL_PATH)
    if file_size < 1000:  # File quá nhỏ, có thể bị lỗi
        _model_load_error = f"Model file too small: {file_size} bytes"
        print(f"[ERROR] {_model_load_error}")
        return None
    
    try:
        emotion_model = keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
        
        # Xác thực model đã load
        if emotion_model is None:
            _model_load_error = "Model loaded but is None"
            return None
        
        # Kiểm tra input shape
        expected_shape = (None, 64, 64, 1)
        if hasattr(emotion_model, 'input_shape'):
            if emotion_model.input_shape != expected_shape:
                print(f"[WARN] Unexpected input shape: {emotion_model.input_shape}")
        
        print(f"[OK] Emotion model loaded: {file_size / 1024:.1f} KB")
        _model_load_error = None
        return emotion_model
        
    except Exception as e:
        _model_load_error = str(e)
        print(f"[ERROR] Failed to load emotion model: {e}")
        return None


# ===============================
# Main Prediction Function
# ===============================

def predict_emotion(face_img, return_all_scores=False):
    """
    Nhận diện cảm xúc từ ảnh khuôn mặt với xác thực đầy đủ.
    
    Args:
        face_img: Ảnh khuôn mặt (BGR format)
        return_all_scores: Trả về tất cả điểm số các cảm xúc
    
    Returns: 
        (emotion_label, emoji, confidence) 
        hoặc (emotion_label, emoji, confidence, all_scores) nếu return_all_scores=True
    """
    default_result = (DEFAULT_EMOTION, DEFAULT_EMOJI, DEFAULT_CONFIDENCE)
    
    # === BƯỚC 1: Xác thực đầu vào ===
    is_valid, error_msg = _validate_image(face_img)
    if not is_valid:
        print(f"[WARN] Invalid input: {error_msg}")
        if return_all_scores:
            return default_result + ({},)
        return default_result
    
    # === BUOC 2: Kiem tra model ===
    if not EMOTION_AVAILABLE:
        if return_all_scores:
            return default_result + ({},)
        return default_result
    
    model = load_emotion_model()
    if model is None:
        if return_all_scores:
            return default_result + ({},)
        return default_result
    
    try:
        # === BUOC 3: Kiem tra chat luong anh ===
        if len(face_img.shape) == 3:
            gray_check = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_check = face_img
        
        quality_score, quality_issues = _check_image_quality(gray_check)
        if quality_issues:
            print(f"[INFO] Image quality issues: {', '.join(quality_issues)}")
        
        # === BUOC 4: Tien xu ly ===
        preprocessed = _preprocess_face(face_img)
        if preprocessed is None:
            if return_all_scores:
                return default_result + ({},)
            return default_result
        
        # === BUOC 5: Du doan ===
        preds = model.predict(preprocessed, verbose=0)
        
        # === BUOC 6: Xac thuc ket qua ===
        is_valid, error_msg = _validate_predictions(preds)
        if not is_valid:
            print(f"[WARN] Invalid predictions: {error_msg}")
            if return_all_scores:
                return default_result + ({},)
            return default_result
        
        # === BUOC 7: Xu ly ket qua ===
        idx = np.argmax(preds[0])
        confidence = float(preds[0][idx])
        
        # Dieu chinh confidence theo chat luong anh
        adjusted_confidence = confidence * quality_score
        
        # Kiem tra nguong tin cay
        if adjusted_confidence < MIN_CONFIDENCE_THRESHOLD:
            print(f"[INFO] Low confidence: {adjusted_confidence:.2%}, returning default")
            emotion = DEFAULT_EMOTION
            emoji = DEFAULT_EMOJI
        else:
            emotion = EMOTION_LABELS[idx]
            emoji = EMOTION_EMOJIS[emotion]
        
        # Tao dict tat ca diem so
        if return_all_scores:
            all_scores = {
                EMOTION_LABELS[i]: float(preds[0][i]) 
                for i in range(len(EMOTION_LABELS))
            }
            return emotion, emoji, adjusted_confidence, all_scores
        
        return emotion, emoji, adjusted_confidence
        
    except Exception as e:
        print(f"[ERROR] Emotion prediction failed: {e}")
        if return_all_scores:
            return default_result + ({},)
        return default_result


# ===============================
# Utility Functions
# ===============================

def get_emotion_info():
    """Trả về thông tin về module emotion recognition"""
    return {
        "available": EMOTION_AVAILABLE,
        "model_loaded": emotion_model is not None,
        "model_path": EMOTION_MODEL_PATH,
        "model_exists": os.path.exists(EMOTION_MODEL_PATH),
        "load_error": _model_load_error,
        "num_emotions": len(EMOTION_LABELS),
        "emotions": EMOTION_LABELS,
        "min_confidence": MIN_CONFIDENCE_THRESHOLD,
        "min_face_size": MIN_FACE_SIZE,
    }


def get_all_emotions():
    """Trả về danh sách tất cả cảm xúc và emoji tương ứng"""
    return [
        {"label": label, "emoji": EMOTION_EMOJIS[label]}
        for label in EMOTION_LABELS
    ]


def check_emotion_model():
    """Kiểm tra và in thông tin model"""
    print("\n" + "="*50)
    print("EMOTION MODEL STATUS")
    print("="*50)
    
    info = get_emotion_info()
    
    print(f"TensorFlow available: {info['available']}")
    print(f"Model path: {info['model_path']}")
    print(f"Model exists: {info['model_exists']}")
    
    if info['model_exists']:
        size_mb = os.path.getsize(EMOTION_MODEL_PATH) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
    
    print(f"Model loaded: {info['model_loaded']}")
    
    if info['load_error']:
        print(f"Load error: {info['load_error']}")
    
    print(f"Supported emotions: {info['num_emotions']}")
    print(f"Min confidence threshold: {info['min_confidence']:.0%}")
    
    print("="*50 + "\n")
    
    return info['model_loaded']


if __name__ == "__main__":
    check_emotion_model()
    print(get_emotion_info())
