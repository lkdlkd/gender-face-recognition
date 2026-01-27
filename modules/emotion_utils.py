import cv2
import numpy as np
import os

# ===============================
# Load Emotion Model
# ===============================

try:
    from tensorflow import keras
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    print("⚠️ TensorFlow not available - emotion detection disabled")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "fer2013_mini_XCEPTION.102-0.66.hdf5")

emotion_model = None

EMOTION_LABELS = [
    "Tức giận", "Ghê tởm", "Sợ hãi",
    "Vui vẻ", "Buồn bã", "Ngạc nhiên", "Bình thường"
]

EMOTION_EMOJIS = {
    "Tức giận": "😠",
    "Ghê tởm": "🤢", 
    "Sợ hãi": "😨",
    "Vui vẻ": "😊",
    "Buồn bã": "😢",
    "Ngạc nhiên": "😲",
    "Bình thường": "😐"
}


def load_emotion_model():
    global emotion_model
    if emotion_model is None and EMOTION_AVAILABLE:
        if os.path.exists(EMOTION_MODEL_PATH):
            emotion_model = keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
        else:
            print(f"⚠️ Emotion model not found: {EMOTION_MODEL_PATH}")
    return emotion_model


def predict_emotion(face_img):
    """
    Nhận diện cảm xúc từ ảnh khuôn mặt.
    Returns: (emotion_label, emoji, confidence)
    """
    if not EMOTION_AVAILABLE:
        return "Bình thường", "😐", 0.0
    
    model = load_emotion_model()
    if model is None:
        return "Bình thường", "😐", 0.0
    
    try:
        # Chuyển sang grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize đúng chuẩn Mini-Xception (64x64)
        gray = cv2.resize(gray, (64, 64))
        
        # Normalize
        gray = gray.astype("float32") / 255.0
        
        # Reshape: (1, 64, 64, 1)
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)
        
        # Predict
        preds = model.predict(gray, verbose=0)
        idx = np.argmax(preds)
        
        emotion = EMOTION_LABELS[idx]
        emoji = EMOTION_EMOJIS[emotion]
        confidence = float(preds[0][idx])
        
        return emotion, emoji, confidence
        
    except Exception as e:
        print(f"⚠️ Emotion prediction error: {e}")
        return "Bình thường", "😐", 0.0
