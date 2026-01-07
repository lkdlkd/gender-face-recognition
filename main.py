import cv2
import numpy as np


# Load gender model (Caffe)

gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt",
    "gender_net.caffemodel"
)

if gender_net.empty():
    print("❌ Không load được gender model")
    exit()
else:
    print("✅ Load gender model thành công")

GENDER_LIST = ['Male', 'Female']


# Load face detector (Haar)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Không load được haarcascade")
    exit()
else:
    print("✅ Load haarcascade thành công")


# Hàm loại bỏ bounding box trùng lặp (NMS)

def self_nms(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(area)[::-1]
    
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / area[idxs[1:]]
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return boxes[pick].tolist()


# Hàm nhận diện giới tính (hỗ trợ ảnh xoay)

def detect_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thử detect ở nhiều góc xoay
    best_result = None
    max_faces = 0
    
    for angle in [0, 90, 180, 270]:
        # Xoay ảnh
        if angle == 90:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rotated_gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
            rotated_gray = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = frame.copy()
            rotated_gray = gray.copy()

        # Detect faces với tham số cân bằng
        faces = face_cascade.detectMultiScale(
            rotated_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Loại bỏ các bounding box trùng lặp (NMS)
        if len(faces) > 0:
            faces = self_nms(faces, 0.3)

        # Lưu kết quả tốt nhất (nhiều face nhất)
        if len(faces) > max_faces:
            max_faces = len(faces)
            best_result = (rotated, faces, angle)

    # Không tìm thấy face nào
    if best_result is None:
        return frame

    rotated, faces, angle = best_result
    
    # Vẽ kết quả trên ảnh đã xoay
    for (x, y, w, h) in faces:
        # tránh lỗi crop ngoài ảnh
        if x < 0 or y < 0 or x+w > rotated.shape[1] or y+h > rotated.shape[0]:
            continue

        face = rotated[y:y+h, x:x+w]
        face = cv2.resize(face, (227, 227))

        blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        gender_net.setInput(blob)
        preds = gender_net.forward()

        confidence = preds[0].max()
        gender = GENDER_LIST[preds[0].argmax()] if confidence > 0.6 else "Unknown"

        # ===== Vẽ bounding box =====
        cv2.rectangle(rotated, (x, y), (x+w, y+h), (255, 0, 0), 2)

        label = gender
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        cv2.rectangle(
            rotated,
            (x, y-th-10),
            (x+tw+5, y),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            rotated,
            label,
            (x+2, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Xoay ngược lại về góc ban đầu
    if angle == 90:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

    return rotated

# MENU

print("\n=== CHỌN CHẾ ĐỘ ===")
print("1. Nhận diện từ ảnh")
print("2. Nhận diện từ webcam")
mode = input("Nhập lựa chọn (1/2): ")


# MODE 1: IMAGE


if mode == "1":
    image_path = input("Nhập đường dẫn ảnh (Enter = test.jpg): ")
    if not image_path:
        image_path = "test.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        exit()

    result = detect_gender(img)

    # Resize ảnh để vừa màn hình
    h, w = result.shape[:2]
    max_height = 800
    if h > max_height:
        ratio = max_height / h
        result = cv2.resize(result, (int(w * ratio), max_height), interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow("Gender Recognition - Image", result)
    
    print("\n📌 Nhấn 's' để lưu ảnh kết quả")
    print("📌 Nhấn phím bất kỳ để thoát...")
    
    key = cv2.waitKey(0)
    
    # Lưu ảnh nếu nhấn 's'
    if key == ord('s'):
        output_path = "result_" + image_path.replace("\\", "/").split("/")[-1]
        cv2.imwrite(output_path, result)
        print(f"✅ Đã lưu ảnh: {output_path}")
    
    cv2.destroyAllWindows()



# MODE 2: WEBCAM REAL TIME

elif mode == "2":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_gender(frame)
        cv2.imshow("Gender Recognition - Webcam", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("❌ Lựa chọn không hợp lệ!")
