# Gender Recognition System

Hệ thống nhận diện giới tính sử dụng OpenCV và Deep Learning (Caffe Model).

## 📋 Mô tả

Ứng dụng nhận diện giới tính (Nam/Nữ) từ khuôn mặt trong ảnh hoặc webcam realtime.

**Tính năng:**
- ✅ Nhận diện giới tính từ ảnh
- ✅ Nhận diện giới tính từ webcam realtime
- ✅ Tự động xoay ảnh để phát hiện khuôn mặt (0°, 90°, 180°, 270°)
- ✅ Loại bỏ bounding box trùng lặp (NMS)
- ✅ Lưu kết quả ảnh

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.7+
- Webcam (nếu sử dụng chế độ realtime)

### 2. Cài đặt thư viện

```bash
pip install opencv-python numpy
```
### 3. Tải model

**Tải các file model sau và đặt vào thư mục gốc:**

1. **gender_deploy.prototxt** - [Download](https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/gender_net.prototxt)
2. **gender_net.caffemodel** - [Download](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel)

**Cấu trúc thư mục:**
```
face_recognition/
├── main.py
├── gender_deploy.prototxt
├── gender_net.caffemodel
├── test.jpg (ảnh test, không bắt buộc)
└── README.md
```
## 🚀 Sử dụng

### Chạy chương trình

```bash
python main.py
```
## 📊 Kết quả

- **Male**: Nam
- **Female**: Nữ
- **Unknown**: Không xác định (confidence < 60%)
