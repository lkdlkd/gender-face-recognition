# 🥉 HỆ THỐNG ĐIỂM DANH SINH VIÊN BẰNG NHẬN DIỆN KHUÔN MẶT + GIỚI TÍNH

## 📌 Tên đề tài
**Xây dựng hệ thống điểm danh sinh viên bằng nhận diện khuôn mặt kết hợp phân tích giới tính**

Hệ thống web điểm danh tự động, gắn liền với thực tế trường học.

---

## 🎯 Tính năng chính

### 🎓 Quản lý sinh viên
- ✅ Đăng ký sinh viên mới (MSSV, họ tên, lớp)
- ✅ Chụp và lưu ảnh khuôn mặt
- ✅ Tự động xác định giới tính bằng AI
- ✅ Xem danh sách sinh viên
- ✅ Xóa sinh viên

### 📸 Điểm danh realtime
- ✅ Camera webcam realtime
- ✅ Phát hiện khuôn mặt (Face Detection)
- ✅ Nhận diện sinh viên (Face Recognition)
- ✅ Phân tích giới tính (Gender Classification)
- ✅ Phân tích cảm xúc (Emotion Recognition)
- ✅ Chế độ tự động điểm danh

### 📊 Báo cáo
- ✅ Lịch sử điểm danh
- ✅ Thống kê theo ngày/tuần
- ✅ Thống kê theo giới tính
- ✅ Xuất báo cáo CSV

---

## ⚙️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python Flask |
| AI/ML | OpenCV DNN, TensorFlow/Keras |
| Face Detection | Haar Cascade |
| Gender Model | CNN (Caffe) |
| Emotion Model | Mini-Xception (FER-2013) |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Camera | WebRTC API |

---

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- Webcam
- RAM >= 4GB

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu trúc thư mục

```
gender-face-recognition/
├── app.py                      # Flask application
├── requirements.txt            # Dependencies
├── database.db                 # SQLite database (auto-created)
│
├── modules/                    # Backend modules
│   ├── __init__.py
│   ├── database.py             # Database CRUD
│   ├── face_utils.py           # Face detection & recognition
│   └── emotion_utils.py        # Emotion recognition
│
├── models/                     # AI Models
│   ├── haarcascade_frontalface_default.xml
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   └── fer2013_mini_XCEPTION.102-0.66.hdf5
│
├── static/
│   ├── css/style.css           # Styles
│   └── uploads/faces/          # Uploaded face images
│
└── templates/                  # HTML templates
    ├── base.html
    ├── index.html              # Dashboard
    ├── register.html           # Đăng ký sinh viên
    ├── attendance.html         # Điểm danh
    ├── students.html           # Danh sách SV
    └── history.html            # Lịch sử
```

---

## 🚀 Sử dụng

### Chạy chương trình

```bash
python app.py
```

### Truy cập website

Mở trình duyệt tại: **http://localhost:5000**

### Hướng dẫn sử dụng

#### 1️⃣ Đăng ký sinh viên
1. Vào menu "Đăng ký SV"
2. Nhập MSSV, họ tên, lớp
3. Bật camera và chụp ảnh khuôn mặt
4. Hệ thống tự động xác định giới tính
5. Nhấn "Đăng ký"

#### 2️⃣ Điểm danh
1. Vào menu "Điểm danh"
2. Bật camera
3. Nhấn "Điểm danh" hoặc bật chế độ tự động
4. Hệ thống nhận diện và ghi điểm danh

#### 3️⃣ Xem lịch sử
1. Vào menu "Lịch sử"
2. Xem danh sách điểm danh
3. Xuất CSV nếu cần

---

## 📊 Kết quả hiển thị

### Trên camera:
- 📦 Bounding box quanh khuôn mặt
- 🏷️ Tên sinh viên (hoặc "Unknown")
- ♂️♀️ Giới tính
- 😊 Cảm xúc

### Thông tin điểm danh:
- MSSV, họ tên, lớp
- Thời gian điểm danh
- Giới tính đăng ký vs phát hiện
- Cảm xúc
- Trạng thái (Có mặt / Muộn)

---

## 🧪 Quy trình hoạt động

```
1️⃣ ĐĂNG KÝ
   └─> Nhập thông tin
   └─> Chụp ảnh
   └─> AI xác định giới tính
   └─> Lưu face encoding vào DB

2️⃣ ĐIỂM DANH
   └─> Mở camera
   └─> Detect khuôn mặt
   └─> So khớp với DB
   └─> Nhận diện giới tính + cảm xúc
   └─> Ghi điểm danh

3️⃣ BÁO CÁO
   └─> Thống kê %
   └─> Xuất CSV
```

---

## 📚 Phương pháp AI

### 1. Face Detection
- **Phương pháp:** Haar Cascade Classifier
- **Mô tả:** Phát hiện vị trí khuôn mặt trong ảnh

### 2. Face Recognition
- **Phương pháp:** Histogram flattening + Cosine similarity
- **Mô tả:** So khớp face encoding với database

### 3. Gender Classification
- **Phương pháp:** CNN (Caffe Model)
- **Output:** Nam / Nữ

### 4. Emotion Recognition
- **Phương pháp:** Mini-Xception CNN
- **Dataset:** FER-2013
- **Output:** 7 cảm xúc (Vui vẻ, Buồn bã, Tức giận, Sợ hãi, Ngạc nhiên, Ghê tởm, Bình thường)

---

## 👨‍💻 Tác giả

Đề tài môn học: **Xử lý ảnh**

---

## 📝 License

MIT License