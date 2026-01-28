# HE THONG DIEM DANH NHAN VIEN BANG NHAN DIEN KHUON MAT + GIOI TINH

## Ten de tai
**Xay dung he thong diem danh nhan vien bang nhan dien khuon mat ket hop phan tich gioi tinh**

He thong web diem danh tu dong, gan lien voi thuc te doanh nghiep.

---

## Tinh nang chinh

### Quan ly nhan vien
- Dang ky nhan vien moi (Ma NV, ho ten, phong ban)
- Chup va luu anh khuon mat
- Tu dong xac dinh gioi tinh bang AI
- Xem danh sach nhan vien
- Xoa nhan vien

### Diem danh realtime
- Camera webcam realtime
- Phat hien khuon mat (Face Detection)
- Nhan dien nhan vien (Face Recognition)
- Phan tich gioi tinh (Gender Classification)
- Phan tich cam xuc (Emotion Recognition)
- Che do tu dong diem danh

### Bao cao
- Lich su diem danh
- Thong ke theo ngay/tuan
- Thong ke theo gioi tinh
- Xuat bao cao CSV

---

## Cong nghe su dung

| Thanh phan | Cong nghe |
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

## Cai dat

### 1. Yeu cau he thong
- Python 3.8+
- Webcam
- RAM >= 4GB

### 2. Cai dat thu vien

```bash
pip install -r requirements.txt
```

### 3. Cau truc thu muc

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
    ├── register.html           # Dang ky nhan vien
    ├── attendance.html         # Diem danh
    ├── employees.html          # Danh sach NV
    └── history.html            # Lich su
```

---

## Su dung

### Chay chuong trinh

```bash
python app.py
```

### Truy cap website

Mo trinh duyet tai: **http://localhost:5000**

### Huong dan su dung

#### 1. Dang ky nhan vien
1. Vao menu "Dang ky NV"
2. Nhap Ma NV, ho ten, phong ban
3. Bat camera va chup anh khuon mat
4. He thong tu dong xac dinh gioi tinh
5. Nhan "Dang ky"

#### 2. Diem danh
1. Vao menu "Diem danh"
2. Bat camera
3. Nhan "Diem danh" hoac bat che do tu dong
4. He thong nhan dien va ghi diem danh

#### 3. Xem lich su
1. Vao menu "Lich su"
2. Xem danh sach diem danh
3. Xuat CSV neu can

---

## Ket qua hien thi

### Tren camera:
- Bounding box quanh khuon mat
- Ten nhan vien (hoac "Unknown")
- Gioi tinh
- Cam xuc

### Thong tin diem danh:
- Ma NV, ho ten, phong ban
- Thoi gian diem danh
- Gioi tinh dang ky vs phat hien
- Cam xuc
- Trang thai (Co mat / Muon)

---

## Quy trinh hoat dong

```
1. DANG KY
   --> Nhap thong tin
   --> Chup anh
   --> AI xac dinh gioi tinh
   --> Luu face encoding vao DB

2. DIEM DANH
   --> Mo camera
   --> Detect khuon mat
   --> So khop voi DB
   --> Nhan dien gioi tinh + cam xuc
   --> Ghi diem danh

3. BAO CAO
   --> Thong ke %
   --> Xuat CSV
```

---

## Phuong phap AI

### 1. Face Detection
- **Phuong phap:** Haar Cascade Classifier
- **Mo ta:** Phat hien vi tri khuon mat trong anh

### 2. Face Recognition
- **Phuong phap:** Histogram flattening + Cosine similarity
- **Mo ta:** So khop face encoding voi database

### 3. Gender Classification
- **Phuong phap:** CNN (Caffe Model)
- **Output:** Nam / Nu

### 4. Emotion Recognition
- **Phuong phap:** Mini-Xception CNN
- **Dataset:** FER-2013
- **Output:** 7 cam xuc (Vui ve, Buon ba, Tuc gian, So hai, Ngac nhien, Ghe tom, Binh thuong)

---

## Tac gia

De tai mon hoc: **Xu ly anh**

---

## License

MIT License
