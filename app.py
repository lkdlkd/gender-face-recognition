from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import numpy as np
import os
import base64
import json
from datetime import datetime
import io

# Import modules
from modules.database import (
    add_student, get_all_students, get_student_by_id, delete_student,
    get_all_face_encodings, add_attendance, get_attendance_today,
    get_attendance_history, check_already_attended_today, get_attendance_stats
)
from modules.face_utils import (
    detect_faces, predict_gender, encode_face, find_best_match
)
from modules.emotion_utils import predict_emotion

app = Flask(__name__)
app.config['SECRET_KEY'] = 'attendance_system_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'faces')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ===============================
# ROUTES - Pages
# ===============================

@app.route('/')
def index():
    """Trang chủ - Dashboard"""
    stats = get_attendance_stats()
    return render_template('index.html', stats=stats)


@app.route('/register')
def register_page():
    """Trang đăng ký sinh viên"""
    return render_template('register.html')


@app.route('/students')
def students_page():
    """Trang danh sách sinh viên"""
    students = get_all_students()
    return render_template('students.html', students=students)


@app.route('/attendance')
def attendance_page():
    """Trang điểm danh"""
    return render_template('attendance.html')


@app.route('/history')
def history_page():
    """Trang lịch sử điểm danh"""
    records = get_attendance_history(days=30)
    return render_template('history.html', records=records)


# ===============================
# API ROUTES
# ===============================

@app.route('/api/register', methods=['POST'])
def api_register():
    """API đăng ký sinh viên mới"""
    try:
        data = request.get_json()
        
        student_code = data.get('student_code', '').strip()
        name = data.get('name', '').strip()
        class_name = data.get('class_name', '').strip()
        image_data = data.get('image')  # Base64 encoded image
        
        if not all([student_code, name, class_name, image_data]):
            return jsonify({'success': False, 'message': 'Vui lòng điền đầy đủ thông tin và chụp ảnh khuôn mặt'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Không thể đọc ảnh'})
        
        # Flip ảnh từ webcam (do webcam thường bị mirror)
        image = cv2.flip(image, 1)
        
        # Detect face
        faces = detect_faces(image)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'Không phát hiện khuôn mặt trong ảnh'})
        
        if len(faces) > 1:
            return jsonify({'success': False, 'message': 'Phát hiện nhiều khuôn mặt. Vui lòng chỉ để 1 người trong khung hình'})
        
        # Get face region
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        
        # Predict gender
        gender, _ = predict_gender(face_img)
        
        # Encode face
        face_encoding = encode_face(face_img)
        
        # Save face image
        filename = f"{student_code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, face_img)
        
        # Save to database
        student_id = add_student(
            student_code=student_code,
            name=name,
            class_name=class_name,
            gender=gender,
            face_encoding=face_encoding,
            face_image=filename
        )
        
        if student_id:
            return jsonify({
                'success': True,
                'message': f'Đăng ký thành công! Giới tính: {gender}',
                'student_id': student_id,
                'gender': gender
            })
        else:
            return jsonify({'success': False, 'message': 'Mã sinh viên đã tồn tại'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API nhận diện khuôn mặt để điểm danh"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Không có ảnh'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Không thể đọc ảnh'})
        
        # Flip ảnh từ webcam (do webcam thường bị mirror)
        image = cv2.flip(image, 1)
        
        # Lưu width để flip tọa độ bbox sau này
        img_width = image.shape[1]
        
        # Detect faces
        faces = detect_faces(image)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'Không phát hiện khuôn mặt'})
        
        results = []
        known_faces = get_all_face_encodings()
        
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            
            # Encode face
            test_encoding = encode_face(face_img)
            
            # Find match
            matched_student, confidence = find_best_match(test_encoding, known_faces, threshold=0.5)
            
            # Predict gender
            gender, gender_conf = predict_gender(face_img)
            
            # Predict emotion
            emotion, emoji, emotion_conf = predict_emotion(face_img)
            
            # Flip tọa độ x để khớp với video gốc trên browser (vì đã flip ảnh khi xử lý)
            flipped_x = img_width - x - w
            
            if matched_student:
                # Check if already attended today
                already_attended = check_already_attended_today(matched_student['id'])
                
                if not already_attended:
                    # Add attendance record
                    add_attendance(
                        student_id=matched_student['id'],
                        gender_detected=gender,
                        emotion_detected=emotion,
                        confidence=confidence,
                        status='present'
                    )
                    status = 'Đã điểm danh thành công!'
                else:
                    status = 'Đã điểm danh hôm nay rồi'
                
                results.append({
                    'found': True,
                    'student_code': matched_student['student_code'],
                    'name': matched_student['name'],
                    'class_name': matched_student['class_name'],
                    'gender': gender,
                    'emotion': f"{emoji} {emotion}",
                    'confidence': round(confidence * 100, 1),
                    'status': status,
                    'already_attended': already_attended,
                    'bbox': {'x': int(flipped_x), 'y': int(y), 'w': int(w), 'h': int(h)}
                })
            else:
                results.append({
                    'found': False,
                    'message': 'Không nhận diện được',
                    'gender': gender,
                    'emotion': f"{emoji} {emotion}",
                    'bbox': {'x': int(flipped_x), 'y': int(y), 'w': int(w), 'h': int(h)}
                })
        
        return jsonify({'success': True, 'faces': results})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})


@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def api_delete_student(student_id):
    """API xóa sinh viên"""
    try:
        success = delete_student(student_id)
        if success:
            return jsonify({'success': True, 'message': 'Đã xóa sinh viên'})
        else:
            return jsonify({'success': False, 'message': 'Không tìm thấy sinh viên'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})


@app.route('/api/attendance/today')
def api_attendance_today():
    """API lấy danh sách điểm danh hôm nay"""
    records = get_attendance_today()
    data = []
    for record in records:
        data.append({
            'id': record['id'],
            'student_code': record['student_code'],
            'name': record['name'],
            'class_name': record['class_name'],
            'check_in_time': record['check_in_time'],
            'gender_detected': record['gender_detected'],
            'emotion_detected': record['emotion_detected'],
            'status': record['status']
        })
    return jsonify({'success': True, 'records': data})


@app.route('/api/stats')
def api_stats():
    """API lấy thống kê"""
    stats = get_attendance_stats()
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/export/csv')
def api_export_csv():
    """Xuất báo cáo CSV"""
    records = get_attendance_history(days=30)
    
    # Create CSV content
    csv_lines = ['MSSV,Họ tên,Lớp,Thời gian,Giới tính,Cảm xúc,Trạng thái']
    for record in records:
        csv_lines.append(f"{record['student_code']},{record['name']},{record['class_name']},{record['check_in_time']},{record['gender_detected']},{record['emotion_detected']},{record['status']}")
    
    csv_content = '\n'.join(csv_lines)
    
    # Return as file download
    output = io.BytesIO()
    output.write(csv_content.encode('utf-8-sig'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_report_{datetime.now().strftime("%Y%m%d")}.csv'
    )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🥉 HỆ THỐNG ĐIỂM DANH SINH VIÊN")
    print("📌 Nhận diện khuôn mặt + Giới tính + Cảm xúc")
    print("="*50)
    print("\n🌐 Mở trình duyệt tại: http://localhost:5000")
    print("📌 Nhấn Ctrl+C để dừng server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
