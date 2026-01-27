import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database.db")


def get_connection():
    """Tạo kết nối đến database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo database và các bảng"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Bảng sinh viên
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            gender TEXT,
            face_encoding TEXT,
            face_image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Bảng điểm danh
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            gender_detected TEXT,
            emotion_detected TEXT,
            confidence REAL,
            status TEXT DEFAULT 'present',
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[OK] Database initialized successfully!")


# ===============================
# Student CRUD Operations
# ===============================

def add_student(student_code, name, class_name, gender, face_encoding, face_image):
    """Thêm sinh viên mới"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # face_encoding đã là list Python thuần túy từ face_utils
        if face_encoding is not None and len(face_encoding) > 0:
            encoding_json = json.dumps(face_encoding)
        else:
            encoding_json = None
            
        cursor.execute('''
            INSERT INTO students (student_code, name, class_name, gender, face_encoding, face_image)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_code, name, class_name, gender, encoding_json, face_image))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        print(f"add_student error: {e}")
        return None
    finally:
        conn.close()


def get_all_students(search=None):
    """Lấy danh sách tất cả sinh viên với tùy chọn tìm kiếm"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if search:
        search_term = f'%{search}%'
        cursor.execute('''
            SELECT * FROM students 
            WHERE student_code LIKE ? 
               OR name LIKE ? 
               OR class_name LIKE ?
            ORDER BY created_at DESC
        ''', (search_term, search_term, search_term))
    else:
        cursor.execute('SELECT * FROM students ORDER BY created_at DESC')
    
    students = cursor.fetchall()
    conn.close()
    return students


def get_student_by_id(student_id):
    """Lấy thông tin sinh viên theo ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
    student = cursor.fetchone()
    conn.close()
    return student


def get_student_by_code(student_code):
    """Lấy thông tin sinh viên theo mã sinh viên"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE student_code = ?', (student_code,))
    student = cursor.fetchone()
    conn.close()
    return student


def delete_student(student_id):
    """Xóa sinh viên"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def get_all_face_encodings():
    """Lấy tất cả face encodings từ database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, student_code, name, class_name, gender, face_encoding FROM students WHERE face_encoding IS NOT NULL')
    students = cursor.fetchall()
    conn.close()
    
    result = []
    for student in students:
        if student['face_encoding']:
            encoding = json.loads(student['face_encoding'])
            result.append({
                'id': student['id'],
                'student_code': student['student_code'],
                'name': student['name'],
                'class_name': student['class_name'],
                'gender': student['gender'],
                'encoding': encoding
            })
    return result


# ===============================
# Attendance CRUD Operations
# ===============================

def add_attendance(student_id, gender_detected, emotion_detected, confidence, status='present'):
    """Ghi điểm danh"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attendance (student_id, gender_detected, emotion_detected, confidence, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (student_id, gender_detected, emotion_detected, confidence, status))
    conn.commit()
    attendance_id = cursor.lastrowid
    conn.close()
    return attendance_id


def get_attendance_today():
    """Lấy danh sách điểm danh hôm nay"""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT a.*, s.student_code, s.name, s.class_name, s.gender as registered_gender
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE DATE(a.check_in_time) = ?
        ORDER BY a.check_in_time DESC
    ''', (today,))
    records = cursor.fetchall()
    conn.close()
    return records


def get_attendance_history(days=7, search=None, date=None):
    """Lấy lịch sử điểm danh với tùy chọn tìm kiếm và lọc theo ngày"""
    conn = get_connection()
    cursor = conn.cursor()
    
    base_query = '''
        SELECT a.*, s.student_code, s.name, s.class_name, s.gender as registered_gender
        FROM attendance a
        JOIN students s ON a.student_id = s.id
    '''
    
    conditions = []
    params = []
    
    # Lọc theo ngày cụ thể hoặc N ngày gần nhất
    if date:
        conditions.append("DATE(a.check_in_time) = ?")
        params.append(date)
    else:
        conditions.append("a.check_in_time >= datetime('now', ?)")
        params.append(f'-{days} days')
    
    # Tìm kiếm theo MSSV, tên, lớp
    if search:
        search_term = f'%{search}%'
        conditions.append("(s.student_code LIKE ? OR s.name LIKE ? OR s.class_name LIKE ?)")
        params.extend([search_term, search_term, search_term])
    
    query = base_query + ' WHERE ' + ' AND '.join(conditions) + ' ORDER BY a.check_in_time DESC'
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()
    return records


def get_attendance_dates():
    """Lấy danh sách các ngày có điểm danh"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT DATE(check_in_time) as date 
        FROM attendance 
        ORDER BY date DESC
        LIMIT 60
    ''')
    dates = [row['date'] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_attendance_stats_by_date(date=None):
    """Thống kê điểm danh theo ngày"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Tổng số sinh viên
    cursor.execute('SELECT COUNT(*) as total FROM students')
    total_students = cursor.fetchone()['total']
    
    # Số sinh viên điểm danh trong ngày
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) as attended 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
    ''', (date,))
    attended = cursor.fetchone()['attended']
    
    # Thống kê theo trạng thái
    cursor.execute('''
        SELECT status, COUNT(*) as count 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
        GROUP BY status
    ''', (date,))
    status_stats = {row['status']: row['count'] for row in cursor.fetchall()}
    
    # Thống kê theo lớp
    cursor.execute('''
        SELECT s.class_name, COUNT(DISTINCT a.student_id) as count
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE DATE(a.check_in_time) = ?
        GROUP BY s.class_name
        ORDER BY count DESC
    ''', (date,))
    class_stats = {row['class_name']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'date': date,
        'total_students': total_students,
        'attended': attended,
        'absent': total_students - attended,
        'attendance_rate': round(attended / total_students * 100, 1) if total_students > 0 else 0,
        'status_stats': status_stats,
        'class_stats': class_stats
    }


def check_already_attended_today(student_id):
    """Kiểm tra sinh viên đã điểm danh hôm nay chưa"""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COUNT(*) as count FROM attendance 
        WHERE student_id = ? AND DATE(check_in_time) = ?
    ''', (student_id, today))
    result = cursor.fetchone()
    conn.close()
    return result['count'] > 0


def get_attendance_stats():
    """Thống kê điểm danh"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tổng số sinh viên
    cursor.execute('SELECT COUNT(*) as total FROM students')
    total_students = cursor.fetchone()['total']
    
    # Số sinh viên đã điểm danh hôm nay
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) as attended 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
    ''', (today,))
    attended_today = cursor.fetchone()['attended']
    
    # Thống kê theo giới tính
    cursor.execute('''
        SELECT gender, COUNT(*) as count 
        FROM students 
        GROUP BY gender
    ''')
    gender_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_students': total_students,
        'attended_today': attended_today,
        'attendance_rate': round(attended_today / total_students * 100, 1) if total_students > 0 else 0,
        'gender_stats': {row['gender']: row['count'] for row in gender_stats}
    }


# Khởi tạo database khi import module
init_db()
