import sqlite3
import os
import json
from datetime import datetime, time, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# Mui gio Viet Nam (UTC+7)
VIETNAM_TZ = timezone(timedelta(hours=7))

# Gio bat dau lam viec (9:30 sang) - diem danh sau gio nay se bi tinh la muon
WORK_START_TIME = time(9, 30, 0)  # 9h30 sáng

def get_vietnam_now():
    """Lay thoi gian hien tai theo gio Viet Nam"""
    return datetime.now(VIETNAM_TZ)


def get_connection():
    """Tao ket noi den database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khoi tao database va cac bang"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Bang nhan vien
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            gender TEXT,
            face_encoding TEXT,
            face_image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Bang diem danh
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            gender_detected TEXT,
            emotion_detected TEXT,
            confidence REAL,
            status TEXT DEFAULT 'present',
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[OK] Database initialized successfully!")


# ===============================
# Employee CRUD Operations
# ===============================

def add_employee(employee_code, name, department, gender, face_encoding, face_image):
    """Them nhan vien moi"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # face_encoding da la list Python thuan tuy tu face_utils
        if face_encoding is not None and len(face_encoding) > 0:
            encoding_json = json.dumps(face_encoding)
        else:
            encoding_json = None
            
        cursor.execute('''
            INSERT INTO employees (employee_code, name, department, gender, face_encoding, face_image)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (employee_code, name, department, gender, encoding_json, face_image))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        print(f"add_employee error: {e}")
        return None
    finally:
        conn.close()


def get_all_employees(search=None):
    """Lay danh sach tat ca nhan vien voi tuy chon tim kiem"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if search:
        search_term = f'%{search}%'
        cursor.execute('''
            SELECT * FROM employees 
            WHERE employee_code LIKE ? 
               OR name LIKE ? 
               OR department LIKE ?
            ORDER BY created_at DESC
        ''', (search_term, search_term, search_term))
    else:
        cursor.execute('SELECT * FROM employees ORDER BY created_at DESC')
    
    employees = cursor.fetchall()
    conn.close()
    return employees


def get_employee_by_id(employee_id):
    """Lay thong tin nhan vien theo ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM employees WHERE id = ?', (employee_id,))
    employee = cursor.fetchone()
    conn.close()
    return employee


def get_employee_by_code(employee_code):
    """Lay thong tin nhan vien theo ma nhan vien"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM employees WHERE employee_code = ?', (employee_code,))
    employee = cursor.fetchone()
    conn.close()
    return employee


def delete_employee(employee_id):
    """Xoa nhan vien"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM employees WHERE id = ?', (employee_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def get_all_face_encodings():
    """Lay tat ca face encodings tu database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, employee_code, name, department, gender, face_encoding FROM employees WHERE face_encoding IS NOT NULL')
    employees = cursor.fetchall()
    conn.close()
    
    result = []
    for emp in employees:
        if emp['face_encoding']:
            encoding = json.loads(emp['face_encoding'])
            result.append({
                'id': emp['id'],
                'employee_code': emp['employee_code'],
                'name': emp['name'],
                'department': emp['department'],
                'gender': emp['gender'],
                'encoding': encoding
            })
    return result


# ===============================
# Attendance CRUD Operations
# ===============================

def add_attendance(employee_id, gender_detected, emotion_detected, confidence, status=None):
    """Ghi diem danh - tu dong xac dinh dung gio hoac muon (theo gio Viet Nam)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Thoi gian hien tai theo gio Viet Nam
    vietnam_now = get_vietnam_now()
    
    # Tu dong xac dinh trang thai dua vao thoi gian hien tai
    if status is None:
        current_time = vietnam_now.time()
        if current_time <= WORK_START_TIME:
            status = 'on_time'  # Dung gio
        else:
            status = 'late'  # Muon
    
    # Luu thoi gian check-in theo gio Viet Nam
    check_in_time = vietnam_now.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO attendance (employee_id, check_in_time, gender_detected, emotion_detected, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (employee_id, check_in_time, gender_detected, emotion_detected, confidence, status))
    conn.commit()
    attendance_id = cursor.lastrowid
    conn.close()
    return attendance_id


def get_attendance_today():
    """Lay danh sach diem danh hom nay (theo gio Viet Nam)"""
    conn = get_connection()
    cursor = conn.cursor()
    today = get_vietnam_now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT a.*, e.employee_code, e.name, e.department, e.gender as registered_gender
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        WHERE DATE(a.check_in_time) = ?
        ORDER BY a.check_in_time DESC
    ''', (today,))
    records = cursor.fetchall()
    conn.close()
    return records


def get_attendance_history(days=7, search=None, date=None):
    """Lay lich su diem danh voi tuy chon tim kiem va loc theo ngay"""
    conn = get_connection()
    cursor = conn.cursor()
    
    base_query = '''
        SELECT a.*, e.employee_code, e.name, e.department, e.gender as registered_gender
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
    '''
    
    conditions = []
    params = []
    
    # Loc theo ngay cu the hoac N ngay gan nhat
    if date:
        conditions.append("DATE(a.check_in_time) = ?")
        params.append(date)
    else:
        conditions.append("a.check_in_time >= datetime('now', ?)")
        params.append(f'-{days} days')
    
    # Tim kiem theo ma NV, ten, phong ban
    if search:
        search_term = f'%{search}%'
        conditions.append("(e.employee_code LIKE ? OR e.name LIKE ? OR e.department LIKE ?)")
        params.extend([search_term, search_term, search_term])
    
    query = base_query + ' WHERE ' + ' AND '.join(conditions) + ' ORDER BY a.check_in_time DESC'
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()
    return records


def get_attendance_dates():
    """Lay danh sach cac ngay co diem danh"""
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
    """Thong ke diem danh theo ngay"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if not date:
        date = get_vietnam_now().strftime('%Y-%m-%d')
    
    # Tong so nhan vien
    cursor.execute('SELECT COUNT(*) as total FROM employees')
    total_employees = cursor.fetchone()['total']
    
    # So nhan vien diem danh trong ngay
    cursor.execute('''
        SELECT COUNT(DISTINCT employee_id) as attended 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
    ''', (date,))
    attended = cursor.fetchone()['attended']
    
    # Thong ke theo trang thai
    cursor.execute('''
        SELECT status, COUNT(*) as count 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
        GROUP BY status
    ''', (date,))
    status_stats = {row['status']: row['count'] for row in cursor.fetchall()}
    
    # Thong ke theo phong ban
    cursor.execute('''
        SELECT e.department, COUNT(DISTINCT a.employee_id) as count
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        WHERE DATE(a.check_in_time) = ?
        GROUP BY e.department
        ORDER BY count DESC
    ''', (date,))
    dept_stats = {row['department']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'date': date,
        'total_employees': total_employees,
        'attended': attended,
        'absent': total_employees - attended,
        'attendance_rate': round(attended / total_employees * 100, 1) if total_employees > 0 else 0,
        'status_stats': status_stats,
        'dept_stats': dept_stats
    }


def check_already_attended_today(employee_id):
    """Kiem tra nhan vien da diem danh hom nay chua (theo gio Viet Nam)"""
    conn = get_connection()
    cursor = conn.cursor()
    today = get_vietnam_now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COUNT(*) as count FROM attendance 
        WHERE employee_id = ? AND DATE(check_in_time) = ?
    ''', (employee_id, today))
    result = cursor.fetchone()
    conn.close()
    return result['count'] > 0


def get_attendance_stats():
    """Thong ke diem danh (theo gio Viet Nam)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tong so nhan vien
    cursor.execute('SELECT COUNT(*) as total FROM employees')
    total_employees = cursor.fetchone()['total']
    
    # So nhan vien da diem danh hom nay
    today = get_vietnam_now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COUNT(DISTINCT employee_id) as attended 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
    ''', (today,))
    attended_today = cursor.fetchone()['attended']
    
    # Thong ke dung gio va muon hom nay
    cursor.execute('''
        SELECT status, COUNT(*) as count 
        FROM attendance 
        WHERE DATE(check_in_time) = ?
        GROUP BY status
    ''', (today,))
    status_stats = {row['status']: row['count'] for row in cursor.fetchall()}
    on_time_count = status_stats.get('on_time', 0)
    late_count = status_stats.get('late', 0)
    
    # Thong ke theo gioi tinh
    cursor.execute('''
        SELECT gender, COUNT(*) as count 
        FROM employees 
        GROUP BY gender
    ''')
    gender_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_employees': total_employees,
        'attended_today': attended_today,
        'on_time_count': on_time_count,
        'late_count': late_count,
        'attendance_rate': round(attended_today / total_employees * 100, 1) if total_employees > 0 else 0,
        'gender_stats': {row['gender']: row['count'] for row in gender_stats}
    }


# Khoi tao database khi import module
init_db()
