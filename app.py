
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, func  # Needed for OR queries and aggregation
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
import re

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials as firebase_credentials
except ImportError:
    firebase_admin = None
    firebase_auth = None
    firebase_credentials = None

app = Flask(__name__)

# ==================== CONFIGURATION ====================
app.secret_key = 'dev_secret_key_change_in_production'

# SQLite Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///civic_assist.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Firebase Web SDK Configuration (used by templates)
DEFAULT_FIREBASE_WEB_CONFIG = {
    "apiKey": "AIzaSyDe3fQO4t5Wnp4URpQOv5fsfDoKOuxrO4I",
    "authDomain": "civicassist-39685.firebaseapp.com",
    "projectId": "civicassist-39685",
    "storageBucket": "civicassist-39685.firebasestorage.app",
    "messagingSenderId": "999705978680",
    "appId": "1:999705978680:web:2422d5c56de64d1b1483cd",
    "measurementId": "G-3P3ZTB9DQ2",
}
firebase_web_config_json = os.getenv('FIREBASE_WEB_CONFIG_JSON')
if firebase_web_config_json:
    try:
        app.config['FIREBASE_WEB_CONFIG'] = json.loads(firebase_web_config_json)
    except json.JSONDecodeError:
        app.config['FIREBASE_WEB_CONFIG'] = DEFAULT_FIREBASE_WEB_CONFIG
else:
    app.config['FIREBASE_WEB_CONFIG'] = DEFAULT_FIREBASE_WEB_CONFIG

# File Uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize DB
db = SQLAlchemy(app)
_firebase_init_error = None

# ==================== DATABASE MODELS ====================

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False) # Should be unique ideally
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    is_admin = db.Column(db.Boolean, default=False)
    complaints = db.relationship('Complaint', backref='author', lazy=True)

class Worker(db.Model):
    __tablename__ = 'workers'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    department = db.Column(db.String(50), nullable=False)
    area = db.Column(db.String(100))
    is_available = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assignments = db.relationship('Complaint', backref='assigned_worker', lazy=True)

class Complaint(db.Model):
    __tablename__ = 'complaints'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    worker_id = db.Column(db.Integer, db.ForeignKey('workers.id'), nullable=True)
    complaint_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    department = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Pending')
    latitude = db.Column(db.String(20), default='N/A')
    longitude = db.Column(db.String(20), default='N/A')
    image_path = db.Column(db.String(255))
    remarks = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_firebase_admin_initialized():
    global _firebase_init_error

    if firebase_admin is None:
        return 'firebase-admin package is not installed.'

    if firebase_admin._apps:
        return None

    if _firebase_init_error:
        return _firebase_init_error

    service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')

    try:
        if service_account_json:
            cred = firebase_credentials.Certificate(json.loads(service_account_json))
        elif service_account_path:
            if not os.path.exists(service_account_path):
                _firebase_init_error = f'Service account file not found: {service_account_path}'
                return _firebase_init_error
            cred = firebase_credentials.Certificate(service_account_path)
        else:
            cred = firebase_credentials.ApplicationDefault()

        firebase_admin.initialize_app(cred)
        _firebase_init_error = None
        return None
    except Exception as exc:
        _firebase_init_error = str(exc)
        return _firebase_init_error

def verify_firebase_token_from_request():
    data = request.get_json(silent=True) or {}
    id_token = data.get('idToken')
    if not id_token:
        return None, 'Missing Firebase ID token.'

    init_error = ensure_firebase_admin_initialized()
    if init_error:
        return None, 'Firebase Admin SDK is not configured on the server.'

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
    except Exception:
        return None, 'Invalid or expired Firebase session. Please sign in again.'

    email = (decoded_token.get('email') or '').strip().lower()
    if not email:
        return None, 'Firebase account email is missing.'

    decoded_token['email'] = email
    return decoded_token, None

def set_citizen_session(user):
    session['user_id'] = user.id
    session['username'] = user.username
    session['is_admin'] = user.is_admin
    session['citizen_logged_in'] = True
    session.pop('admin_logged_in', None)

def sanitize_username(raw_value):
    raw_value = (raw_value or '').strip().lower()
    cleaned = re.sub(r'[^a-z0-9_]+', '_', raw_value)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned[:30] or 'citizen'

def make_unique_username(seed_value):
    base = sanitize_username(seed_value)
    candidate = base
    suffix = 1
    while User.query.filter(func.lower(User.username) == candidate.lower()).first():
        suffix += 1
        suffix_text = f"_{suffix}"
        trimmed_base = base[:max(1, 30 - len(suffix_text))]
        candidate = f"{trimmed_base}{suffix_text}"
    return candidate

# ==================== ROUTES ====================

@app.context_processor
def inject_firebase_config():
    return {'firebase_web_config': app.config.get('FIREBASE_WEB_CONFIG', {})}

@app.route('/')
def index():
    return render_template('index.html')

# --- CITIZEN AUTH ---

@app.route('/citizen/register', methods=['GET', 'POST'])
def citizen_register():
    if request.method == 'POST':
        flash('Registration is now handled by Firebase Authentication. Please use the web form.', 'warning')
        return redirect(url_for('citizen_register'))
    
    return render_template('citizen_register.html')

@app.route('/citizen/login', methods=['GET', 'POST'])
def citizen_login():
    if request.method == 'POST':
        flash('Login is now handled by Firebase Authentication. Please use the web form.', 'warning')
        return redirect(url_for('citizen_login'))
            
    return render_template('citizen_login.html')

@app.route('/citizen/firebase/resolve-email', methods=['POST'])
def citizen_firebase_resolve_email():
    data = request.get_json(silent=True) or {}
    login_input = (data.get('loginInput') or '').strip()
    if not login_input:
        return jsonify({'error': 'Username or email is required.'}), 400

    if '@' in login_input:
        return jsonify({'email': login_input.lower()}), 200

    user = User.query.filter(
        func.lower(User.username) == login_input.lower(),
        User.is_admin == False
    ).first()

    if not user:
        return jsonify({'error': 'No citizen account found for that username.'}), 404

    return jsonify({'email': user.email}), 200

@app.route('/citizen/firebase/register', methods=['POST'])
def citizen_firebase_register():
    decoded_token, token_error = verify_firebase_token_from_request()
    if token_error:
        return jsonify({'error': token_error}), 401

    data = request.get_json(silent=True) or {}
    uid = (decoded_token.get('uid') or '').strip()
    email = decoded_token['email']
    username = (data.get('username') or '').strip()
    name = (data.get('name') or '').strip()
    phone = (data.get('phone') or '').strip()
    address = (data.get('address') or '').strip()

    if not username:
        return jsonify({'error': 'Username is required.'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters.'}), 400
    if not name:
        return jsonify({'error': 'Full name is required.'}), 400

    existing_by_email = User.query.filter(func.lower(User.email) == email).first()
    existing_by_username = User.query.filter(func.lower(User.username) == username.lower()).first()

    if existing_by_email and existing_by_email.is_admin:
        return jsonify({'error': 'This email is reserved for an admin account.'}), 403

    if existing_by_username and (not existing_by_email or existing_by_username.id != existing_by_email.id):
        return jsonify({'error': 'Username already exists. Choose another one.'}), 409

    if existing_by_email:
        user = existing_by_email
        user.username = username
        user.name = name
        user.phone = phone
        user.address = address
        user.password_hash = generate_password_hash(f"firebase::{uid or email}")
    else:
        user = User(
            username=username,
            password_hash=generate_password_hash(f"firebase::{uid or email}"),
            name=name,
            email=email,
            phone=phone,
            address=address,
            is_admin=False
        )
        db.session.add(user)

    db.session.commit()
    set_citizen_session(user)

    return jsonify({
        'success': True,
        'message': f'Welcome, {user.name}!',
        'redirect_url': url_for('citizen_dashboard')
    }), 200

@app.route('/citizen/firebase/login', methods=['POST'])
def citizen_firebase_login():
    decoded_token, token_error = verify_firebase_token_from_request()
    if token_error:
        return jsonify({'error': token_error}), 401

    email = decoded_token['email']
    uid = (decoded_token.get('uid') or '').strip()
    display_name = (decoded_token.get('name') or '').strip()

    user = User.query.filter(
        func.lower(User.email) == email,
        User.is_admin == False
    ).first()

    admin_user = User.query.filter(
        func.lower(User.email) == email,
        User.is_admin == True
    ).first()
    if admin_user:
        return jsonify({'error': 'This account is restricted to the admin portal.'}), 403

    if not user:
        fallback_name = display_name or email.split('@')[0]
        user = User(
            username=make_unique_username(fallback_name),
            password_hash=generate_password_hash(f"firebase::{uid or email}"),
            name=fallback_name,
            email=email,
            phone='',
            address='',
            is_admin=False
        )
        db.session.add(user)
        db.session.commit()

    set_citizen_session(user)

    return jsonify({
        'success': True,
        'message': f'Welcome back, {user.name}!',
        'redirect_url': url_for('citizen_dashboard')
    }), 200

# --- CITIZEN DASHBOARD ---

@app.route('/citizen/dashboard')
def citizen_dashboard():
    if 'user_id' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('citizen_login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('citizen_login'))

    # Get complaints for this user only
    my_complaints = Complaint.query.filter_by(user_id=user.id).order_by(Complaint.created_at.desc()).all()
    
    return render_template('citizen_dashboard.html', citizen=user, complaints=my_complaints)

@app.route('/citizen/register_complaint', methods=['POST'])
def register_complaint():
    if 'user_id' not in session:
        return redirect(url_for('citizen_login'))
        
    image_path = None
    if 'complaint_image' in request.files:
        file = request.files['complaint_image']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:  # secure_filename can return empty string for special-char filenames
                filename = f"image_{int(datetime.now().timestamp())}.jpg"
            unique_name = f"{session['user_id']}_{int(datetime.now().timestamp())}_{filename}"
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(full_path)
            image_path = f"uploads/{unique_name}"

    new_complaint = Complaint(
        user_id=session['user_id'],
        complaint_type=request.form.get('complaint_type'),
        description=request.form.get('description'),
        latitude=request.form.get('latitude') or 'N/A',
        longitude=request.form.get('longitude') or 'N/A',
        department='Unassigned',
        image_path=image_path
    )
    
    db.session.add(new_complaint)
    db.session.commit()
    
    flash('Complaint registered successfully!', 'success')
    return redirect(url_for('citizen_dashboard'))

@app.route('/citizen/logout')
def citizen_logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/citizen/profile/update', methods=['POST'])
def citizen_update_profile():
    if 'user_id' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('citizen_login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('citizen_login'))
    
    user.name = request.form.get('name', user.name)
    submitted_email = (request.form.get('email') or '').strip().lower()
    if submitted_email and submitted_email != (user.email or '').lower():
        flash('Email is managed by Firebase Authentication and cannot be changed here.', 'info')

    user.phone = request.form.get('phone', user.phone)
    user.address = request.form.get('address', user.address)
    
    new_password = request.form.get('new_password')
    if new_password:
        flash('Password updates are managed by Firebase Authentication.', 'info')
    
    db.session.commit()
    flash('Profile updated successfully.', 'success')
    return redirect(url_for('citizen_dashboard') + '#profile')

# --- ADMIN ROUTES ---

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        login_input = request.form.get('username')
        password = request.form.get('password')
        
        # FIXED: Admin login now also supports Username OR Email
        user = User.query.filter(
            or_(User.username == login_input, User.email == login_input)
        ).filter_by(is_admin=True).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['is_admin'] = True
            session['admin_logged_in'] = True
            flash('Admin access granted.', 'success')
            return redirect(url_for('admin_dashboard'))
        
        flash('Invalid admin credentials.', 'danger')
        
    return render_template('admin_login.html')

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')

        # Check if username or email already exists
        existing_user = User.query.filter(
            or_(User.username == username, User.email == email)
        ).first()

        if existing_user:
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('admin_register'))

        hashed_pw = generate_password_hash(password)
        new_admin = User(
            username=username,
            password_hash=hashed_pw,
            name=name,
            email=email,
            phone=request.form.get('phone', ''),
            address=request.form.get('department', ''),
            is_admin=True
        )

        db.session.add(new_admin)
        db.session.commit()

        flash('Admin registration successful! Please login.', 'success')
        return redirect(url_for('admin_login'))

    return render_template('admin_registration.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    admin_user = User.query.get(session.get('user_id'))
    all_complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    
    # Compute department workload stats
    dept_counts = {}
    for c in all_complaints:
        dept = c.department or 'Unassigned'
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    total = len(all_complaints) if all_complaints else 1  # avoid division by zero
    dept_stats = [{'name': dept, 'count': count, 'pct': int(count / total * 100)} for dept, count in dept_counts.items()]
    dept_stats.sort(key=lambda x: x['count'], reverse=True)
    
    all_workers = Worker.query.filter_by(is_available=True).order_by(Worker.name).all()
    
    return render_template('admin_dashboard.html', complaints=all_complaints, admin=admin_user, dept_stats=dept_stats, workers=all_workers)

@app.route('/admin/update_complaint/<int:complaint_id>', methods=['POST'])
def update_complaint(complaint_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
        
    complaint = Complaint.query.get_or_404(complaint_id)
    complaint.status = request.form.get('status')
    complaint.department = request.form.get('department')
    complaint.remarks = request.form.get('remarks')
    
    worker_id = request.form.get('worker_id')
    if worker_id:
        complaint.worker_id = int(worker_id)
    
    db.session.commit()
    flash('Complaint updated successfully.', 'success')
    
    # Redirect back to the page the admin came from
    next_page = request.form.get('next', url_for('admin_dashboard'))
    return redirect(next_page)

# --- ALL COMPLAINTS PAGE ---

@app.route('/admin/complaints')
def admin_complaints():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    admin_user = User.query.get(session.get('user_id'))
    all_complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    all_workers = Worker.query.filter_by(is_available=True).order_by(Worker.name).all()
    return render_template('admin_complaints.html', complaints=all_complaints, admin=admin_user, workers=all_workers)

# --- DEPARTMENT REPORTS ---

@app.route('/admin/reports')
def admin_reports():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    admin_user = User.query.get(session.get('user_id'))
    all_complaints = Complaint.query.all()
    
    # Build department-level report data
    dept_data = {}
    for c in all_complaints:
        dept = c.department or 'Unassigned'
        if dept not in dept_data:
            dept_data[dept] = {'total': 0, 'pending': 0, 'in_progress': 0, 'resolved': 0, 'rejected': 0}
        dept_data[dept]['total'] += 1
        if c.status == 'Pending':
            dept_data[dept]['pending'] += 1
        elif c.status == 'In Progress':
            dept_data[dept]['in_progress'] += 1
        elif c.status == 'Resolved':
            dept_data[dept]['resolved'] += 1
        elif c.status == 'Rejected':
            dept_data[dept]['rejected'] += 1
    
    return render_template('admin_reports.html', dept_data=dept_data, total_complaints=len(all_complaints), admin=admin_user)

# --- USER MANAGEMENT ---

@app.route('/admin/users')
def admin_users():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    admin_user = User.query.get(session.get('user_id'))
    citizens = User.query.filter_by(is_admin=False).order_by(User.name).all()
    admins = User.query.filter_by(is_admin=True).order_by(User.name).all()
    return render_template('admin_users.html', citizens=citizens, admins=admins, admin=admin_user)

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash('Cannot delete admin accounts from here.', 'warning')
        return redirect(url_for('admin_users'))
    
    # Delete user's complaints first
    Complaint.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{user.name}" deleted.', 'success')
    return redirect(url_for('admin_users'))

# --- WORKER MANAGEMENT ---

@app.route('/admin/workers', methods=['GET', 'POST'])
def admin_workers():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        new_worker = Worker(
            name=request.form.get('name'),
            phone=request.form.get('phone'),
            department=request.form.get('department'),
            area=request.form.get('area'),
            is_available=True
        )
        db.session.add(new_worker)
        db.session.commit()
        flash(f'Worker "{new_worker.name}" added successfully.', 'success')
        return redirect(url_for('admin_workers'))
    
    admin_user = User.query.get(session.get('user_id'))
    workers = Worker.query.order_by(Worker.created_at.desc()).all()
    return render_template('admin_workers.html', workers=workers, admin=admin_user)

@app.route('/admin/workers/<int:worker_id>/delete', methods=['POST'])
def delete_worker(worker_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    
    worker = Worker.query.get_or_404(worker_id)
    # Unassign worker from any complaints
    Complaint.query.filter_by(worker_id=worker.id).update({'worker_id': None})
    db.session.delete(worker)
    db.session.commit()
    flash(f'Worker "{worker.name}" deleted.', 'success')
    return redirect(url_for('admin_workers'))

@app.route('/admin/workers/<int:worker_id>/toggle', methods=['POST'])
def toggle_worker(worker_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    
    worker = Worker.query.get_or_404(worker_id)
    worker.is_available = not worker.is_available
    db.session.commit()
    status_text = 'available' if worker.is_available else 'unavailable'
    flash(f'Worker "{worker.name}" marked as {status_text}.', 'success')
    return redirect(url_for('admin_workers'))

# --- ADMIN SETTINGS ---

@app.route('/admin/settings', methods=['GET', 'POST'])
def admin_settings():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))
    
    admin_user = User.query.get(session.get('user_id'))
    
    if request.method == 'POST':
        admin_user.name = request.form.get('name', admin_user.name)
        admin_user.email = request.form.get('email', admin_user.email)
        admin_user.phone = request.form.get('phone', admin_user.phone)
        
        new_password = request.form.get('new_password')
        if new_password and len(new_password) >= 6:
            admin_user.password_hash = generate_password_hash(new_password)
            flash('Password updated.', 'success')
        
        db.session.commit()
        flash('Settings saved.', 'success')
        return redirect(url_for('admin_settings'))
    
    return render_template('admin_settings.html', admin=admin_user)

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Admin logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/complaint/<int:complaint_id>')
def view_complaint(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    return render_template('complaint_detail.html', complaint=complaint)

# ==================== INITIALIZATION ====================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        if not User.query.filter_by(username='admin').first():
            print("Creating default admin account (admin / admin123)...")
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                name='System Admin',
                email='admin@civic.local',
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            
    app.run(debug=True)
