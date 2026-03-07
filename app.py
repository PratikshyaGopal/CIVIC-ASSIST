
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, func  # Needed for OR queries and aggregation
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json

app = Flask(__name__)

# ==================== CONFIGURATION ====================
app.secret_key = 'dev_secret_key_change_in_production'

# SQLite Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///civic_assist.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File Uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize DB
db = SQLAlchemy(app)

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

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

# --- CITIZEN AUTH ---

@app.route('/citizen/register', methods=['GET', 'POST'])
def citizen_register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username OR email already exists
        existing_user = User.query.filter(
            or_(User.username == username, User.email == email)
        ).first()
        
        if existing_user:
            flash('Username or Email already exists! Please try logging in.', 'danger')
            return redirect(url_for('citizen_register'))
            
        # Create new user
        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            password_hash=hashed_pw,
            name=request.form.get('name'),
            email=email,
            phone=request.form.get('phone'),
            address=request.form.get('address'),
            is_admin=False
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('citizen_login'))
    
    return render_template('citizen_register.html')

@app.route('/citizen/login', methods=['GET', 'POST'])
def citizen_login():
    if request.method == 'POST':
        # Input can be username OR email (field name in HTML is 'username')
        login_input = request.form.get('username')
        password = request.form.get('password')
        
        # FIXED: Check against both username and email columns
        user = User.query.filter(
            or_(User.username == login_input, User.email == login_input)
        ).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            session['citizen_logged_in'] = True
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('citizen_dashboard'))
        else:
            flash('Invalid credentials. Please check your username/email and password.', 'danger')
            
    return render_template('citizen_login.html')

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
    user.email = request.form.get('email', user.email)
    user.phone = request.form.get('phone', user.phone)
    user.address = request.form.get('address', user.address)
    
    new_password = request.form.get('new_password')
    if new_password and len(new_password) >= 6:
        user.password_hash = generate_password_hash(new_password)
        flash('Password updated.', 'success')
    
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