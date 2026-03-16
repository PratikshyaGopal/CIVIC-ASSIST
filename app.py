
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from urllib.parse import urlparse
import os
import json
import re
import logging
import uuid

try:
    import jwt as pyjwt
    _pyjwt_available = True
except ImportError:
    pyjwt = None
    _pyjwt_available = False

try:
    import requests as http_requests
except ImportError:
    http_requests = None

# JWKS cache: { 'keys': {...}, 'fetched_at': datetime }
_jwks_cache = {}

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials as firebase_credentials, db as firebase_db
except ImportError:
    firebase_admin = None
    firebase_auth = None
    firebase_credentials = None
    firebase_db = None

app = Flask(__name__)

# ==================== CONFIGURATION ====================
app.secret_key = os.getenv(
    'FLASK_SECRET_KEY',
    'civic-assist-local-dev-only-change-me-in-production-32chars'
)

app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
# On Vercel (HTTPS) enforce Secure flag so session cookie is never sent over plain HTTP.
if os.getenv('VERCEL'):
    app.config['SESSION_COOKIE_SECURE'] = True

# Firebase Web SDK Configuration (used by templates)
DEFAULT_FIREBASE_WEB_CONFIG = {
    "apiKey": "AIzaSyDe3fQO4t5Wnp4URpQOv5fsfDoKOuxrO4I",
    "authDomain": "civicassist-39685.firebaseapp.com",
    "projectId": "civicassist-39685",
    "storageBucket": "civicassist-39685.firebasestorage.app",
    "messagingSenderId": "999705978680",
    "appId": "1:999705978680:web:2422d5c56de64d1b1483cd",
    "measurementId": "G-3P3ZTB9DQ2",
    "databaseURL": "https://civicassist-39685-default-rtdb.asia-southeast1.firebasedatabase.app/",
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
    try:
        os.makedirs(UPLOAD_FOLDER)
    except OSError:
        pass  # Read-only filesystem on Vercel — uploads disabled but app still runs

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Firebase Realtime Database URL
FIREBASE_DB_URL = "https://civicassist-39685-default-rtdb.asia-southeast1.firebasedatabase.app/"

# ==================== FIREBASE ADMIN INIT ====================
_firebase_init_error = None
_rtdb_initialized = False


def ensure_firebase_admin_initialized():
    """Initialize Firebase Admin SDK (for RTDB and optionally Auth). Returns error string or None."""
    global _firebase_init_error, _rtdb_initialized

    if firebase_admin is None:
        return 'firebase-admin package is not installed.'

    if firebase_admin._apps:
        _rtdb_initialized = True
        return None

    if _firebase_init_error:
        return _firebase_init_error

    service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')

    try:
        if service_account_json:
            cred = firebase_credentials.Certificate(json.loads(service_account_json))
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
        elif service_account_path:
            if not os.path.exists(service_account_path):
                _firebase_init_error = f'Service account file not found: {service_account_path}'
                return _firebase_init_error
            cred = firebase_credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
        else:
            # No service account — will use RTDB REST API directly (no Admin SDK for RTDB)
            _firebase_init_error = 'No service account configured; using JWKS + REST fallback.'
            return _firebase_init_error

        _firebase_init_error = None
        _rtdb_initialized = True
        return None
    except Exception as exc:
        _firebase_init_error = str(exc)
        app.logger.warning('Firebase Admin SDK init failed: %s', exc)
        return _firebase_init_error


# ==================== FIREBASE RTDB REST HELPERS ====================
# We use the Firebase REST API so the app works even without a service account.
# All data is stored without auth rules check (public read+write) or you should
# configure RTDB rules on the Firebase console to secure it.

def _rtdb_url(path):
    """Build the full REST URL for a given RTDB path."""
    base = FIREBASE_DB_URL.rstrip('/')
    return f"{base}/{path.lstrip('/')}.json"


def _rtdb_get(path):
    """GET a node from RTDB. Returns parsed JSON or None."""
    try:
        # Try Admin SDK first
        if firebase_admin and firebase_admin._apps and firebase_db:
            ref = firebase_db.reference(path)
            return ref.get()
        # Fallback: REST API
        if http_requests is None:
            app.logger.error('RTDB GET %s failed: requests package is not installed.', path)
            return None
        resp = http_requests.get(_rtdb_url(path), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        app.logger.error('RTDB GET %s failed: %s', path, exc)
        return None


def _rtdb_set(path, data):
    """PUT (overwrite) a node in RTDB. Returns True on success."""
    try:
        if firebase_admin and firebase_admin._apps and firebase_db:
            ref = firebase_db.reference(path)
            ref.set(data)
            return True
        if http_requests is None:
            app.logger.error('RTDB SET %s failed: requests package is not installed.', path)
            return False
        resp = http_requests.put(_rtdb_url(path), json=data, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        app.logger.error('RTDB SET %s failed: %s', path, exc)
        return False


def _rtdb_update(path, data):
    """PATCH (partial update) a node in RTDB. Returns True on success."""
    try:
        if firebase_admin and firebase_admin._apps and firebase_db:
            ref = firebase_db.reference(path)
            ref.update(data)
            return True
        if http_requests is None:
            app.logger.error('RTDB UPDATE %s failed: requests package is not installed.', path)
            return False
        resp = http_requests.patch(_rtdb_url(path), json=data, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        app.logger.error('RTDB UPDATE %s failed: %s', path, exc)
        return False


def _rtdb_push(path, data):
    """POST (push with auto-ID) a node to RTDB. Returns the new key, or None."""
    try:
        if firebase_admin and firebase_admin._apps and firebase_db:
            ref = firebase_db.reference(path)
            new_ref = ref.push(data)
            return new_ref.key
        if http_requests is None:
            app.logger.error('RTDB PUSH %s failed: requests package is not installed.', path)
            return None
        resp = http_requests.post(_rtdb_url(path), json=data, timeout=10)
        resp.raise_for_status()
        return resp.json().get('name')
    except Exception as exc:
        app.logger.error('RTDB PUSH %s failed: %s', path, exc)
        return None


def _rtdb_delete(path):
    """DELETE a node from RTDB. Returns True on success."""
    try:
        if firebase_admin and firebase_admin._apps and firebase_db:
            ref = firebase_db.reference(path)
            ref.delete()
            return True
        if http_requests is None:
            app.logger.error('RTDB DELETE %s failed: requests package is not installed.', path)
            return False
        resp = http_requests.delete(_rtdb_url(path), timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        app.logger.error('RTDB DELETE %s failed: %s', path, exc)
        return False


# ==================== DATA ACCESS LAYER ====================

# ---- USERS ----

def _all_users():
    """Return dict of {user_id: user_dict} or {} on error."""
    data = _rtdb_get('users')
    return data if isinstance(data, dict) else {}


def get_user_by_id(user_id):
    if not user_id:
        return None
    u = _rtdb_get(f'users/{user_id}')
    if isinstance(u, dict):
        u['id'] = user_id
        return u
    return None


def get_user_by_email(email):
    """Return first user whose email matches (case-insensitive)."""
    email = (email or '').strip().lower()
    for uid, u in _all_users().items():
        if (u.get('email') or '').strip().lower() == email:
            u['id'] = uid
            return u
    return None


def get_user_by_username(username):
    username = (username or '').strip().lower()
    for uid, u in _all_users().items():
        if (u.get('username') or '').strip().lower() == username:
            u['id'] = uid
            return u
    return None


def save_user(user_id, data):
    """Overwrite user node. Returns True on success."""
    return _rtdb_set(f'users/{user_id}', data)


def update_user(user_id, partial):
    return _rtdb_update(f'users/{user_id}', partial)


def delete_user_data(user_id):
    return _rtdb_delete(f'users/{user_id}')


def create_user(user_dict):
    """Generate a unique user ID and push user data. Returns user_id."""
    user_id = str(uuid.uuid4()).replace('-', '')[:20]
    _rtdb_set(f'users/{user_id}', user_dict)
    return user_id


def make_unique_username(seed_value):
    base = _sanitize_username(seed_value)
    candidate = base
    suffix = 1
    all_usernames = {(u.get('username') or '').lower() for u in _all_users().values()}
    while candidate.lower() in all_usernames:
        suffix += 1
        suffix_text = f"_{suffix}"
        trimmed_base = base[:max(1, 30 - len(suffix_text))]
        candidate = f"{trimmed_base}{suffix_text}"
    return candidate


def _sanitize_username(raw_value):
    raw_value = (raw_value or '').strip().lower()
    cleaned = re.sub(r'[^a-z0-9_]+', '_', raw_value)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned[:30] or 'citizen'


# ---- WORKERS ----

def _all_workers():
    data = _rtdb_get('workers')
    return data if isinstance(data, dict) else {}


def get_all_workers(available_only=False):
    workers = []
    for wid, w in _all_workers().items():
        if not isinstance(w, dict):
            continue
        w['id'] = wid
        if available_only and not w.get('is_available', True):
            continue
        workers.append(w)
    # Sort by name
    workers.sort(key=lambda x: x.get('name', ''))
    return workers


def get_worker_by_id(worker_id):
    w = _rtdb_get(f'workers/{worker_id}')
    if isinstance(w, dict):
        w['id'] = worker_id
        return w
    return None


def add_worker(worker_dict):
    """Push new worker, return new key."""
    return _rtdb_push('workers', worker_dict)


def update_worker(worker_id, partial):
    return _rtdb_update(f'workers/{worker_id}', partial)


def delete_worker(worker_id):
    return _rtdb_delete(f'workers/{worker_id}')


# ---- COMPLAINTS ----

def _all_complaints():
    data = _rtdb_get('complaints')
    return data if isinstance(data, dict) else {}


def get_all_complaints():
    complaints = []
    for cid, c in _all_complaints().items():
        if not isinstance(c, dict):
            continue
        c['id'] = cid
        complaints.append(c)
    complaints.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return complaints


def get_complaints_by_user(user_id):
    complaints = []
    for cid, c in _all_complaints().items():
        if not isinstance(c, dict):
            continue
        if str(c.get('user_id', '')) == str(user_id):
            c['id'] = cid
            complaints.append(c)
    complaints.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return complaints


def get_complaint_by_id(complaint_id):
    c = _rtdb_get(f'complaints/{complaint_id}')
    if isinstance(c, dict):
        c['id'] = complaint_id
        return c
    return None


def add_complaint(complaint_dict):
    return _rtdb_push('complaints', complaint_dict)


def update_complaint_data(complaint_id, partial):
    return _rtdb_update(f'complaints/{complaint_id}', partial)


def delete_complaint(complaint_id):
    return _rtdb_delete(f'complaints/{complaint_id}')


def delete_complaints_by_user(user_id):
    for cid, c in _all_complaints().items():
        if isinstance(c, dict) and str(c.get('user_id', '')) == str(user_id):
            _rtdb_delete(f'complaints/{cid}')


# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# JWKS-based Firebase token verification (works without a service account)
# ---------------------------------------------------------------------------
_FIREBASE_JWKS_URL = (
    'https://www.googleapis.com/service_accounts/v1/jwk/'
    'securetoken@system.gserviceaccount.com'
)
# NOTE: Do NOT read projectId at module load time — config may not be fully populated yet.
# We resolve it lazily inside _verify_firebase_id_token_jwks().


def _get_firebase_jwks():
    """Return cached JWKS public keys, refreshing at most once every 60 min."""
    global _jwks_cache
    now = datetime.utcnow()
    cached_at = _jwks_cache.get('fetched_at')
    if cached_at and (now - cached_at).total_seconds() < 3600:
        return _jwks_cache.get('keys', {})
    try:
        resp = http_requests.get(_FIREBASE_JWKS_URL, timeout=5)
        resp.raise_for_status()
        jwks = resp.json()
        keys = {k['kid']: pyjwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
                for k in jwks.get('keys', [])}
        _jwks_cache = {'keys': keys, 'fetched_at': now}
        return keys
    except Exception as exc:
        app.logger.warning('Failed to fetch Firebase JWKS: %s', exc)
        return _jwks_cache.get('keys', {})


def _verify_firebase_id_token_jwks(id_token):
    """Verify a Firebase ID token using Google's public JWKS (no service account needed)."""
    if not _pyjwt_available:
        raise RuntimeError('PyJWT is not installed. Run: pip install PyJWT cryptography')

    try:
        header = pyjwt.get_unverified_header(id_token)
    except pyjwt.exceptions.DecodeError as exc:
        raise ValueError(f'Malformed ID token: {exc}') from exc

    kid = header.get('kid')
    keys = _get_firebase_jwks()
    public_key = keys.get(kid)
    if not public_key:
        _jwks_cache.clear()
        keys = _get_firebase_jwks()
        public_key = keys.get(kid)
    if not public_key:
        raise ValueError('Firebase public key not found for kid: ' + str(kid))

    # Resolve project ID lazily so it always reads the live config value.
    project_id = (
        app.config.get('FIREBASE_WEB_CONFIG', {}).get('projectId', '')
        or os.getenv('FIREBASE_PROJECT_ID', '')
    )
    payload = pyjwt.decode(
        id_token,
        public_key,
        algorithms=['RS256'],
        audience=project_id,
        options={'verify_exp': True},
    )
    return payload


def verify_firebase_id_token(id_token):
    """Verify a Firebase ID token string. Returns (decoded_token, error_string)."""
    if not id_token:
        return None, 'Missing Firebase ID token.'

    # --- Try firebase-admin SDK first (requires service account) ---
    init_error = ensure_firebase_admin_initialized()
    if not init_error and firebase_auth is not None:
        try:
            decoded_token = firebase_auth.verify_id_token(id_token)
            email = (decoded_token.get('email') or '').strip().lower()
            if not email:
                return None, 'Firebase account email is missing.'
            decoded_token['email'] = email
            return decoded_token, None
        except Exception as exc:
            app.logger.warning('firebase-admin verify_id_token failed: %s', exc)

    # --- JWKS fallback ---
    if not _pyjwt_available:
        return None, (
            'Firebase authentication is unavailable. '
            'Install PyJWT & cryptography, or configure a service account.'
        )
    try:
        decoded_token = _verify_firebase_id_token_jwks(id_token)
    except Exception as exc:
        app.logger.warning('Firebase JWKS token verification failed: %s', exc)
        return None, 'Invalid or expired Firebase session. Please sign in again.'

    email = (decoded_token.get('email') or '').strip().lower()
    if not email:
        return None, 'Firebase account email is missing.'

    decoded_token['email'] = email
    return decoded_token, None


def verify_firebase_token_from_request():
    """Read JSON from request ONCE and verify the Firebase ID token."""
    data = request.get_json(silent=True, force=True) or {}
    id_token = data.get('idToken')
    decoded_token, error = verify_firebase_id_token(id_token)
    return decoded_token, error, data


def set_citizen_session(user_id, user):
    session['user_id'] = user_id
    session['username'] = user.get('username', '')
    session['is_admin'] = user.get('is_admin', False)
    session['citizen_logged_in'] = True
    session.pop('admin_logged_in', None)


# ==================== JINJA2 CUSTOM FILTERS ====================

@app.template_filter('format_date')
def format_date_filter(value, fmt='%b %d, %Y'):
    """Convert an ISO 8601 datetime string (from Firebase RTDB) to a formatted string."""
    if not value:
        return ''
    # Try parsing with and without microseconds
    for pattern in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
        try:
            dt = datetime.strptime(str(value)[:26], pattern)
            return dt.strftime(fmt)
        except ValueError:
            continue
    return str(value)  # fallback: return raw value

# ==================== SECURITY HEADERS ====================

@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file-too-large uploads gracefully."""
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'error': 'File is too large. Maximum allowed size is 5 MB.'}), 413
    flash('The uploaded file is too large. Maximum allowed size is 5 MB.', 'danger')
    return redirect(request.referrer or url_for('citizen_dashboard'))


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

    user = get_user_by_username(login_input)
    if not user or user.get('is_admin'):
        return jsonify({'error': 'No citizen account found for that username.'}), 404

    return jsonify({'email': user['email']}), 200


@app.route('/citizen/firebase/register', methods=['POST'])
def citizen_firebase_register():
    decoded_token, token_error, data = verify_firebase_token_from_request()
    if token_error:
        return jsonify({'error': token_error}), 401

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

    existing_by_email = get_user_by_email(email)
    existing_by_username = get_user_by_username(username)

    if existing_by_email and existing_by_email.get('is_admin'):
        return jsonify({'error': 'This email is reserved for an admin account.'}), 403

    if existing_by_username and (
        not existing_by_email or existing_by_username['id'] != existing_by_email['id']
    ):
        return jsonify({'error': 'Username already exists. Choose another one.'}), 409

    if existing_by_email:
        user_id = existing_by_email['id']
        update_user(user_id, {
            'username': username,
            'name': name,
            'phone': phone,
            'address': address,
            'password_hash': generate_password_hash(f"firebase::{uid or email}"),
        })
        user = get_user_by_id(user_id)
    else:
        user_id = uid if uid else str(uuid.uuid4()).replace('-', '')[:20]
        user_dict = {
            'username': username,
            'password_hash': generate_password_hash(f"firebase::{uid or email}"),
            'name': name,
            'email': email,
            'phone': phone,
            'address': address,
            'is_admin': False,
        }
        save_user(user_id, user_dict)
        user = user_dict

    set_citizen_session(user_id, user)

    return jsonify({
        'success': True,
        'message': f'Welcome, {user["name"]}!',
        'redirect_url': url_for('citizen_dashboard')
    }), 200


@app.route('/citizen/firebase/login', methods=['POST'])
def citizen_firebase_login():
    decoded_token, token_error, _data = verify_firebase_token_from_request()
    if token_error:
        return jsonify({'error': token_error}), 401

    email = decoded_token['email']
    uid = (decoded_token.get('uid') or '').strip()
    display_name = (decoded_token.get('name') or '').strip()

    # Check if this email belongs to an admin
    admin_user = get_user_by_email(email)
    if admin_user and admin_user.get('is_admin'):
        return jsonify({'error': 'This account is restricted to the admin portal.'}), 403

    user = get_user_by_email(email)

    if not user:
        fallback_name = display_name or email.split('@')[0]
        user_id = uid if uid else str(uuid.uuid4()).replace('-', '')[:20]
        user_dict = {
            'username': make_unique_username(fallback_name),
            'password_hash': generate_password_hash(f"firebase::{uid or email}"),
            'name': fallback_name,
            'email': email,
            'phone': '',
            'address': '',
            'is_admin': False,
        }
        save_user(user_id, user_dict)
        user = user_dict
        user['id'] = user_id
    
    user_id = user['id']
    set_citizen_session(user_id, user)

    return jsonify({
        'success': True,
        'message': f'Welcome back, {user["name"]}!',
        'redirect_url': url_for('citizen_dashboard')
    }), 200


# --- CITIZEN DASHBOARD ---

@app.route('/citizen/dashboard')
def citizen_dashboard():
    if 'user_id' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('citizen_login'))

    user = get_user_by_id(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('citizen_login'))

    my_complaints = get_complaints_by_user(session['user_id'])

    # Enrich complaints with worker name for display
    for c in my_complaints:
        w_id = c.get('worker_id')
        if w_id:
            w = get_worker_by_id(w_id)
            c['worker_name'] = w.get('name', '') if w else ''
        else:
            c['worker_name'] = ''

    return render_template('citizen_dashboard.html', citizen=user, complaints=my_complaints)


@app.route('/citizen/register_complaint', methods=['POST'])
def register_complaint():
    if 'user_id' not in session:
        return redirect(url_for('citizen_login'))

    complaint_type = (request.form.get('complaint_type') or '').strip()
    description = (request.form.get('description') or '').strip()

    if not complaint_type:
        flash('Please select a complaint type.', 'danger')
        return redirect(url_for('citizen_dashboard'))
    if not description or len(description) < 10:
        flash('Description must be at least 10 characters.', 'danger')
        return redirect(url_for('citizen_dashboard'))

    image_path = None
    if 'complaint_image' in request.files:
        file = request.files['complaint_image']
        if file and file.filename != '' and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                if not filename:
                    filename = f"image_{int(datetime.now().timestamp())}.jpg"
                unique_name = f"{session['user_id']}_{int(datetime.now().timestamp())}_{filename}"
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(full_path)
                image_path = f"uploads/{unique_name}"
            except OSError:
                # Vercel has a read-only filesystem — image cannot be saved.
                # Inform the citizen so they're not confused about the missing photo.
                flash(
                    'Image upload is not available in the hosted version. '
                    'Your complaint was still submitted successfully.',
                    'warning'
                )

    complaint_dict = {
        'user_id': session['user_id'],
        'worker_id': None,
        'complaint_type': complaint_type,
        'description': description,
        'latitude': request.form.get('latitude') or 'N/A',
        'longitude': request.form.get('longitude') or 'N/A',
        'department': 'Unassigned',
        'status': 'Pending',
        'image_path': image_path,
        'remarks': None,
        'created_at': datetime.utcnow().isoformat(),
    }

    add_complaint(complaint_dict)

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

    user = get_user_by_id(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('citizen_login'))

    submitted_email = (request.form.get('email') or '').strip().lower()
    if submitted_email and submitted_email != (user.get('email') or '').lower():
        flash('Email is managed by Firebase Authentication and cannot be changed here.', 'info')

    new_password = request.form.get('new_password')
    if new_password:
        flash('Password updates are managed by Firebase Authentication.', 'info')

    update_user(session['user_id'], {
        'name': request.form.get('name', user.get('name', '')),
        'phone': request.form.get('phone', user.get('phone', '')),
        'address': request.form.get('address', user.get('address', '')),
    })

    flash('Profile updated successfully.', 'success')
    return redirect(url_for('citizen_dashboard') + '#profile')


# --- ADMIN ROUTES ---

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        login_input = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        # Find admin by username OR email
        user = None
        for uid, u in _all_users().items():
            if not u.get('is_admin'):
                continue
            if (u.get('username', '').lower() == login_input.lower() or
                    u.get('email', '').lower() == login_input.lower()):
                u['id'] = uid
                user = u
                break

        if user and check_password_hash(user.get('password_hash', ''), password):
            session['user_id'] = user['id']
            session['is_admin'] = True
            session['admin_logged_in'] = True
            session.pop('citizen_logged_in', None)
            flash('Admin access granted.', 'success')
            return redirect(url_for('admin_dashboard'))

        flash('Invalid admin credentials.', 'danger')

    return render_template('admin_login.html')


@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        name = (request.form.get('name') or '').strip()

        if get_user_by_username(username) or get_user_by_email(email):
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('admin_register'))

        admin_id = str(uuid.uuid4()).replace('-', '')[:20]
        save_user(admin_id, {
            'username': username,
            'password_hash': generate_password_hash(password),
            'name': name,
            'email': email,
            'phone': request.form.get('phone', ''),
            'address': request.form.get('address', ''),
            'department': request.form.get('department', ''),
            'is_admin': True,
        })

        flash('Admin registration successful! Please login.', 'success')
        return redirect(url_for('admin_login'))

    return render_template('admin_registration.html')


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    admin_user = get_user_by_id(session.get('user_id'))
    all_complaints = get_all_complaints()

    # Enrich with user and worker info
    for c in all_complaints:
        u = get_user_by_id(c.get('user_id'))
        c['author_name'] = u.get('name', 'Unknown') if u else 'Unknown'
        w_id = c.get('worker_id')
        if w_id:
            w = get_worker_by_id(w_id)
            c['worker_name'] = w.get('name', '') if w else ''
        else:
            c['worker_name'] = ''

    dept_counts = {}
    for c in all_complaints:
        dept = c.get('department') or 'Unassigned'
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    total = len(all_complaints) if all_complaints else 1
    dept_stats = [{'name': dept, 'count': count, 'pct': int(count / total * 100)}
                  for dept, count in dept_counts.items()]
    dept_stats.sort(key=lambda x: x['count'], reverse=True)

    all_workers = get_all_workers(available_only=True)

    return render_template('admin_dashboard.html',
                           complaints=all_complaints,
                           admin=admin_user,
                           dept_stats=dept_stats,
                           workers=all_workers)


@app.route('/admin/update_complaint/<complaint_id>', methods=['POST'])
def update_complaint(complaint_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        flash('Complaint not found.', 'danger')
        return redirect(url_for('admin_dashboard'))

    worker_id = (request.form.get('worker_id') or '').strip()

    update_complaint_data(complaint_id, {
        'status': request.form.get('status', complaint.get('status')),
        'department': request.form.get('department', complaint.get('department')),
        'remarks': request.form.get('remarks', complaint.get('remarks')),
        'worker_id': worker_id if worker_id else None,
    })

    flash('Complaint updated successfully.', 'success')

    next_page = request.form.get('next', '')
    parsed = urlparse(next_page)
    if parsed.netloc or parsed.scheme or not next_page.startswith('/'):
        next_page = url_for('admin_dashboard')
    return redirect(next_page)


# --- ALL COMPLAINTS PAGE ---

@app.route('/admin/complaints')
def admin_complaints():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    admin_user = get_user_by_id(session.get('user_id'))
    all_complaints = get_all_complaints()

    for c in all_complaints:
        u = get_user_by_id(c.get('user_id'))
        c['author_name'] = u.get('name', 'Unknown') if u else 'Unknown'
        w_id = c.get('worker_id')
        if w_id:
            w = get_worker_by_id(w_id)
            c['worker_name'] = w.get('name', '') if w else ''
        else:
            c['worker_name'] = ''

    all_workers = get_all_workers(available_only=True)
    return render_template('admin_complaints.html',
                           complaints=all_complaints,
                           admin=admin_user,
                           workers=all_workers)


# --- DEPARTMENT REPORTS ---

@app.route('/admin/reports')
def admin_reports():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    admin_user = get_user_by_id(session.get('user_id'))
    all_complaints = get_all_complaints()

    dept_data = {}
    for c in all_complaints:
        dept = c.get('department') or 'Unassigned'
        if dept not in dept_data:
            dept_data[dept] = {'total': 0, 'pending': 0, 'in_progress': 0, 'resolved': 0, 'rejected': 0}
        dept_data[dept]['total'] += 1
        status = c.get('status', '')
        if status == 'Pending':
            dept_data[dept]['pending'] += 1
        elif status == 'In Progress':
            dept_data[dept]['in_progress'] += 1
        elif status == 'Resolved':
            dept_data[dept]['resolved'] += 1
        elif status == 'Rejected':
            dept_data[dept]['rejected'] += 1

    return render_template('admin_reports.html',
                           dept_data=dept_data,
                           total_complaints=len(all_complaints),
                           admin=admin_user)


# --- USER MANAGEMENT ---

@app.route('/admin/users')
def admin_users():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    admin_user = get_user_by_id(session.get('user_id'))
    all_users_raw = _all_users()
    citizens = []
    admins = []
    for uid, u in all_users_raw.items():
        if not isinstance(u, dict):
            continue
        u['id'] = uid
        if u.get('is_admin'):
            admins.append(u)
        else:
            citizens.append(u)
    citizens.sort(key=lambda x: x.get('name', ''))
    admins.sort(key=lambda x: x.get('name', ''))

    return render_template('admin_users.html', citizens=citizens, admins=admins, admin=admin_user)


@app.route('/admin/users/<user_id>/delete', methods=['POST'])
def delete_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    if user_id == session.get('user_id'):
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))

    user = get_user_by_id(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))
    if user.get('is_admin'):
        flash('Cannot delete admin accounts from here.', 'warning')
        return redirect(url_for('admin_users'))

    delete_complaints_by_user(user_id)
    delete_user_data(user_id)
    flash(f'User "{user.get("name", user_id)}" deleted.', 'success')
    return redirect(url_for('admin_users'))


# --- WORKER MANAGEMENT ---

@app.route('/admin/workers', methods=['GET', 'POST'])
def admin_workers():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        worker_dict = {
            'name': request.form.get('name', ''),
            'phone': request.form.get('phone', ''),
            'department': request.form.get('department', ''),
            'area': request.form.get('area', ''),
            'is_available': True,
            'created_at': datetime.utcnow().isoformat(),
        }
        new_key = add_worker(worker_dict)
        flash(f'Worker "{worker_dict["name"]}" added successfully.', 'success')
        return redirect(url_for('admin_workers'))

    admin_user = get_user_by_id(session.get('user_id'))
    workers = get_all_workers()
    workers.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return render_template('admin_workers.html', workers=workers, admin=admin_user)


@app.route('/admin/workers/<worker_id>/delete', methods=['POST'])
def delete_worker_route(worker_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    worker = get_worker_by_id(worker_id)
    if not worker:
        flash('Worker not found.', 'danger')
        return redirect(url_for('admin_workers'))

    # Unassign worker from complaints
    for cid, c in _all_complaints().items():
        if isinstance(c, dict) and str(c.get('worker_id', '')) == str(worker_id):
            update_complaint_data(cid, {'worker_id': None})

    delete_worker(worker_id)
    flash(f'Worker "{worker.get("name", worker_id)}" deleted.', 'success')
    return redirect(url_for('admin_workers'))


@app.route('/admin/workers/<worker_id>/toggle', methods=['POST'])
def toggle_worker(worker_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    worker = get_worker_by_id(worker_id)
    if not worker:
        flash('Worker not found.', 'danger')
        return redirect(url_for('admin_workers'))

    new_status = not worker.get('is_available', True)
    update_worker(worker_id, {'is_available': new_status})
    status_text = 'available' if new_status else 'unavailable'
    flash(f'Worker "{worker.get("name", worker_id)}" marked as {status_text}.', 'success')
    return redirect(url_for('admin_workers'))


# --- ADMIN SETTINGS ---

@app.route('/admin/settings', methods=['GET', 'POST'])
def admin_settings():
    if not session.get('is_admin'):
        flash('Restricted access.', 'danger')
        return redirect(url_for('admin_login'))

    admin_user = get_user_by_id(session.get('user_id'))

    if request.method == 'POST':
        updates = {
            'name': request.form.get('name', admin_user.get('name', '')),
            'email': request.form.get('email', admin_user.get('email', '')),
            'phone': request.form.get('phone', admin_user.get('phone', '')),
        }
        new_password = request.form.get('new_password')
        if new_password and len(new_password) >= 6:
            updates['password_hash'] = generate_password_hash(new_password)
            flash('Password updated.', 'success')

        update_user(session.get('user_id'), updates)
        flash('Settings saved.', 'success')
        return redirect(url_for('admin_settings'))

    return render_template('admin_settings.html', admin=admin_user)


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Admin logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/complaint/<complaint_id>')
def view_complaint(complaint_id):
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        flash('Complaint not found.', 'danger')
        return redirect(url_for('index'))
    return render_template('complaint_detail.html', complaint=complaint)


# --- NLP ANALYSIS (admin) ---

# Simple keyword → department mapping (no heavy ML required)
_NLP_KEYWORD_MAP = {
    'Sanitation': ['garbage', 'waste', 'trash', 'litter', 'drain', 'sewage', 'sewer',
                   'stink', 'smell', 'dirty', 'filth', 'sanitation', 'compost'],
    'Roads & Traffic': ['road', 'pothole', 'traffic', 'signal', 'crack', 'street',
                        'pavement', 'footpath', 'sidewalk', 'lane', 'highway', 'bridge'],
    'Water Supply': ['water', 'pipe', 'leak', 'leakage', 'tap', 'supply', 'flood',
                     'waterlog', 'overflow', 'contaminate', 'shortage', 'pump'],
    'Electricity': ['electricity', 'electric', 'light', 'power', 'wire', 'pole',
                    'streetlight', 'transformer', 'outage', 'blackout', 'voltage'],
    'Public Health': ['health', 'hospital', 'clinic', 'disease', 'mosquito', 'dengue',
                      'malaria', 'flu', 'covid', 'pandemic', 'sanitary', 'hygiene'],
}

_NEGATIVE_WORDS = {'broken', 'damaged', 'dirty', 'dangerous', 'urgent', 'bad', 'worst',
                   'terrible', 'horrible', 'disgusting', 'unacceptable', 'severe',
                   'immediate', 'overflowing', 'blocked', 'leaking', 'crack', 'pothole',
                   'smell', 'stink', 'dead', 'flooding', 'contaminated'}

_POSITIVE_WORDS = {'good', 'great', 'excellent', 'clean', 'resolved', 'fixed', 'ok',
                   'better', 'improved', 'nice', 'working', 'repaired'}

_STOP_WORDS = {'the', 'is', 'in', 'it', 'and', 'a', 'an', 'of', 'to', 'for', 'on',
               'at', 'by', 'we', 'our', 'my', 'this', 'that', 'are', 'was', 'with',
               'there', 'has', 'have', 'be', 'been', 'not', 'no', 'very', 'also',
               'from', 'but', 'its', 'they', 'their', 'near', 'since', 'i', 'im'}


def _nlp_analyze_text(text):
    """Keyword extraction, department suggestion, and sentiment on complaint text."""
    words = re.findall(r'[a-z]+', (text or '').lower())
    # Frequency count (ignore stop words, single chars)
    freq = {}
    for w in words:
        if w not in _STOP_WORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    # Top keywords
    keywords = sorted(freq, key=freq.get, reverse=True)[:10]

    # Department suggestion
    dept_scores = {}
    for dept, kws in _NLP_KEYWORD_MAP.items():
        score = sum(freq.get(k, 0) for k in kws)
        if score:
            dept_scores[dept] = score
    suggested_dept = max(dept_scores, key=dept_scores.get) if dept_scores else 'Unassigned'

    # Simple sentiment
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    if neg > pos:
        sentiment = 'Negative'
    elif pos > neg:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'

    return {
        'keywords': keywords,
        'suggested_department': suggested_dept,
        'sentiment': sentiment,
        'word_count': len(words),
    }


@app.route('/admin/nlp_analyze', methods=['POST'])
def admin_nlp_analyze():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json(silent=True) or {}
    complaint_id = (data.get('complaint_id') or '').strip()
    description = (data.get('description') or '').strip()

    # Prefer fetching description from DB if complaint_id given
    if complaint_id:
        complaint = get_complaint_by_id(complaint_id)
        if not complaint:
            return jsonify({'error': 'Complaint not found.'}), 404
        description = complaint.get('description', '')

    if not description:
        return jsonify({'error': 'No description text to analyze.'}), 400

    result = _nlp_analyze_text(description)
    return jsonify(result), 200


# ==================== INITIALIZATION ====================
def seed_default_admin():
    """Create a default admin account if none exists in RTDB."""
    for uid, u in _all_users().items():
        if u.get('is_admin'):
            return  # at least one admin exists
    print("No admin found in RTDB. Creating default admin account (admin / admin123)...")
    admin_id = str(uuid.uuid4()).replace('-', '')[:20]
    save_user(admin_id, {
        'username': 'admin',
        'password_hash': generate_password_hash('admin123'),
        'name': 'System Admin',
        'email': 'admin@civic.local',
        'phone': '',
        'address': '',
        'is_admin': True,
    })
    print(f"Default admin created with ID: {admin_id}")


if __name__ == '__main__':
    ensure_firebase_admin_initialized()
    seed_default_admin()
    app.run(debug=True)
