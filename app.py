from flask import Flask, render_template, request, redirect
import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter error
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import uuid


from flask import Flask, request, render_template,redirect, flash, session,jsonify,g
import numpy as np
import pickle
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import abort, redirect, url_for
import secrets
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import requests


app = Flask(__name__)

app.secret_key =secrets.token_hex(16)
model = load_model("audio_deepfake_cnn.h5")
class_names = ['Real Voice', 'Fake Voice']

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ✅ 1. For Model Prediction
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or crop to 128x128
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :128]

    return mel_db.reshape(1, 128, 128, 1) / 255.0, mel_db

# ✅ 2. For Saving Image
def create_melspectrogram(mel_db, sr):
    filename = f"mel_{uuid.uuid4().hex}.png"
    image_path = os.path.join(STATIC_FOLDER, filename)

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return image_path

# ✅ 3. Index Route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/services/login')
def login():
    return render_template('services/login.html')

@app.route('/services/register')
def register():
    return render_template('services/register.html')

@app.route('/services/index')
def index():
    return render_template("services/index.html")

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # to access columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        
# ✅ 4. Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    if not file:
        return redirect(request.url)

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    X, mel_db = extract_mel(path)
    pred = model.predict(X)
    label = np.argmax(pred)
    confidence = round(float(pred[0][label]) * 100, 2)

    image_path = create_melspectrogram(mel_db, sr=16000)

    return render_template('services/index.html',
                           prediction=class_names[label],
                           confidence=confidence,
                           image=image_path,
                           label_class="real" if label == 0 else "fake")

@app.route('/register_details', methods=['GET', 'POST'])
def register_details():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return render_template("services/login.html")
        except sqlite3.IntegrityError:
            flash("Account already exists with this email.", "error")
        conn.close()
    return render_template('services/register.html')

@app.route('/login_details', methods=['POST'])
def login_details():
    email = request.form['email']
    password = request.form['password']

    conn = sqlite3.connect('database.db')
    cur  = conn.cursor()
    cur.execute("SELECT name, password, is_admin FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if row and check_password_hash(row[1], password):
        session['user']       = row[0]        # username
        session['user_email'] = email         # email
        session['is_admin']   = bool(row[2])  # True/False
        flash("Login successful!", "success")
        return render_template('home.html')
    else:
        flash("Invalid email or password.", "error")
        return render_template('services/login.html')



def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('login'))
        if not session.get('is_admin', False):
            abort(403)   # Forbidden
        return f(*args, **kwargs)
    return wrapped


   
@app.route('/admin_users')
def admin_users():
    if 'user_email' not in session or session['user_email'] != 'megharashokashok@gmail.com':
        return redirect(url_for('login'))  # restrict non-admins
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT name, email,password FROM users")
    users = c.fetchall()
    conn.close()
    return render_template('services/admin_users.html', users=users)


    
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return render_template('home.html')
from flask import session, redirect, url_for, flash




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

