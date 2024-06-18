from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
face_classifier = cv2.CascadeClassifier('D:/Mus_Ap/emotionDetection/emotionDetection/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('D:/Mus_Ap/emotionDetection/emotionDetection/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('user_management.db')
    conn.row_factory = sqlite3.Row
    return conn

def generate_unique_username(email, conn):
    username = email.split('@')[0]
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
    count = cursor.fetchone()[0]
    
    if count == 0:
        return username
    else:
        suffix = 1
        new_username = f"{username}{suffix}"
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (new_username,))
        count = cursor.fetchone()[0]
        
        while count > 0:
            suffix += 1
            new_username = f"{username}{suffix}"
            cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (new_username,))
            count = cursor.fetchone()[0]
        
        return new_username

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label
    return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        # Handle signup form submission
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect(url_for('home'))
        
        # Your signup logic here
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Insert user data into the database
            # Don't forget to hash the password before storing it
            hashed_password = generate_password_hash(password)
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)',
                           (email, hashed_password))
            conn.commit()
            flash('Signup successful!')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists!')
            return redirect(url_for('home'))
        finally:
            conn.close()

    # If GET request, render the signup.html template
    
    return render_template('signup.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        # Handle login form submission
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            # Set the user's ID in the session
            session['user_id'] = user['id']
            flash('Login successful!')
            return redirect(url_for('index'))  # Assuming the index route exists
        else:
            flash('Invalid email or password!')
            return redirect(url_for('home'))  # Redirect to home page on login failure
    
    # If GET request, render the login.html template
    return render_template('login.html')  # Make sure you have a login.html template


@app.route('/detect_emotion', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get('image_data')

        # Convert base64 image data to numpy array
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detected_emotion = detect_emotion(frame)
        if detected_emotion:
            return jsonify({'emotion': detected_emotion})
        else:
            return jsonify({'error': 'No emotion detected.'})
    except Exception as e:
        return jsonify({'error': 'Error processing image.'})

if __name__ == '__main__':
    # Load emotion detection model
    emotion_classifier = load_model('D:\Mus_Ap\emotionDetection\emotionDetection\model.h5')
    print('Emotion detection model loaded successfully.')
    app.run(debug=True)
