import os
import sqlite3
import numpy as np
import pandas as pd
import cv2 # type: ignore
from werkzeug.security import generate_password_hash, check_password_hash # type: ignore
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Load the pre-trained models
mobilenetmodel = load_model('mobilenet_model.keras')  # Model for Mild/Severe classification
inceptionmodel = load_model('final_model.keras')  # Model for Cataract/Normal classification

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  
    return conn


def detect_eye(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return False  # Return False if the image cannot be loaded

    # Resize the image for consistent detection (optional, helps with very large images)
    scale_percent = 50  # Adjust as needed
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast (helps with detecting faint edges or details)
    gray = cv2.equalizeHist(gray)

    # Detect eyes in the image using Haar Cascade
    eyes = eye_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # Reduce step size for finer detection
        minNeighbors=3,    # Lower threshold for sensitivity
        minSize=(15, 15),  # Allow detection of smaller features
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Check if eyes are detected
    if len(eyes) == 0:
        return False  # No eyes detected
    else:
        # Optionally, draw rectangles around detected eyes for debugging
        for (x, y, w, h) in eyes:
            cv2.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Save the debug image to visualize detection (optional)
        debug_path = 'static/uploads/detected_eyes.jpg'
        cv2.imwrite(debug_path, resized_img)

        return True  # Eyes detected


# Function to classify cataract or normal
def classify_cataract_or_normal(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 

    # Predict the cataract or normal class
    predictions = inceptionmodel.predict(img_array)
    predicted_class = (predictions > 0.5).astype(int)
    
    return 'Normal' if predicted_class[0][0] == 1 else 'Cataract'

# Function to classify mild or severe
def classify_mild_or_severe(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 

    # Predict the mild or severe class
    predictions = mobilenetmodel.predict(img_array)
    predicted_class = (predictions > 0.5).astype(int)
    
    return 'Severe' if predicted_class[0][0] == 1 else 'Mild'

# Registration route
@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('register'))

        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                hashed_password = generate_password_hash(password)
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Username already exists. Please log in.", "error")
                return redirect(url_for('login'))
    return render_template("register.html")

# Login route
@app.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):  # Validate password
            session['logged_in'] = True
            return redirect(url_for('index'))  # Redirect to index page after successful login
        else:
            flash('Invalid credentials. Please try again or register.', 'error')
    return render_template('login.html')




# Logout route
@app.route("/logout/")
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

UPLOAD_FOLDER = './GUI/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/export")
def export_data():
    conn = sqlite3.connect("cataract.db")
    df = pd.read_sql_query("SELECT * FROM patient_records", conn)
    conn.close()

    csv_path = "static/patient_data.csv"
    df.to_csv(csv_path, index=False)

    return send_file(csv_path, as_attachment=True, download_name="cataract_data.csv")




@app.route('/index/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':

        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        dob = request.form.get("dob")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        address = request.form.get("address")
        medical_history = request.form.get("medical-history")
        allergies = request.form.get("allergies")
        current_medications = request.form.get("current-medications")

        
        conn1 = sqlite3.connect("cataract.db")
        cursor = conn1.cursor()

        # Check if an image file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        # Create the uploads folder if it doesn't exist
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Save the uploaded file
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath) #this is file pathstatic/uploads/img6.jpeg
        # print("this is file path"+filepath)

        relative_image_path = os.path.join('uploads', file.filename)
        # print("This is relative path: "+ relative_image_path)

        # First, check if the image has an eye
        eye_detected = detect_eye(filepath)

        if not eye_detected:
            result = "Eye not detected. Ensure the image is clear and well-lit."
            cursor.execute('''
                INSERT INTO patient_records (name, age, gender, dob, email, mobile, address, medical_history, allergies, current_medications, image_path, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, age, gender, dob, email, mobile, address, medical_history, allergies, current_medications, relative_image_path, result))
            conn1.commit()
            conn1.close()
            return render_template('result.html', result=result, image_path=filepath)
        
        # if not eye_detected:
        #     return render_template('result.html', result="Eye not detected. Ensure the image is clear and well-lit.", image_path=filepath)

        
        # If eye is detected, proceed with cataract/normal classification
        cataract_result = classify_cataract_or_normal(filepath)


        if cataract_result == "Normal":
            result = f"Eye detected: {cataract_result}"
        else:
            severity_result = classify_mild_or_severe(filepath)
            result = f"Cataract detected. Severity: {severity_result}"

        # Store patient data in SQLite
        cursor.execute('''
            INSERT INTO patient_records (name, age, gender, dob, email, mobile, address, medical_history, allergies, current_medications, image_path, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, dob, email, mobile, address, medical_history, allergies, current_medications, relative_image_path, result))
        conn1.commit()
        conn1.close()
        return render_template('result.html', result=result, image_path=filepath)


        
        # if cataract_result == "Normal":
        #     return render_template('result.html', result=f"Eye detected: {cataract_result}", image_path=filepath,name=name,
        #     age=age,
        #     gender=gender,
        #     mobile=mobile,)
        
        # # If cataract detected, proceed with mild/severe classification
        # severity_result = classify_mild_or_severe(filepath)
        
        # return render_template('result.html', result=f"Cataract detected. Severity: {severity_result}", image_path=filepath,name=name,
        #     age=age,
        #     gender=gender,
        #     mobile=mobile,)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
