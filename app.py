from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file, Response
from flask_bcrypt import Bcrypt
from deepface import DeepFace
import logging
import warnings
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import cv2
import pymysql
import os
from werkzeug.utils import secure_filename
# ======chatbot====================
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ======chatbot====================


# ====================Ayurveda Presciption Generator================
# Suppress TensorFlow and OpenCV warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
# ====================Ayurveda Presciption Generator================


# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'Team23'  # Keep this secret in production
bcrypt = Bcrypt(app)

# MySQL connection
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='Gani@2000',
    database='ayurcure_db',
    cursorclass=pymysql.cursors.DictCursor  # Fetch results as dictionaries
)
cursor = conn.cursor()

# Create uploads folder if not exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ====================Ayurveda Oresciption Generator================
camera = cv2.VideoCapture(0)
logging.basicConfig(level=logging.INFO)


# Emotion to health suggestion
emotion_symptom_map = {
    "happy": "You seem fine! Keep smiling ",
    "sad": "You might be feeling low. Herbal tea and rest can help.",
    "angry": "Try deep breathing or Brahmi tea to calm your nerves.",
    "surprise": "Surprised? Make sure it's not stress-induced shock.",
    "fear": "Ashwagandha and meditation may help with anxiety.",
    "disgust": "Feeling uneasy? Fresh air and hydration might help.",
    "neutral": "You look neutral. Stay hydrated and take breaks."
}

# ======== Load datasets ==========
df = pd.read_excel("Healthcare_Disease_Treatment_Data_500.xlsx")
X = df['Symptoms']
y_disease = df['Disease']
medicines = df[['Medicine Suggested', 'Medicinal Composition', 'Treatment Description']]

X_train, X_test, y_train, y_test, med_train, med_test = train_test_split(X, y_disease, medicines, test_size=0.2, random_state=42)

pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
pipeline.fit(X_train, y_train)

plant_data = pd.read_excel("medicinalplantsdataset.xlsx")
X_plant = plant_data['symptoms']
y_plant = plant_data[['Hindi Name', 'English Name', 'Botanical Name', 'Remedy']]

plant_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
plant_model.fit(X_plant, y_plant['Hindi Name'])

# ====================Ayurveda Oresciption Generator================

# Home Route
@app.route('/')
def home():
    return redirect(url_for('login'))

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        mobile = request.form.get('mobile').strip()
        profile_pic = request.files.get('profile_pic')
        agree = request.form.get('agree')

        # Validations
        if not all([username, password, confirm_password, mobile, profile_pic, agree]):
            flash('All fields are required and you must agree to the terms.', 'danger')
            return redirect(url_for('register'))

        if not (mobile.isdigit() and len(mobile) == 10):
            flash('Mobile number must be exactly 10 digits!', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            flash('Username already exists. Please login.', 'warning')
            return redirect(url_for('register'))

        filename = secure_filename(profile_pic.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        profile_pic.save(filepath)

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        cursor.execute("""
            INSERT INTO users (username, password, mobile, profile_pic) 
            VALUES (%s, %s, %s, %s)
        """, (username, hashed_password, mobile, filename))
        conn.commit()

        flash('Registered successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['profile_pic'] = user['profile_pic']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))

    return render_template('dashboard.html', 
                           username=session['username'], 
                           profile_pic=session['profile_pic'], 
                           active_page='dashboard')

# Ayurveda Prescription Route
@app.route('/ayurveda_prescription')
def ayurveda_prescription():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))

    return render_template('ayurveda_prescription.html', 
                           username=session['username'], 
                           profile_pic=session['profile_pic'], 
                           active_page='ayurveda_prescription')

# Contact Route
@app.route('/contact')
def contact():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))

    return render_template('contact.html', 
                           username=session['username'], 
                           profile_pic=session['profile_pic'], 
                           active_page='contact')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# ====================Ayurveda Oresciption Generator================
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_emotion', methods=['POST'])
def capture_emotion():
    global captured_emotion, captured_health_tip, captured_image_path

    camera = cv2.VideoCapture(0)  # Open camera locally for capture

    success, frame = camera.read()
    if not success:
        camera.release()
        return {"error": "Failed to capture image"}, 500

    resized_frame = cv2.resize(frame, (320, 240))

    try:
        result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        detected_emotion = result[0]['dominant_emotion']
        emotions = result[0]['emotion']

        if 'disgust' in emotions and emotions['disgust'] > 15:
            detected_emotion = "disgust"

        health_tip = emotion_symptom_map.get(detected_emotion.lower(), "Stay Healthy!")

        captured_emotion = detected_emotion.capitalize()
        captured_health_tip = health_tip

        os.makedirs('static/captures', exist_ok=True)
        captured_image_path = f'static/captures/captured_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        cv2.imwrite(captured_image_path, frame)

        camera.release()  # ✅ Release the camera after capture
        cv2.destroyAllWindows()

        return {
            "emotion": captured_emotion,
            "health_tip": captured_health_tip
        }
    except Exception as e:
        camera.release()  # ✅ Release camera even if exception happens
        cv2.destroyAllWindows()
        logging.warning(f"Emotion detection error: {e}")
        return {"error": "Emotion detection failed"}, 500


@app.route('/generate_plant_report', methods=['POST'])
def generate_plant_report():
    global captured_emotion, captured_health_tip, captured_image_path

    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    symptoms = request.form.get('symptoms')

    # Check if emotion was captured
    if not (captured_emotion and captured_health_tip and captured_image_path):
        return "Please capture the emotion before submitting!", 400

    recommendation = predict_plant_medicine(symptoms)

    pdf_path = generate_plant_pdf(symptoms, recommendation, patient_name, patient_age,
                                  captured_emotion, captured_health_tip, captured_image_path)

    # Reset after PDF generation
    captured_emotion = None
    captured_health_tip = None
    captured_image_path = None

    return send_file(pdf_path, as_attachment=True)

# ========== Helper Functions ==========

def gen_frames():
    frame_count = 0
    last_emotion = "neutral"
    detection_cooldown = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (320, 240))

        if frame_count % 10 == 0 and detection_cooldown <= 0:
            try:
                result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                detected_emotion = result[0]['dominant_emotion']
                emotions = result[0]['emotion']

                if 'disgust' in emotions and emotions['disgust'] > 15:
                    detected_emotion = "disgust"

                last_emotion = detected_emotion
                detection_cooldown = 30
            except Exception as e:
                logging.warning(f"Emotion detection error: {e}")

        if detection_cooldown > 0:
            detection_cooldown -= 1

        display_text = f"Emotion: {last_emotion.capitalize()}"
        health_tip = emotion_symptom_map.get(last_emotion.lower(), "Stay Healthy!")

        cv2.putText(frame, display_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, health_tip, (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def predict_plant_medicine(symptoms):
    predicted_plant_name = plant_model.predict([symptoms])[0]
    plant_details = plant_data[plant_data['Hindi Name'] == predicted_plant_name].iloc[0]

    return {
        'Hindi Name': plant_details['Hindi Name'],
        'English Name': plant_details['English Name'],
        'Botanical Name': plant_details['Botanical Name'],
        'Remedy': plant_details['Remedy']
    }

def generate_plant_pdf(symptoms, recommendation, patient_name="Patient", patient_age="N/A",
                       emotion="Neutral", health_tip="Stay Healthy!", captured_image_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Top-right: Generated timestamp
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.set_xy(150, 10)  # Move cursor to near top-right
    pdf.cell(50, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='R')
    
    # Header
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(0, 100, 0)
    pdf.ln(10)  # Add a little space after timestamp
    pdf.cell(0, 15, 'AYURVEDIC PRESCRIPTION', ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_draw_color(0, 100, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Patient Details
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'PATIENT DETAILS', ln=True)
    pdf.set_font('Arial', '', 12)
    col_width = 90
    row_height = 10
    pdf.cell(col_width, row_height, f'Name: {patient_name}', border=0)
    pdf.cell(col_width, row_height, f'Age: {patient_age}', border=0, ln=True)
    pdf.ln(5)
    
    # Diagnosis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'DIAGNOSIS', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Reported Symptoms:', ln=True)
    # No background color here
    pdf.multi_cell(0, 8, symptoms, border=0, fill=False)
    pdf.ln(5)
    
    # Prescription and Image side-by-side
    y_before = pdf.get_y()
    
    # LEFT: Prescription (Plant Details)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_xy(10, y_before)
    pdf.cell(90, 10, 'PRESCRIPTION', ln=True)
    pdf.set_font('Arial', '', 12)
    
    plant_details = [
        ("Hindi Name", recommendation['Hindi Name']),
        ("English Name", recommendation['English Name']),
        ("Botanical Name", recommendation['Botanical Name']),
        ("Remedy", recommendation['Remedy'])
    ]
    
    for label, value in plant_details:
        pdf.set_x(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(40, 8, f"{label}:", border=0)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(50, 8, value)
    
    # RIGHT: Captured Image
    if captured_image_path and os.path.exists(captured_image_path):
        try:
            pdf.set_xy(110, y_before)
            pdf.image(captured_image_path, x=110, y=y_before, w=80, h=80)
        except Exception as e:
            logging.error(f"Failed to add image to PDF: {e}")
            pdf.set_xy(110, y_before)
            pdf.set_font('Arial', 'I', 12)
            pdf.cell(80, 10, '[Image unavailable]', border=1)
    else:
        pdf.set_xy(110, y_before)
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(80, 10, '[No image captured]', border=1)
    
    # Move cursor below the image area
    pdf.set_y(y_before + 85)

    # Now BELOW IMAGE: Add Emotion Analysis
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'EMOTION ANALYSIS', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, f"Detected Emotion: {emotion}\nHealth Tip: {health_tip}", border=1)
    pdf.ln(10)

    # Footer Note (no timestamp here now)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Note: This prescription is based on AI.', ln=True)
    
    # Save the file
    os.makedirs('static/reports', exist_ok=True)
    file_path = f"static/reports/{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(file_path)
    
    return file_path

# ====================Ayurveda Oresciption Generator================

# ======chatbot====================
# Load and preprocess dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['Disease/Symptoms'] = df['Disease/Symptoms'].str.lower().str.strip()
    df['Remedy'] = df['Remedy'].str.lower().str.strip()
    return df

# Train TF-IDF model
def train_tfidf_model(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Disease/Symptoms'])
    return vectorizer, tfidf_matrix

# Handle greetings and farewells
def check_greetings_or_farewells(query):
    greetings = ["hello", "hi", "hey", "greetings"]
    farewells = ["bye", "goodbye", "exit", "see you", "farewell"]

    query_lower = query.lower().strip()
    if query_lower in greetings:
        return "Hello! How can I assist you with Ayurveda remedies today?"
    elif query_lower in farewells:
        return "Goodbye! Stay healthy and take care. 😊"
    return None  # Not a greeting or farewell

# Find remedies for multiple symptoms
def find_best_match(query, df, vectorizer, tfidf_matrix):
    query_check = check_greetings_or_farewells(query)
    if query_check:
        return [query_check]  # Return greeting/farewell response

    symptoms = [symptom.strip().lower() for symptom in query.split(',')]  # Process comma-separated input
    remedies = set()

    for symptom in symptoms:
        query_vector = vectorizer.transform([symptom])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        top_n = 3
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        found_remedies = [df.iloc[i]['Remedy'] for i in top_indices if similarities[i] > 0.1]
        
        remedies.update(found_remedies)

    return list(remedies) if remedies else ["No relevant remedy found."]

# Load dataset and train model
df = load_dataset("ayurveda_remedies_extended.csv")
vectorizer, tfidf_matrix = train_tfidf_model(df)


# Chatbot Route
@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))

    return render_template('chatbot.html', 
                           username=session['username'], 
                           profile_pic=session['profile_pic'], 
                           active_page='chatbot')


@app.route('/get_remedy', methods=['POST'])
def get_remedy():
    data = request.json
    query = data.get("query", "")
    remedies = find_best_match(query, df, vectorizer, tfidf_matrix)
    
    return jsonify({"remedy": ', '.join(remedies)})

# ======chatbot====================


# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


