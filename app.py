import os
from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import speech_recognition as sr
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///facts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp3', 'wav', 'ogg'}
db = SQLAlchemy(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database
class Fact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_type = db.Column(db.String(32))
    hypothesis = db.Column(db.Text)
    content = db.Column(db.Text)
    credibility = db.Column(db.Float)
    final_score = db.Column(db.Float)
    proven = db.Column(db.Boolean, default=False)
    result = db.Column(db.String(8))
    source_identity = db.Column(db.Float, default=0.0)
    expertise = db.Column(db.Float, default=0.0)
    bias = db.Column(db.Float, default=0.0)
    consistency = db.Column(db.Float, default=0.0)
    past_reliability = db.Column(db.Float, default=0.0)
    metadata_score = db.Column(db.Float, default=0.0)
    metadata = db.Column(db.Text)

# --- AI/NLP Pipelines ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# --- Metadata extraction ---
def get_image_metadata(image_path):
    metadata = {}
    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[str(tag)] = str(value)
    return metadata

def get_audio_metadata(audio_path):
    try:
        audio = MP3(audio_path, ID3=EasyID3)
        return dict(audio)
    except Exception as e:
        return {"error": str(e)}

# --- Analysis functions ---
def analyze_text(text):
    summary = summarizer(text, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
    credibility_result = classifier(text)[0]
    credibility_label = credibility_result['label']
    credibility_score = credibility_result['score']
    details = {
        "source_identity": 0.75,
        "expertise": 0.8,
        "credibility_score": round(credibility_score, 2),
        "bias": 0.6,
        "consistency": 0.7,
        "past_reliability": 0.7,
        "metadata_score": 0.6,
    }
    analysis = {
        "input_type": f"Text ({credibility_label})",
        "score": round(sum(details.values()) / len(details), 2),
        "details": details,
        "closest_match": summary,
        "summary": summary,
        "metadata": {},
        "source_identity": details["source_identity"],
        "expertise": details["expertise"],
        "bias": details["bias"],
        "consistency": details["consistency"],
        "past_reliability": details["past_reliability"],
        "metadata_score": details["metadata_score"],
    }
    return analysis

def analyze_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    manipulation_score = np.sum(edges) / (image.shape[0] * image.shape[1])
    details = {
        "source_identity": 0.6,
        "expertise": 0.65,
        "credibility_score": 0.55,
        "bias": 0.5,
        "consistency": 0.65,
        "past_reliability": 0.7,
        "metadata_score": 0.6,
    }
    closest_match = (
        "Image seems original."
        if manipulation_score < 2.0
        else "Possible manipulation detected."
    )
    metadata = get_image_metadata(image_path)
    analysis = {
        "input_type": "Image",
        "score": round(sum(details.values()) / len(details), 2),
        "details": details,
        "closest_match": closest_match,
        "summary": closest_match,
        "metadata": metadata,
        "source_identity": details["source_identity"],
        "expertise": details["expertise"],
        "bias": details["bias"],
        "consistency": details["consistency"],
        "past_reliability": details["past_reliability"],
        "metadata_score": details["metadata_score"],
    }
    return analysis

def analyze_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        credibility_result = classifier(text)[0]
        credibility_score = credibility_result['score']
        details = {
            "source_identity": 0.7,
            "expertise": 0.75,
            "credibility_score": round(credibility_score, 2),
            "bias": 0.55,
            "consistency": 0.7,
            "past_reliability": 0.6,
            "metadata_score": 0.65,
        }
        match = f"Transcribed: {text[:80]}..."
    except Exception as e:
        details = {k: 0.5 for k in [
            "source_identity","expertise","credibility_score","bias","consistency","past_reliability","metadata_score"]}
        match = f"Transcription failed. ({e})"
        text = ""
    metadata = get_audio_metadata(audio_path)
    analysis = {
        "input_type": "Audio",
        "score": round(sum(details.values()) / len(details), 2),
        "details": details,
        "closest_match": match,
        "summary": match,
        "metadata": metadata,
        "source_identity": details["source_identity"],
        "expertise": details["expertise"],
        "bias": details["bias"],
        "consistency": details["consistency"],
        "past_reliability": details["past_reliability"],
        "metadata_score": details["metadata_score"],
        "transcription": text
    }
    return analysis

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    analysis_metadata = {}
    input_type = ""
    hypothesis = ""
    content = ""
    credibility_db = ""
    if request.method == 'POST':
        text_input = request.form.get('text_input', '').strip()
        file = request.files.get('file_input', None)
        if text_input:
            analysis = analyze_text(text_input)
            results = analysis
            input_type = analysis["input_type"]
            hypothesis = analysis.get("summary", "")
            content = text_input
            credibility_db = analysis.get("score", "")
            analysis_metadata = analysis.get("metadata", {})
        elif file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['jpg', 'jpeg', 'png']:
                analysis = analyze_image(filepath)
                results = analysis
                input_type = "Image"
                hypothesis = analysis.get("summary", "")
                content = filename
                credibility_db = analysis.get("score", "")
                analysis_metadata = analysis.get("metadata", {})
            elif ext in ['mp3', 'wav', 'ogg']:
                analysis = analyze_audio(filepath)
                results = analysis
                input_type = "Audio"
                hypothesis = analysis.get("summary", "")
                content = analysis.get('transcription', '')
                credibility_db = analysis.get("score", "")
                analysis_metadata = analysis.get("metadata", {})
        else:
            results = {"error": "No input or unsupported file type."}
    if results and not results.get("error"):
        results['input_type_db'] = input_type
        results['hypothesis_db'] = hypothesis
        results['content_db'] = content
        results['credibility_db'] = credibility_db
        results['metadata_db'] = str(analysis_metadata)
    return render_template('index.html', results=results)

@app.route('/save_fact', methods=['POST'])
def save_fact():
    result = request.form.get('result', 'wrong')
    input_type = request.form.get('input_type_db', '')
    hypothesis = request.form.get('hypothesis_db', '')
    content = request.form.get('content_db', '')
    credibility = float(request.form.get('credibility_db', 0))
    metadata = request.form.get('metadata_db', '')[:8000]
    proven = request.form.get('proven', 'false') == 'true'
    new_fact = Fact(
        input_type=input_type,
        hypothesis=hypothesis,
        content=content,
        credibility=credibility,
        final_score=credibility,
        proven=proven,
        result=result,
        metadata=metadata,
    )
    db.session.add(new_fact)
    db.session.commit()
    return redirect(url_for('facts'))

@app.route('/facts')
def facts():
    all_facts = Fact.query.order_by(Fact.id.desc()).all()
    return render_template('facts.html', facts=all_facts)

@app.route('/toggle_proven/<int:fact_id>', methods=['POST'])
def toggle_proven(fact_id):
    fact = Fact.query.get_or_404(fact_id)
    fact.proven = not fact.proven
    db.session.commit()
    return redirect(url_for('facts'))

@app.route('/delete_fact/<int:fact_id>', methods=['POST'])
def delete_fact(fact_id):
    fact = Fact.query.get_or_404(fact_id)
    db.session.delete(fact)
    db.session.commit()
    return redirect(url_for('facts'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
