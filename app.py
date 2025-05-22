from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()

def rank_resumes(job_desc, resume_texts):
    documents = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity(job_vector, resume_vectors).flatten()
    return scores.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        job_desc = request.form['job_desc']
        uploaded_files = request.files.getlist('resumes')

        resume_texts = []
        filenames = []

        for file in uploaded_files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                text = extract_text_from_pdf(filepath)
                if text:
                    resume_texts.append(text)
                    filenames.append(filename)

        scores = rank_resumes(job_desc, resume_texts)
        results = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
