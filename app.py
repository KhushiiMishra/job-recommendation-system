import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ✅ Home Route
@app.route('/')
def home():
    return render_template('index.html')


# ✅ Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# ✅ Extract Skills
def extract_skills(text):
    skills_list = [
        'python', 'java', 'c++', 'machine learning',
        'data science', 'html', 'css', 'javascript',
        'sql', 'react', 'node', 'django'
    ]
    found_skills = []
    text = text.lower()

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills

# Match jobs
def match_jobs(user_text, selected_role):
    try:
        jobs = pd.read_csv("jobs_dataset.csv")
    except:
        return []

    job_descriptions = jobs['skills'].fillna("").tolist()

    # Combine resume + jobs
    all_texts = [user_text] + job_descriptions

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)

    # Compare resume with jobs
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    matched_jobs = []

    for i in range(len(similarity)):
        score = int(similarity[i] * 100)
        job_title = str(jobs.iloc[i]['job_title']).lower()

        if score > 10:
            if selected_role == "all" or selected_role.lower() in job_title:
                matched_jobs.append(
                    (
                        jobs.iloc[i]['job_title'],
                        score,
                        jobs.iloc[i]['link']
                    )
                )

    # ✅ Remove duplicates
    unique_jobs = []
    seen = set()

    for job in matched_jobs:
        if job[0] not in seen:
            unique_jobs.append(job)
            seen.add(job[0])

    # ✅ Sort by score
    unique_jobs = sorted(unique_jobs, key=lambda x: x[1], reverse=True)

    return unique_jobs[:10]

# ✅ Upload Route
@app.route('/upload', methods=['POST'])
def upload_file():

    if 'resume' not in request.files:
        return "No file uploaded"

    file = request.files['resume']

    if file.filename == '':
        return "No selected file"

    selected_role = request.form.get("role", "all")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)
    skills = extract_skills(resume_text)
    jobs = match_jobs(resume_text, selected_role)

    # ✅ ATS Score
    if jobs:
        ats_score = sum([job[1] for job in jobs]) // len(jobs)
    else:
        ats_score = 0

    # ✅ Graph Data (VERY IMPORTANT)
    job_titles = [job[0] for job in jobs] if jobs else []
    job_scores = [job[1] for job in jobs] if jobs else []

    # ✅ Suggestions
    suggestions = []

    if ats_score < 30:
        suggestions = [
            "Add more relevant skills",
            "Improve resume formatting",
            "Include projects and experience"
        ]

    elif ats_score < 60:
        suggestions = [
            "Add more technical keywords",
            "Improve project descriptions"
        ]

    else:
        suggestions = [
            "Your resume looks strong!"
        ]

    # ✅ FINAL RETURN
    return  render_template(
        "result.html",
        jobs=jobs,
        skills=skills,
        ats_score=ats_score,
        suggestions=suggestions,
        job_titles=job_titles,   # MUST
        job_scores=job_scores    # MUST
    )
# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True)
