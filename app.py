import os
import sqlite3
import requests
import PyPDF2
import pandas as pd

from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Flask App Setup
# =========================

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Automatically create resumes folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Database Setup
# =========================

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )
    ''')

    conn.commit()
    conn.close()

init_db()

# =========================
# Web Scraping Jobs
# =========================

def scrape_jobs(selected_role):

    url = "https://realpython.github.io/fake-jobs/"

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        jobs = []

        job_cards = soup.find_all('div', class_='card-content')

        for job in job_cards:

            title = job.find('h2').text.strip()

            score = len(title) % 100

            if selected_role.lower() == "all" or selected_role.lower() in title.lower():

                google_link = (
                    "https://www.google.com/search?q="
                    + title.replace(" ", "+")
                    + "+jobs"
                )

                original_link = job.find('a')['href']

                jobs.append(
                    (
                        title,
                        score,
                        google_link,
                        original_link
                    )
                )

        return jobs[:5]

    except:
        return []

# =========================
# Home Route
# =========================

@app.route('/')
def home():
    return render_template('index.html')

# =========================
# Signup Route
# =========================

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username,email,password) VALUES (?,?,?)",
            (username, email, password)
        )

        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('signup.html')

# =========================
# Login Route
# =========================
@app.route('/login', methods=['GET', 'POST'])
def login():

    error = None

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:
            return redirect('/')

        else:
            error = "❌ Invalid Email or Password"

    return render_template('login.html', error=error)

# =========================
# Extract Text From PDF
# =========================

def extract_text_from_pdf(filepath):

    text = ""

    try:
        with open(filepath, 'rb') as file:

            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

    except:
        return ""

    return text

# =========================
# Extract Skills
# =========================

def extract_skills(text):

    skills_list = [
        'python',
        'java',
        'c++',
        'machine learning',
        'data science',
        'html',
        'css',
        'javascript',
        'sql',
        'react',
        'node',
        'django'
    ]

    found_skills = []

    text = text.lower()

    for skill in skills_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills

# =========================
# Match Jobs
# =========================

def match_jobs(user_text, selected_role):

    try:
        jobs = pd.read_csv("jobs_dataset.csv")

    except:
        return []

    job_descriptions = jobs['skills'].fillna("").tolist()

    all_texts = [user_text] + job_descriptions

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(all_texts)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    ).flatten()

    matched_jobs = []

    for i in range(len(similarity)):

        score = int(similarity[i] * 100)

        job_title = str(jobs.iloc[i]['job_title']).lower()

        if score > 5:

            if (
                selected_role == "all"
                or selected_role.lower() in job_title
            ):

                matched_jobs.append(
                    (
                        jobs.iloc[i]['job_title'],
                        score,
                        jobs.iloc[i]['link']
                    )
                )

    # Remove duplicates
    unique_jobs = []

    seen = set()

    for job in matched_jobs:

        if job[0] not in seen:

            unique_jobs.append(job)

            seen.add(job[0])

    # Sort by score
    unique_jobs = sorted(
        unique_jobs,
        key=lambda x: x[1],
        reverse=True
    )

    # Add extra jobs if less than 5
    if len(unique_jobs) < 5:

        extra_jobs = []

        for i in range(min(5, len(jobs))):

            extra_jobs.append(
                (
                    jobs.iloc[i]['job_title'],
                    int(similarity[i] * 100),
                    jobs.iloc[i]['link']
                )
            )

        return unique_jobs + extra_jobs

    return unique_jobs[:10]

# =========================
# Upload Resume Route

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'GET':
        return redirect('/')

    if 'resume' not in request.files:
        return "No file uploaded"

    file = request.files['resume']

    if file.filename == '':
        return "No selected file"

    selected_role = request.form.get("role", "all")

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)

    skills = extract_skills(resume_text)

    jobs = match_jobs(resume_text, selected_role)

    scraped_jobs = scrape_jobs(selected_role)

    # ATS Score
    if jobs:
        ats_score = sum([job[1] for job in jobs]) // len(jobs)
    else:
        ats_score = 0

    # Graph Data
    job_titles = [job[0] for job in jobs] if jobs else []

    job_scores = [job[1] for job in jobs] if jobs else []

    # Suggestions
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

    # Remove duplicate suggestions
    suggestions = list(set(suggestions))

    return render_template(
        "result.html",
        jobs=jobs,
        scraped_jobs=scraped_jobs,
        skills=skills,
        ats_score=ats_score,
        suggestions=suggestions,
        job_titles=job_titles,
        job_scores=job_scores
    )
    
    
# =========================
# Recruiter Route
# =========================

@app.route('/recruiter')
def recruiter():
    return render_template('recruiter.html')

# =========================
# Run App
# =========================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )