import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import PyPDF2
import pandas as pd
from flask import Flask, render_template, request, redirect, send_file

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')
    
# Extract text from Resume 
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
#extract skill fun
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

#adding matching jobs here 
def match_jobs(user_text, selected_role):
    jobs = pd.read_csv("jobs_dataset.csv")

    job_descriptions = jobs['skills'].tolist()
    # Combine resume + jobs
    all_texts = [user_text] + job_descriptions

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
# Compare resume with jobs
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    matched_jobs = []

    for i in range(len(similarity)):
        score = int(similarity[i] * 100)
        job_title =jobs.iloc[i]['job_title'].lower()
        if score > 10:
            if selected_role =="all" or selected_role in job_title:
                matched_jobs.append((jobs.iloc[i]['job_title'],score,jobs.iloc[i]['link']))
    matched_jobs = sorted(matched_jobs, key=lambda x: x[1], reverse=True)

    return matched_jobs[:10]
#Upload fun modify
@app.route('/upload', methods=['POST'])

def upload_file():
    if 'resume' not in request.files:
        return "No file uploaded" 
    file = request.files['resume']
    selected_role= request.form.get("role")
    if file.filename == '':
        return "No selected file"   
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    resume_text = extract_text_from_pdf(filepath) 
    skills = extract_skills(resume_text)
    jobs = match_jobs(resume_text,selected_role)
    return render_template("result.html",jobs=jobs, skills=skills)
    
import os

if __name__=='__main__':
    app.run(debug=True)