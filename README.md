# 🚀 SCANLINE — AI Resume Analyzer

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge\&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-AI-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered Resume Analyzer that evaluates ATS compatibility, analyzes resume quality, performs semantic Job Description matching, and provides intelligent improvement suggestions using Machine Learning and NLP.**

</p>

---

## 📌 Overview

SCANLINE is a modern AI Resume Analyzer built using **Flask, Machine Learning, NLP, and JavaScript**. It analyzes resumes, predicts ATS performance, extracts important information, compares resumes with job descriptions, and generates actionable recommendations to improve interview chances.

Unlike traditional keyword checkers, SCANLINE combines **rule-based NLP**, **Sentence-BERT**, **spaCy**, **TF-IDF**, and **XGBoost** to provide more accurate resume analysis.

---

## ✨ Features

* 📄 Resume Upload (PDF, DOCX, TXT)
* 🤖 AI-powered Resume Analysis
* 📊 ATS Compatibility Score
* 🎯 Job Description Matching
* 🧠 Semantic Similarity using Sentence-BERT
* 🔍 Skill Extraction
* 📌 Missing Skills Detection
* 📈 Resume Quality Score
* 📚 Experience Detection
* 🎓 Education Extraction
* 🏆 Certification Detection
* 💼 Candidate Level Prediction
* 🌐 Domain Detection
* 📋 Resume Completeness Analysis
* 🔥 ATS Risk Analysis
* ✍️ Bullet Point Rewrite Suggestions
* 🚀 Project Enhancement Suggestions
* 📊 Dashboard with Interactive Charts
* 📥 PDF Report Download
* 🎨 Modern Responsive UI

---

# 🛠 Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js
* HTML2PDF

### Backend

* Flask
* Flask-CORS

### Machine Learning & NLP

* Scikit-Learn
* Sentence-BERT
* spaCy
* XGBoost
* FlashText
* TF-IDF
* Cosine Similarity
* Regex

---

# 🧠 Machine Learning Models

| Model             | Purpose                                                           |
| ----------------- | ----------------------------------------------------------------- |
| Sentence-BERT     | Semantic Resume ↔ Job Description Matching                        |
| spaCy             | Entity Extraction (Skills, Education, Experience, Certifications) |
| XGBoost           | ATS Parseability Prediction                                       |
| TF-IDF            | Keyword Similarity                                                |
| Cosine Similarity | Resume Matching                                                   |
| FlashText         | Fast Skill Detection                                              |

---

# 📂 Project Structure

```text
SCANLINE/
│
├── app.py
├── analyzer.py
├── ml_models.py
├── train_ats_model.py
├── ats_model.joblib
├── requirements.txt
├── index.html
├── style.css
├── script.js
└── README.md
```

---

# ⚙ Installation

```bash
git clone https://github.com/yourusername/SCANLINE.git

cd SCANLINE

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

# ▶ Running the Project

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# 🔄 Analysis Workflow

```text
Resume Upload
      │
      ▼
Text Extraction
      │
      ▼
Resume Parsing
      │
      ▼
Skill Extraction
      │
      ▼
ATS Analysis
      │
      ▼
Semantic JD Matching
      │
      ▼
Resume Quality Analysis
      │
      ▼
AI Suggestions
      │
      ▼
Interactive Dashboard
```

---

# 📊 Analysis Modules

* Overall Resume Score
* ATS Score
* Resume Completeness
* Resume Structure Analysis
* Content Quality Score
* Skill Coverage
* Experience Detection
* Candidate Level
* Domain Prediction
* Section-wise Scoring
* Missing Keywords
* Missing Skills
* ATS Risk Detection
* Resume Suggestions
* Project Suggestions
* Bullet Point Improvements

---

# 🎯 Supported Resume Formats

* PDF
* DOCX
* TXT

---

# 📈 Dashboard Features

* Animated Score Gauge
* ATS Risk Card
* Resume Heatmap
* Skill Chips
* Candidate Profile
* Interactive Charts
* Section Scores
* Improvement Suggestions
* PDF Report Export

---

# 📡 API Endpoint

### Analyze Resume

```http
POST /api/analyze
```

### Parameters

* Resume File
* Target Role *(Optional)*
* Job Description *(Optional)*

Returns a detailed JSON report containing ATS score, skills, suggestions, JD similarity, candidate profile, and more.

---

# 🚀 Future Enhancements

* Resume Ranking
* Cover Letter Generator
* AI Resume Builder
* LinkedIn Profile Analysis
* Portfolio Evaluation
* Multi-language Resume Support
* Recruiter Dashboard
* Authentication System
* Resume Version History
* Cloud Deployment

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Prayas Gupta**

🎓 B.Tech – Artificial Intelligence & Machine Learning

💼 Passionate about AI, Machine Learning, NLP, and Full-Stack Development.

If you found this project helpful, don't forget to ⭐ the repository.
