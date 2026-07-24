# SCANLINE — AI Resume Analyzer

> **An intelligent, locally powered AI Resume Analyzer that evaluates ATS compatibility, resume quality, skill coverage, and job-description alignment using Machine Learning and NLP — with zero API keys required.**

---

# 🚀 Overview

**SCANLINE** is a modern AI-powered Resume Analyzer built with **Flask**, **HTML**, **CSS**, **JavaScript**, and **Machine Learning**.

It analyzes resumes for:

- ATS Compatibility
- Resume Quality
- Section-wise Performance
- Skills Coverage
- Resume Completeness
- Content Strength
- Job Description Matching
- ATS Formatting Risks
- Bullet Point Improvements

Unlike many resume analyzers, **SCANLINE runs entirely offline**. Every prediction is generated locally using **Sentence-BERT**, **spaCy**, **XGBoost**, **TF-IDF**, and rule-based NLP—**no APIs, subscriptions, or internet connection required.**

---

# ✨ Features

## 📊 Resume Analysis

- Overall Resume Score
- ATS Compatibility Score
- Resume Completeness Score
- Structure Quality Analysis
- Content Quality Analysis
- Skills Coverage Analysis

## 🎯 Job Description Matching

- Semantic Resume ↔ Job Description Similarity
- Required Keyword Detection
- Missing Skills Identification
- Preferred Skills Matching
- TF-IDF + Sentence-BERT Similarity

## 📄 Section-wise Evaluation

Individual scores for:

- Contact Information
- Professional Summary
- Experience
- Education
- Skills
- Projects

## ⚠️ ATS Risk Detection

Detects formatting issues including:

- Tables
- Images & Logos
- Multi-column Layouts
- Non-standard Bullets
- Missing Standard Sections

Provides:

- ATS Risk Level
- ML ATS Pass Score
- Detailed Issue Breakdown
- Actionable Suggestions

## 💡 Smart Resume Suggestions

- Weak Bullet Detection
- Strong Action Verb Suggestions
- Quantified Achievement Recommendations
- Resume Improvement Tips

## 📂 Supported File Types

- PDF
- DOCX
- TXT

Maximum upload size: **5 MB**

---

# 🧠 Machine Learning Stack

SCANLINE combines traditional NLP with multiple Machine Learning models. Every model automatically falls back to the rule-based engine if unavailable, ensuring the application remains fully functional.

| Model | Purpose |
|--------|---------|
| **Sentence-BERT (all-MiniLM-L6-v2)** | Semantic Resume ↔ Job Description Matching |
| **spaCy (NER + PhraseMatcher)** | Education, Experience, Certification & Skill Extraction |
| **XGBoost** | ML-based ATS Pass Score Prediction |
| **TF-IDF + Cosine Similarity** | Keyword Matching & Similarity Calculation |
| **Regex + Rule Engine** | Resume Parsing & Quality Analysis |

---

# 🏗️ Project Structure

```text
resume-analyzer/
│
├── app.py                 # Flask server & API routes
├── analyzer.py            # Resume parsing and scoring engine
├── ml_models.py           # Machine Learning integrations
├── train_ats_model.py     # ATS model training script
├── ats_model.joblib       # Pre-trained XGBoost model
├── requirements.txt
├── index.html
├── style.css
└── script.js
```

The frontend is served directly by **Flask**, allowing the complete application to run from a single server without requiring React, Node.js, or any separate frontend framework.

---

# ⚙️ Scoring System

The final resume score is calculated using multiple evaluation metrics.

| Component | What It Evaluates |
|------------|-------------------|
| **Structure Score** | Resume organization, formatting, and required sections |
| **Content Score** | Action verbs, quantified achievements, writing quality |
| **Skills Score** | Technical and soft skill coverage |
| **JD Match Score** | Semantic similarity with the target job description |
| **ATS Score** | Machine Learning + heuristic ATS compatibility |

When a Job Description is provided, the **JD Match Score** receives higher weight in the final score.

---

# 📈 Resume Quality Metrics

SCANLINE evaluates resumes using multiple independent scoring systems:

- Overall Resume Score
- ATS Compatibility Score
- Resume Completeness Score
- Section-wise Scores
- Skills Coverage Score
- Content Quality Score
- Structure Score
- Job Description Match Score

This multi-dimensional evaluation helps users identify exactly which areas require improvement instead of relying on a single overall percentage.

---

# 🤖 ATS Analysis

The ATS engine detects formatting patterns commonly rejected by Applicant Tracking Systems.

It analyzes:

- Tables
- Images
- Logos
- Multi-column Layouts
- Decorative Icons
- Non-standard Section Headers
- Special Bullet Symbols

The system provides:

- ATS Risk Level
- ML Predicted ATS Pass Score
- Detailed Issue Breakdown
- Actionable Recommendations

---

# 🎯 Job Description Matching

When a target Job Description is provided, SCANLINE performs semantic comparison between the resume and the job requirements.

Outputs include:

- Resume ↔ Job Description Similarity Score
- Matched Keywords
- Missing Keywords
- Required Skills
- Preferred Skills
- Personalized Improvement Suggestions

Sentence-BERT is used for semantic matching, while TF-IDF acts as a fallback whenever the model is unavailable.

---

# ✍️ Resume Rewrite Assistance

SCANLINE identifies weak bullet points within the **Experience** and **Projects** sections and recommends stronger alternatives by:

- Replacing weak action verbs
- Encouraging quantified achievements
- Improving impact-oriented writing
- Enhancing readability

---

# 🛠️ Installation

Clone the repository and install the required dependencies.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

The repository already includes:

- Pre-trained XGBoost Model
- spaCy English Model
- Required Machine Learning Dependencies

No additional downloads or API keys are required.

---

# ▶️ Running the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

> **Note:** Do not open `index.html` directly. The application must be served through Flask so it can access the backend API endpoints.

---

# 🔄 Retraining the ATS Model

To retrain the ATS prediction model:

```bash
python train_ats_model.py
```

This generates a new:

```text
ats_model.joblib
```

The server loads this pre-trained model during startup and never performs training while handling requests.

---

# 🛠️ Tech Stack

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask

### Machine Learning

- scikit-learn
- Sentence-BERT
- spaCy
- XGBoost
- TF-IDF
- Cosine Similarity

### NLP

- Regex
- PhraseMatcher
- Named Entity Recognition (NER)

---

# 🌟 Why SCANLINE?

- ✅ Fully Offline
- ✅ No API Keys Required
- ✅ Multiple Machine Learning Models
- ✅ ATS Risk Detection
- ✅ Semantic Job Description Matching
- ✅ Section-wise Resume Analysis
- ✅ Resume Completeness Evaluation
- ✅ Bullet Point Enhancement
- ✅ Lightweight & Modern Architecture

---
