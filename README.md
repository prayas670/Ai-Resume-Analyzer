# 🚀 SCANLINE — AI Resume Analyzer

<p align="center">
  <b>An intelligent Resume Analyzer that evaluates ATS compatibility, analyzes resume quality, matches resumes with job descriptions, and provides AI-powered improvement suggestions using Machine Learning and NLP.</b>
</p>

---

## ✨ Features

- 📄 Upload resumes in **PDF, DOCX, and TXT**
- 🤖 AI-powered Resume Analysis
- 📊 ATS Compatibility Score
- 🎯 Job Description Matching
- 🧠 Sentence-BERT Semantic Similarity
- 🔍 Skill Extraction using FlashText & spaCy
- 📈 Resume Completeness Analysis
- 📑 Section-wise Resume Scoring
- 🌡️ Resume Heatmap Visualization
- 📊 Interactive Dashboard with Charts
- ⭐ Candidate Level Detection
- 💼 Resume Domain Detection
- ⏳ Experience Detection
- ✍️ Bullet Point Rewrite Suggestions
- 🚀 Project Enhancement Suggestions
- 📋 STAR Resume Analysis
- 📄 Downloadable PDF Report
- ⚡ Optional AI Feedback using Groq Llama 3

---

## 🧠 AI & Machine Learning

SCANLINE combines traditional NLP with Machine Learning models to deliver accurate resume insights.

| Model | Purpose |
|--------|----------|
| Sentence-BERT | Semantic Resume ↔ JD Matching |
| XGBoost | ATS Score Prediction |
| spaCy | Entity & Skill Extraction |
| TF-IDF + Cosine Similarity | Keyword Matching |
| FlashText | Fast Skill Detection |
| Rule-Based NLP | Resume Evaluation |

> **Works completely offline** without any paid API. AI feedback is optional using a free Groq API key.

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning & NLP
- Scikit-Learn
- Sentence Transformers
- spaCy
- XGBoost
- FlashText
- TF-IDF
- Cosine Similarity

### Utilities
- PDFPlumber
- python-docx
- Joblib
- Requests

---

## 📂 Project Structure

```text
resume-analyzer/
│── app.py
│── analyzer.py
│── ml_models.py
│── train_ats_model.py
│── ats_model.joblib
│── requirements.txt
│── index.html
│── style.css
│── script.js
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/SCANLINE.git

cd SCANLINE

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

---

## 📊 Analysis Includes

- ATS Score
- Resume Score
- Resume Completeness
- Skill Coverage
- Resume Heatmap
- Section-wise Analysis
- Experience Detection
- Candidate Level
- Domain Prediction
- JD Similarity Score
- STAR Analysis
- Missing Keywords
- Resume Suggestions
- AI Feedback (Optional)

---

## 🚀 Future Improvements

- Resume Ranking
- Resume Comparison
- AI Cover Letter Generator
- Resume Version History
- User Authentication
- Recruiter Dashboard

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request


---

## 👨‍💻 Author

**Prayas Gupta**

- 🎓 B.Tech Artificial Intelligence & Machine Learning
- 💻 Aspiring AI Engineer
- 🌟 Passionate about AI, Machine Learning & Full-Stack Development

---

<p align="center">
⭐ If you found this project helpful, don't forget to star the repository!
</p>
