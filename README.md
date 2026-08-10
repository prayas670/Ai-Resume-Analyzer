# 🚀 SCANLINE — AI Resume Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-ATS%20Scoring-green)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy)

**SCANLINE** is a modern **AI-powered Resume Analyzer** that evaluates resumes for **ATS compatibility, content quality, structure, and skill coverage**.

Built with a **Flask backend** and a **responsive HTML/CSS/JavaScript frontend**, the entire scoring pipeline runs **locally** using **Scikit-Learn, spaCy, Sentence-BERT, and XGBoost** — **no API keys, subscriptions, or external AI services required**.

---

## ✨ Features

* 📄 **Upload PDF, DOCX, or TXT resumes**
* 🤖 **AI-powered ATS scoring**
* 📊 **Section-wise score breakdown**
* 🎯 **Job Description (JD) similarity matching**
* 🧠 **Semantic matching with Sentence-BERT**
* 🏷️ **Skill extraction and entity recognition with spaCy**
* ⚡ **ML-based ATS pass prediction using XGBoost**
* 🧾 **Resume completeness checklist**
* 🔍 **Highlighted weak sentences and matched keywords**
* ♻️ **Duplicate content detection**
* ✍️ **Automatic bullet-point rewrite suggestions**
* 🎨 **Modern dashboard UI with charts and visual analytics**

---

## 🖼️ Demo Preview

> Upload a resume → Run the scan → Get a detailed ATS-style report with actionable improvements.

The report includes:

* Overall ATS score
* Structure score
* Content quality score
* Skills coverage score
* JD similarity score
* ATS risk level
* Extracted profile information
* Improvement suggestions

---

## 🏗️ Project Structure

```text
resume-analyzer/
├── app.py                  # Flask server + API routes
├── analyzer.py             # Resume parsing, scoring & suggestions
├── ml_models.py            # SBERT, spaCy & XGBoost integrations
├── train_ats_model.py      # Offline ATS model trainer
├── requirements.txt
├── index.html              # Frontend UI
├── style.css               # Styling and dashboard design
├── script.js               # Frontend logic & API communication
└── README.md
```

> `app.py` serves the frontend directly, so a **single Flask process runs the entire application**.

---

## 🧠 Machine Learning Pipeline

SCANLINE combines **rule-based NLP** with **three real ML models** for more accurate analysis.

| Model                                  | Purpose                                                    |
| -------------------------------------- | ---------------------------------------------------------- |
| **Sentence-BERT** (`all-MiniLM-L6-v2`) | Semantic resume ↔ job description similarity               |
| **spaCy** (`en_core_web_sm`)           | Extracts education, experience, certifications, and skills |
| **XGBoost**                            | Predicts ATS parseability score (0–100)                    |

All models **degrade gracefully** if unavailable, ensuring the application still works with the rule-based engine.

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/scanline-ai-resume-analyzer.git
cd scanline-ai-resume-analyzer
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate it

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\\Scripts\\activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

> **Important:** Do **not** open `index.html` directly using `file://`. Always run the Flask server first.

---

## 📊 How Scoring Works

| Component           | What It Evaluates                                      |
| ------------------- | ------------------------------------------------------ |
| **Structure Score** | Resume sections, contact info, formatting, length      |
| **Content Score**   | Action verbs, quantified achievements, weak phrases    |
| **Skills Score**    | Coverage of technical & professional skills            |
| **JD Match**        | TF-IDF + semantic similarity against a job description |

The **overall score** is a weighted blend of these components.

---

## 📌 Advanced Analysis Features

### ✅ Section-wise Resume Score

Each section receives its own **0–100 score**:

* Contact
* Summary
* Experience
* Education
* Skills
* Projects

---

### 📋 Resume Completeness Score

Checks whether the resume includes:

* Email
* Phone number
* LinkedIn profile
* Experience section
* Education section
* Skills section
* Quantified achievements

---

### ⚠️ ATS Risk Analysis

Detects formatting patterns that commonly break ATS parsers:

* Tables used for layout
* Embedded images/logos
* Multi-column layouts
* Non-standard bullet symbols
* Missing standard section headers

Returns **Low / Medium / High** risk with improvement tips.

---

### 🛡️ Personal Info & Bias-Risk Check

Flags potentially problematic demographic details such as:

* Photo references
* Date of birth / age
* Marital status
* Nationality
* Gender
* Religion

This feature is **advisory only** and does **not affect the ATS score**.

---

### 🔍 Highlighted Resume View

Visual feedback directly inside the extracted resume text:

* 🟥 Weak sentences and buzzwords
* 🟩 Matched skills and keywords
* 🟨 Missing JD keywords listed separately

---

### ♻️ Duplicate Content Detection

Detects:

* Repeated skills
* Duplicate sentences
* Near-identical bullet points across projects or jobs
* Possible keyword stuffing

---

### ✍️ Bullet Point Rewriter

Automatically rewrites weak bullets by:

* Adding stronger action verbs
* Suggesting measurable outcomes
* Improving project descriptions with technology context

---

**Form fields**

| Field             | Type | Required |
| ----------------- | ---- | -------- |
| `resume`          | File | ✅        |
| `job_description` | Text | ❌        |
| `target_role`     | Text | ❌        |

---

## 📄 Supported File Types

* **PDF**
* **DOCX**
* **TXT**

**Maximum size:** `5 MB`

---

## 🛠️ Retraining the ATS Model

If you want to retrain the XGBoost ATS model:

```bash
python train_ats_model.py
```

This generates a fresh `ats_model.joblib` file using the synthetic ATS training data defined in `ml_models.py`.

---

## 🔧 Tech Stack

### Backend

* **Python 3.9+**
* **Flask**
* **Flask-CORS**
* **Scikit-Learn**
* **XGBoost**
* **spaCy**
* **Sentence-Transformers**
* **FlashText**
* **pdfplumber**
* **python-docx**

### Frontend

* **HTML5**
* **CSS3**
* **Vanilla JavaScript**
* **Chart.js**

---

## 📈 Future Improvements

* 🌍 Multi-language resume support
* 📑 Resume version comparison
* ☁️ Optional cloud deployment 
* 🧪 Real-world ATS outcome training data 

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## ⭐ Show Your Support

If you found this project useful, please consider giving it a **⭐ star** on GitHub — it helps others discover the project and motivates further development!

---
