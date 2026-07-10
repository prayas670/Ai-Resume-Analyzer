import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from analyzer import extract_text, score_resume, get_ai_feedback, analyze_ats_risk, get_bullet_rewrites, get_project_enhancements

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "ai_feedback_enabled": bool(os.environ.get("GROQ_API_KEY"))})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded. Use form field 'resume'."}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file."}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Max size is 5 MB."}), 400

    jd_text = request.form.get("job_description", "").strip()
    want_ai_feedback = request.form.get("ai_feedback", "false").lower() == "true"

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path, file.filename)
        if not text or not text.strip():
            return jsonify({"error": "Could not extract any text from this file. Try a different format."}), 422

        ats_risk = analyze_ats_risk(tmp_path, file.filename, text)
        analysis = score_resume(text, jd_text if jd_text else None, ats_risk=ats_risk)
        analysis["ats_risk"] = ats_risk

        # Bullet point rewrite suggestions — uses the AI reviewer for higher
        # quality rewrites when ai_feedback is requested and a key is set,
        # otherwise falls back to the local rule-based rewriter.
        analysis["bullet_rewrites"] = get_bullet_rewrites(text, use_ai=want_ai_feedback)
        analysis["project_enhancements"] = get_project_enhancements(text, use_ai=want_ai_feedback)

        if want_ai_feedback:
            analysis["ai_feedback"] = get_ai_feedback(text, jd_text, analysis)
        else:
            analysis["ai_feedback"] = None

        return jsonify(analysis)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
