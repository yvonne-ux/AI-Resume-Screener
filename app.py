import os
import io
import json
import re
import anthropic
import pdfplumber
import openpyxl
from docx import Document
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    text = text.replace("\u00a0", "\n").replace("\u2028", "\n")
    text = text.replace("\u00ad", " ").replace("\ufeff", "")
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text.strip()


def extract_text_from_pdf(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(clean_text(page_text))
    return "\n".join(pages)


def extract_text_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(paragraphs))


def extract_text(file_bytes, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("doc", "docx"):
        return extract_text_from_docx(file_bytes)
    return ""


def analyze_jd_with_claude(job_title, job_description):
    client = anthropic.Anthropic()

    prompt = f"""You are an expert recruiter. Analyze this job description and extract what a junior recruiter needs to know to screen candidates effectively.

Job Title: {job_title}
Job Description:
{job_description[:6000]}

Respond with ONLY a valid JSON object (no markdown, no explanation):
{{
  "role_summary": "2-3 sentence plain-English explanation of what this role does and what kind of person would succeed in it",
  "must_have": ["Critical requirement 1", "Critical requirement 2", "Critical requirement 3"],
  "nice_to_have": ["Good-to-have 1", "Good-to-have 2"],
  "watch_out_for": ["Common mismatch or red flag to watch for 1", "Red flag 2"]
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def score_resume_with_claude(cv_text, job_title, job_description):
    client = anthropic.Anthropic()

    prompt = f"""You are an expert recruiter. Evaluate this CV against the job requirements below.

Job Title: {job_title}
Job Description:
{job_description}

CV:
---
{cv_text[:8000]}
---

Respond with ONLY a valid JSON object (no markdown, no code fences, no explanation):
{{
  "candidate_name": "Full name of the candidate, or 'Unknown'",
  "score": <integer 0-100>,
  "suitable": <true if this candidate genuinely meets the core requirements and is worth presenting to the client, false otherwise>,
  "strengths": ["Specific strength tied to a JD requirement", "Strength 2", "Strength 3"],
  "gaps": ["Key gap or concern 1", "Gap 2"],
  "key_skills_found": ["skill1", "skill2", "skill3"],
  "years_experience": <number or null if unclear>,
  "email_bullets": ["Specific, factual highlight 1 a recruiter would tell a client about this candidate for this role", "Highlight 2", "Highlight 3"]
}}

Instructions:
- strengths: 2-4 specific reasons this person meets the JD requirements, referencing actual CV details
- gaps: 1-3 notable gaps; use an empty array if there are none
- email_bullets: 3-4 compelling, specific points drawn from the CV that directly address this JD

Scoring guide: 90-100 exceptional, 70-89 strong, 50-69 partial, 30-49 weak, 0-29 poor."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze-jd", methods=["POST"])
def analyze_jd():
    data = request.get_json()
    job_title = (data.get("job_title") or "").strip()
    job_description = (data.get("job_description") or "").strip()

    if not job_title or not job_description:
        return jsonify({"error": "Job title and job description are required."}), 400

    try:
        result = analyze_jd_with_claude(job_title, job_description)
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JD analysis response."}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"API error — {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract-jd", methods=["POST"])
def extract_jd():
    """Extract text from an uploaded JD file (PDF or DOCX)."""
    file = request.files.get("jd_file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file type. Use PDF or DOCX."}), 400

    try:
        file_bytes = file.read()
        text = extract_text(file_bytes, filename)
        if not text:
            return jsonify({"error": "Could not extract text from file."}), 400
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/screen", methods=["POST"])
def screen():
    job_title = request.form.get("job_title", "").strip()
    job_description = request.form.get("job_description", "").strip()

    if not job_title or not job_description:
        return jsonify({"error": "Job title and job description are required."}), 400

    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Please upload at least one resume."}), 400

    results = []
    errors = []

    for file in files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            errors.append(f"{filename}: unsupported file type (use PDF or DOCX).")
            continue

        try:
            file_bytes = file.read()
            cv_text = extract_text(file_bytes, filename)

            if not cv_text:
                errors.append(f"{filename}: could not extract text from file.")
                continue

            result = score_resume_with_claude(cv_text, job_title, job_description)
            result["filename"] = filename
            results.append(result)

        except json.JSONDecodeError:
            errors.append(f"{filename}: Claude returned an unexpected response format.")
        except anthropic.APIError as e:
            errors.append(f"{filename}: API error — {str(e)}")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return jsonify({"results": results, "errors": errors})


@app.route("/export", methods=["POST"])
def export():
    data = request.get_json()
    results = data.get("results", [])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Screening Results"

    headers = ["Rank", "Candidate Name", "Score", "Verdict", "Suitable", "Key Skills", "Strengths", "Gaps", "Email Bullets"]
    ws.append(headers)

    header_font = openpyxl.styles.Font(bold=True, color="FFFFFF")
    header_fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="1D9E75")
    header_alignment = openpyxl.styles.Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_num, _ in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment

    def get_verdict(score):
        if score >= 75: return "Hire"
        if score >= 50: return "Maybe"
        return "Pass"

    for rank, r in enumerate(results, 1):
        score = r.get("score", 0)
        verdict = get_verdict(score)
        suitable = "Yes" if r.get("suitable") else "No"
        skills = ", ".join(r.get("key_skills_found", []))
        strengths = "\n".join(f"• {s}" for s in r.get("strengths", []))
        gaps = "\n".join(f"• {g}" for g in r.get("gaps", []))
        email_bullets = "\n".join(f"• {b}" for b in r.get("email_bullets", []))

        row = [rank, r.get("candidate_name", "Unknown"), score, verdict, suitable, skills, strengths, gaps, email_bullets]
        ws.append(row)

        score_cell = ws.cell(row=rank + 1, column=3)
        if score >= 90:
            score_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="16A34A")
        elif score >= 70:
            score_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="DCFCE7")
        elif score >= 50:
            score_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="FEF9C3")
        elif score >= 30:
            score_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="FED7AA")
        else:
            score_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="FEE2E2")

        verdict_cell = ws.cell(row=rank + 1, column=4)
        if verdict == "Hire":
            verdict_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="DCFCE7")
            verdict_cell.font = openpyxl.styles.Font(bold=True, color="15803D")
        elif verdict == "Maybe":
            verdict_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="FEF9C3")
            verdict_cell.font = openpyxl.styles.Font(bold=True, color="92400E")
        else:
            verdict_cell.fill = openpyxl.styles.PatternFill(fill_type="solid", fgColor="FEE2E2")
            verdict_cell.font = openpyxl.styles.Font(bold=True, color="B91C1C")

        for col in [7, 8, 9]:
            ws.cell(row=rank + 1, column=col).alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")

    col_widths = [6, 22, 8, 10, 10, 32, 40, 32, 50]
    for i, width in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = width

    for row_num in range(2, len(results) + 2):
        ws.row_dimensions[row_num].height = 80

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="resume_screening_results.xlsx",
    )


if __name__ == "__main__":
    app.run(debug=True)
