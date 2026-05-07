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
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB to support 20+ resumes

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
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""You are a senior healthcare/professional services recruiter with 15 years of experience. Evaluate this CV fairly and accurately against the job requirements below.

Job Title: {job_title}
Job Description:
{job_description}

CV:
---
{cv_text[:8000]}
---

IMPORTANT EVALUATION RULES:
1. TRANSFERABLE EXPERIENCE: Give full credit for transferable skills. A candidate with project management in a related government or healthcare-adjacent body (e.g. HPB, MOH, restructured hospitals) DOES have relevant healthcare operations experience. Do not penalise for sector variation if the core skills match.
2. OVERQUALIFIED: If the candidate's current title, seniority, or salary range appears significantly above the role being hired (e.g. Director applying for Executive role), set "overqualified" to true. This is a risk flag, not automatic rejection.
3. SCORING: Base score on actual evidence in the CV. Do not penalise for information that is simply not mentioned — absence of evidence is not evidence of absence.
4. SUITABILITY: Set suitable=true if the candidate has the core competencies required, even if some nice-to-haves are missing.

Respond with ONLY a valid JSON object (no markdown, no code fences, no explanation):
{{
  "candidate_name": "Full name of the candidate, or 'Unknown'",
  "score": <integer 0-100>,
  "suitable": <true or false>,
  "overqualified": <true if candidate seniority is clearly above the role, false otherwise>,
  "strengths": ["Full sentence: specific strength tied to a JD requirement with evidence from CV", "Strength 2", "Strength 3"],
  "gaps": ["Full sentence: specific gap or concern with explanation", "Gap 2"],
  "key_skills_found": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "years_experience": <number or null if unclear>,
  "requirements_assessment": [
    {{"requirement": "Full requirement text from JD", "met": true, "evidence": "Brief evidence from CV or reason not met"}},
    {{"requirement": "Requirement 2", "met": true, "evidence": "..."}},
    {{"requirement": "Requirement 3", "met": false, "evidence": "Not evidenced in CV"}}
  ],
  "email_bullets": ["Specific, factual highlight 1 a recruiter would tell a client about this candidate", "Highlight 2", "Highlight 3"]
}}

Instructions:
- strengths and gaps: write FULL sentences, no truncation, no ellipsis
- requirements_assessment: extract the 4-6 most important requirements from the JD and assess each one
- email_bullets: neutral, factual highlights — do NOT recommend for or against, just present facts
- If overqualified=true, include it as a gap: "Candidate's current seniority may be above this role — worth discussing expectations"

Scoring guide: 90-100 exceptional, 70-89 strong, 50-69 partial match (still worth considering), 30-49 weak, 0-29 poor."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


@app.route("/interview-questions", methods=["POST"])
def interview_questions():
    data = request.get_json()
    job_title = (data.get("job_title") or "").strip()
    job_description = (data.get("job_description") or "").strip()
    candidate_name = data.get("candidate_name", "the candidate")
    strengths = data.get("strengths", [])
    gaps = data.get("gaps", [])
    key_skills = data.get("key_skills", [])

    if not job_title or not job_description:
        return jsonify({"error": "Job title and description are required."}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set."}), 500

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are a senior recruiter preparing a phone screening call. Generate 5 targeted interview questions for this specific candidate and role.

Job Title: {job_title}
Job Description:
{job_description[:3000]}

Candidate: {candidate_name}
Their strengths: {', '.join(strengths[:3])}
Their gaps: {', '.join(gaps[:3])}
Key skills found: {', '.join(key_skills[:5])}

Generate 5 role-specific questions that:
1. Probe the candidate's actual experience relevant to THIS job
2. Dig into any gaps or areas needing verification
3. Are open-ended and encourage detailed answers
4. Are appropriate for a phone/video screening call (not a deep technical interview)

Respond with ONLY a valid JSON object (no markdown, no explanation):
{{
  "questions": [
    {{
      "category": "Short category label e.g. 'Leadership' or 'Technical' or 'Industry knowledge'",
      "question": "Full question text",
      "why": "One sentence: why this question is important for this role/candidate"
    }}
  ]
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/debug-env")
def debug_env():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return jsonify({
        "key_found": bool(key),
        "key_preview": key[:12] + "..." if key else "NOT SET",
        "env_vars": [k for k in os.environ.keys()]
    })


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
