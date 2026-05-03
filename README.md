# AI Resume Screener

An AI-powered resume screening web app built with Flask and Claude (Anthropic). Upload multiple CVs, specify job criteria, and get a ranked table of candidates with match scores and summaries — all in your browser.

## Features

- Bulk upload PDF and DOCX resumes (drag & drop or file picker)
- Define job title, required skills, and minimum years of experience
- Claude scores each resume 0–100 and explains why
- Results displayed in a sortable ranked table
- Export results to a colour-coded Excel spreadsheet
- No database, no login — everything processed in memory

## Requirements

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

### 1. Clone / navigate to the project folder

```bash
cd "AI Resume Screener"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS / Linux
# set ANTHROPIC_API_KEY=sk-ant-...      # Windows CMD
# $env:ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell
```

### 5. Run the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

## Usage

1. Fill in **Job Title**, **Required Skills** (comma-separated), and **Min. Years of Experience**.
2. Drag & drop or browse to upload one or more PDF/DOCX resume files.
3. Click **Screen Resumes**. Claude will read and score each CV.
4. View the ranked results table. Green = strong match (80+), Yellow = partial (60–79), Red = weak (<60).
5. Click **Export to Excel** to download a formatted `.xlsx` file.

## Project Structure

```
AI Resume Screener/
├── app.py              # Flask backend — routes, text extraction, Claude API calls
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/
│   └── index.html      # Single-page UI
└── static/
    └── style.css       # Styles
```

## Notes

- Resumes are processed entirely in memory; nothing is written to disk or stored.
- PDF text extraction uses `pdfplumber`. Scanned/image-only PDFs will return empty text and be skipped.
- Only the first ~8,000 characters of each CV are sent to Claude to keep token costs low.
- The app uses the `claude-sonnet-4-6` model by default.
