# GenAI Regulation Extraction Tool (PoC)

A lightweight Streamlit application that demonstrates how Google Gemini can be
used to transform large regulation PDFs into a structured, machine-readable
representation (JSON or YAML).

---

## Features

* **PDF upload** – drag-and-drop a regulation (up to ~50 MB).
* **Selective page processing** – optionally focus on specific pages for rapid
  iteration.
* **Chunked Gemini calls** – configurable page-per-chunk setting mitigates the
  model's output-length limit.
* **Live streaming UI** – watch the structured output appear in real-time for
  each chunk.
* **Robust parsing & merging** – helper utilities salvage minor formatting
  issues and merge chunk-level structures into a single document.
* **One-click download** – grab the final JSON or YAML file.

---

## Quickstart

1. **Clone the repo** (or download the code).
   ```bash
   git clone <your-fork-url>
   cd reg_analyser  # project root
   ```

2. **Create a virtual environment** (recommended).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**.
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Gemini API key**.
   * Sign up for the Gemini API and grab an API key.
   * Either:
     * Export it as an env var: `export GOOGLE_API_KEY="<YOUR_KEY>"`, **or**
     * Create a `.env` file in the project root:
       ```
       GOOGLE_API_KEY=<YOUR_KEY>
       ```

5. **Run the app**.
   ```bash
   streamlit run app.py
   ```
   The browser tab should open automatically (usually <http://localhost:8501>).

---

## Usage Tips

* **Pages per chunk** – If you encounter errors about incomplete JSON/YAML,
  lower this value. 1-2 pages per chunk is safest, but slower.
* **Page range** – Provide values like `1-5, 10, 12-14` to restrict processing
  during prompt tuning.
* **Output inspection** – Invalid chunks are flagged in red; use the tips shown
  to adjust chunk size or refine prompts.

---

## File Overview

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI and orchestration logic |
| `pdf_parser.py` | Text extraction via PyMuPDF |
| `gemini_client.py` | Thin wrapper around the `google-genai` client |
| `output_utils.py` | Parsing and merging helpers |
| `requirements.txt` | Python dependencies |
| `context/` | Design docs & specifications |

---

## Roadmap

This PoC focuses on demonstrating feasibility. Potential next steps include:

* Smarter section-ID deduplication during merging.
* Better prompts and evaluation against a gold-standard dataset.
* OCR support for scanned PDFs.
* Dockerfile & GitHub Actions CI.

---

## License

MIT — see `LICENSE` (or replace with your chosen license). 