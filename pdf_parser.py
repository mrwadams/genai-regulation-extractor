"""PDF parsing utilities for the GenAI Regulation Extraction Tool (PoC).

Currently contains :func:`extract_text_from_pdf`, which extracts plain text from
an uploaded PDF file (Streamlit uploader) using PyMuPDF. Additional parsing
functions or more advanced OCR capabilities can be added here in future
iterations.
"""

import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(uploaded_file) -> List[str]:
    """
    Extracts text from each page of a PDF file.
    Args:
        uploaded_file: A file-like object (from Streamlit uploader)
    Returns:
        List of strings, one per page.
    Raises:
        RuntimeError: If PDF cannot be parsed.
    """
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        return pages_text
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}") 