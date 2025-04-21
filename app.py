import streamlit as st
from pdf_parser import extract_text_from_pdf

st.set_page_config(page_title="GenAI Regulation Extraction Tool", layout="centered")

st.title("GenAI Regulation Extraction Tool (PoC)")

# Layout containers
status_placeholder = st.empty()

# File uploader (PDF only)
uploaded_file = st.file_uploader("Upload a regulation PDF", type=["pdf"])

# Output format selection
output_format = st.radio("Select output format", ["JSON", "YAML"])

# Process button
process_clicked = st.button("Extract Requirements")

# Placeholders for preview and download
preview_placeholder = st.empty()
download_placeholder = st.empty()

if process_clicked and uploaded_file is not None:
    with status_placeholder.spinner("Extracting text from PDF..."):
        try:
            pages_text = extract_text_from_pdf(uploaded_file)
            status_placeholder.success(f"Extracted text from {len(pages_text)} pages.")
            # Show preview of first page
            if pages_text:
                preview_placeholder.code(pages_text[0][:2000], language="text")
            else:
                preview_placeholder.info("No text found in PDF.")
        except Exception as e:
            status_placeholder.error(str(e))
            preview_placeholder.empty()
else:
    status_placeholder.empty()
    preview_placeholder.empty()

# (No processing logic yet) 