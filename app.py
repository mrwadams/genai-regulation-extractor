import streamlit as st
from pdf_parser import extract_text_from_pdf
from gemini_client import get_structured_data_from_llm, stream_structured_data_from_llm
import os
from dotenv import load_dotenv

st.set_page_config(page_title="GenAI Regulation Extraction Tool", layout="centered")

st.title("GenAI Regulation Extraction Tool (PoC)")

# Layout containers
status_placeholder = st.empty()

# File uploader (PDF only)
uploaded_file = st.file_uploader("Upload a regulation PDF", type=["pdf"])

# Output format selection
output_format = st.radio("Select output format", ["JSON", "YAML"])

# Utility to get API key from user input, env, or .env
load_dotenv()
def get_api_key(user_input: str) -> str:
    if user_input:
        return user_input
    env_key = os.getenv("GOOGLE_API_KEY")
    return env_key or ""

# Add API key input (pre-fill if found)
default_api_key = os.getenv("GOOGLE_API_KEY")
api_key = st.text_input("Enter your Gemini API key", type="password", value=default_api_key or "")
api_key = get_api_key(api_key)

# Test feature: Page range selection
page_range = st.text_input("(Optional) Enter page range to process (e.g. 1-3,5,7-8):", value="")

def parse_page_range(page_range_str, num_pages):
    if not page_range_str.strip():
        return list(range(num_pages))
    pages = set()
    for part in page_range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            try:
                start, end = int(start)-1, int(end)-1
                if start < 0 or end >= num_pages or start > end:
                    continue
                pages.update(range(start, end+1))
            except Exception:
                continue
        else:
            try:
                idx = int(part)-1
                if 0 <= idx < num_pages:
                    pages.add(idx)
            except Exception:
                continue
    return sorted(pages)

# Chunk size selection (pages per LLM call)
chunk_size = st.number_input("Pages per chunk (for LLM call)", min_value=1, max_value=20, value=5, step=1)

# Process button
process_clicked = st.button("Extract Requirements")

# Placeholders for preview and download
preview_placeholder = st.empty()
download_placeholder = st.empty()

if process_clicked and uploaded_file is not None and api_key:
    with st.spinner("Extracting text from PDF..."):
        try:
            pages_text = extract_text_from_pdf(uploaded_file)
            selected_pages = parse_page_range(page_range, len(pages_text))
            pages_text = [pages_text[i] for i in selected_pages]
            status_placeholder.success(f"Extracted text from {len(pages_text)} selected pages.")
            # Show preview of first page
            if pages_text:
                preview_placeholder.code(pages_text[0][:2000], language="text")
            else:
                preview_placeholder.info("No text found in PDF.")
        except Exception as e:
            status_placeholder.error(str(e))
            preview_placeholder.empty()
    # --- Chunked LLM processing ---
    chunks = ["\n".join(pages_text[i:i+chunk_size]) for i in range(0, len(pages_text), chunk_size)]
    llm_results = []
    for idx, chunk_text in enumerate(chunks):
        st.write(f"### Chunk {idx+1}/{len(chunks)}")
        output_box = st.empty()  # placeholder for incremental update
        accumulated = ""
        try:
            # Stream tokens incrementally
            for part in stream_structured_data_from_llm(chunk_text, output_format, api_key):
                accumulated += part
                output_box.code(accumulated, language="text")
            llm_results.append(accumulated)
        except Exception as e:
            error_msg = f"Error: {e}"
            output_box.error(error_msg)
            llm_results.append(error_msg)
    # No separate preview placeholder needed; live output boxes show content.
else:
    status_placeholder.empty()
    preview_placeholder.empty()

# (No processing logic yet) 