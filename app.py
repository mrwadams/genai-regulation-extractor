"""Streamlit web application for the GenAI Regulation Extraction Tool (PoC).

This app allows a user to upload a regulation PDF, chunk the document, send each chunk to
Google Gemini (via the `google-genai` client), receive structured JSON/YAML output, and
finally merge and download the complete hierarchical structure.

Configuration notes:
  • The Gemini API key is loaded automatically from the `GOOGLE_API_KEY` environment
    variable or a `.env` file if available. It can also be entered manually in the UI.
  • Adjust the "Pages per chunk" control if you encounter output-length errors.
"""

import streamlit as st
from pdf_parser import extract_text_from_pdf
from gemini_client import (
    stream_structured_data_from_llm,
)
from output_utils import parse_llm_output, merge_structures, flatten_requirements
import os
from dotenv import load_dotenv
import json
import yaml
import pandas as pd

st.set_page_config(page_title="GenAI Regulation Extraction Tool", layout="wide")

st.title("GenAI Regulation Extraction Tool (PoC)")

# Utility to get API key from user input, env, or .env
load_dotenv()
def get_api_key(user_input: str) -> str:
    if user_input:
        return user_input
    env_key = os.getenv("GOOGLE_API_KEY")
    return env_key or ""

# Function to parse page range string into a list of page indices
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

# Create two columns for the input parameters
col1, col2 = st.columns([2, 1])

with col1:
    # Create a container with a border for input parameters
    with st.container(border=True):
        st.subheader("Input Parameters")
        
        # File uploader (PDF only)
        uploaded_file = st.file_uploader("Upload a regulation PDF", type=["pdf"], 
                                        help="Select a PDF document containing the regulation text")
        
        # Test feature: Page range selection
        page_range = st.text_input("(Optional) Enter page range to process (e.g. 1-3,5,7-8):", value="",
                                 help="Specify pages to process, leave empty to process all pages")
        
        # Chunk size selection (pages per LLM call)
        chunk_size = st.number_input(
            "Pages per chunk (for LLM call)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help=(
                "Controls how many PDF pages are sent to the AI in each request. "
                "If you see errors about incomplete or truncated output, try lowering this number (e.g., 1 or 2). "
                "Smaller values reduce the chance of hitting the model's output limit, but may take longer to process the whole document."
            ),
        )

with col2:
    # Create a container with a border for output settings
    with st.container(border=True):
        st.subheader("Output Settings")
        
        # Output format selection
        output_format = st.radio("Select output format", ["JSON", "YAML"], 
                               help="Choose the format for the structured output")
        
        # Add API key input (pre-fill if found)
        default_api_key = os.getenv("GOOGLE_API_KEY")
        api_key = st.text_input("Enter your Gemini API key", type="password", 
                              value=default_api_key or "", 
                              help="Your Google Gemini API key for text processing")
        api_key = get_api_key(api_key)

# Process button with icon
st.markdown("---")
process_clicked = st.button("🚀 Extract Requirements", use_container_width=True, type="primary")

# Place status placeholder here, after all widgets
status_placeholder = st.empty()

# If the user clicked process but is missing inputs, give immediate feedback
if process_clicked and uploaded_file is None:
    status_placeholder.warning("Please upload a PDF file before processing.")
elif process_clicked and not api_key:
    status_placeholder.warning("Please enter your Gemini API key before processing.")

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
            # Show preview of first page with heading in an expander
            if pages_text:
                with st.expander("Preview of First Page Extracted Text", expanded=True):
                    st.code(pages_text[0][:2000], language="text")
            else:
                st.info("No text found in PDF.")
        except Exception as e:
            status_placeholder.error(str(e))
            preview_placeholder.empty()
    # --- Chunked LLM processing ---
    chunks = ["\n".join(pages_text[i:i+chunk_size]) for i in range(0, len(pages_text), chunk_size)]
    llm_results = []

    # Progress bar across chunks
    st.markdown("### Processing Chunks")
    progress_bar = st.progress(0)
    chunk_status = st.empty()

    for idx, chunk_text in enumerate(chunks):
        chunk_status.info(f"Processing chunk {idx+1} of {len(chunks)}...")
        with st.expander(f"Chunk {idx+1}/{len(chunks)}", expanded=False):
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
        # Update overall progress
        progress_val = int(((idx + 1) / len(chunks)) * 100)
        progress_bar.progress(progress_val, text=f"Processing: {progress_val}%")
        
    progress_bar.empty()
    chunk_status.empty()

    # --- Parse chunk responses & merge into a single structure ---
    st.markdown("---")
    st.subheader("Post-processing chunks")

    parsed_structs = []
    for idx, raw in enumerate(llm_results):
        try:
            struct = parse_llm_output(raw, output_format)
            parsed_structs.append(struct)
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Chunk {idx + 1}: parsing failed – {e}\n"
                "Tip: If you see this error, try reducing the 'Pages per chunk' number above. "
                "This helps avoid output truncation by the AI model."
            )

    if parsed_structs:
        try:
            merged_structure = merge_structures(parsed_structs)
            
            # Create tabs for different output views
            st.markdown("---")
            json_tab, yaml_tab, table_tab = st.tabs(["JSON View", "YAML View", "Table View"])
            
            # Prepare the download button data
            if output_format.upper() == "JSON":
                download_bytes = json.dumps(merged_structure, indent=2).encode()
                mime = "application/json"
                file_name = "extracted_structure.json"
            else:
                yaml_str = yaml.safe_dump(merged_structure, sort_keys=False, indent=2)
                download_bytes = yaml_str.encode()
                mime = "text/yaml"
                file_name = "extracted_structure.yaml"
            
            # Download buttons in a row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 Download Structured Output",
                    data=download_bytes,
                    file_name=file_name,
                    mime=mime,
                )

            # Content for JSON tab
            with json_tab:
                with st.expander("Merged JSON Structure", expanded=True):
                    st.json(merged_structure)
            
            # Content for YAML tab
            with yaml_tab:
                with st.expander("Merged YAML Structure", expanded=True):
                    yaml_str = yaml.safe_dump(merged_structure, sort_keys=False, indent=2)
                    st.code(yaml_str, language="yaml")
            
            # Content for Table tab
            with table_tab:
                rows = flatten_requirements(merged_structure)
                if rows:
                    df = pd.DataFrame(rows)
                    st.subheader("Requirements Table")

                    # Always show the full table
                    st.info(f"Showing all {len(df)} requirements.")

                    st.dataframe(
                        df, # Display the original, unfiltered dataframe
                        use_container_width=True,
                        column_config={
                            "requirement_text": st.column_config.TextColumn(
                                "Requirement Text",
                                width="large",
                            ),
                            "keywords": st.column_config.ListColumn(
                                "Keywords"
                            ) if "keywords" in df.columns else None
                        }
                    )
                    
                    # Download CSV button for the full data
                    if not df.empty:
                        csv_bytes = df.to_csv(index=False).encode()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_bytes,
                            file_name="requirements.csv",
                            mime="text/csv",
                            key="download_csv_button" 
                        )
                    else:
                        st.warning("No data to download.")
                else:
                    st.info("No requirements found in the merged structure to display in table.")
                    
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to merge structured outputs: {e}")
    else:
        st.warning("No valid structured outputs to merge.")
    # End of processing logic
else:
    status_placeholder.empty()
    preview_placeholder.empty()

# End of script 