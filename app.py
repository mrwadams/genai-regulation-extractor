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

# Initialize session state
if 'app_data' not in st.session_state:
    st.session_state.app_data = {
        "merged_structure": None,
        "df_requirements": None,
        "download_structured_bytes": None,
        "download_structured_mime": None,
        "download_structured_filename": None,
        "download_csv_bytes": None,
        "last_error_message": None,
        "show_results": False,
        "extracted_text_preview": None
    }

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
        api_key_input = st.text_input("Enter your Gemini API key", type="password", 
                                      value=default_api_key or "", 
                                      help="Your Google Gemini API key for text processing")
        api_key = get_api_key(api_key_input)

# Process button with icon
st.markdown("---")
process_clicked = st.button("🚀 Extract Requirements", use_container_width=True, type="primary")

# Place status placeholder here, after all widgets
status_placeholder = st.empty()

# If the user clicked process but is missing inputs, give immediate feedback
if process_clicked:
    if uploaded_file is None:
        status_placeholder.warning("Please upload a PDF file before processing.")
        st.session_state.app_data["show_results"] = False
    elif not api_key:
        status_placeholder.warning("Please enter your Gemini API key before processing.")
        st.session_state.app_data["show_results"] = False
    else:
        # Inputs are present, proceed with processing
        st.session_state.app_data["show_results"] = False # Reset on new processing attempt
        st.session_state.app_data["last_error_message"] = None
        st.session_state.app_data["extracted_text_preview"] = None
        pages_text_global = [] # To check if PDF extraction was successful

        with st.spinner("Extracting text from PDF..."):
            try:
                pages_text_extracted = extract_text_from_pdf(uploaded_file)
                selected_pages = parse_page_range(page_range, len(pages_text_extracted))
                pages_text_global = [pages_text_extracted[i] for i in selected_pages]
                status_placeholder.success(f"Extracted text from {len(pages_text_global)} selected pages.")
                if pages_text_global:
                    # Show preview of first page with heading in an expander
                    st.session_state.app_data["extracted_text_preview"] = pages_text_global[0][:2000]
                    with st.expander("Preview of First Page Extracted Text", expanded=True):
                        st.code(st.session_state.app_data["extracted_text_preview"], language="text")
                else:
                    msg = "No text found in PDF or selected page range."
                    status_placeholder.info(msg)
                    st.session_state.app_data["last_error_message"] = msg
            except Exception as e:
                error_msg = f"PDF Extraction Error: {str(e)}"
                status_placeholder.error(error_msg)
                st.session_state.app_data["last_error_message"] = error_msg
        
        if st.session_state.app_data["last_error_message"] is None and pages_text_global:
            # --- Chunked LLM processing ---
            chunks = ["\\n".join(pages_text_global[i:i+chunk_size]) for i in range(0, len(pages_text_global), chunk_size)]
            llm_results = []

            st.markdown("### Processing Chunks")
            progress_bar = st.progress(0)
            chunk_status = st.empty()

            for idx, chunk_text in enumerate(chunks):
                chunk_status.info(f"Processing chunk {idx+1} of {len(chunks)}...")
                with st.expander(f"Chunk {idx+1}/{len(chunks)}", expanded=False):
                    output_box = st.empty()
                    accumulated = ""
                    try:
                        for part in stream_structured_data_from_llm(chunk_text, output_format, api_key):
                            accumulated += part
                            output_box.code(accumulated, language="text")
                        llm_results.append(accumulated)
                    except Exception as e:
                        error_msg = f"Error processing chunk {idx+1}: {e}"
                        output_box.error(error_msg)
                        llm_results.append(error_msg) # Store error to indicate failure for this chunk
                progress_val = int(((idx + 1) / len(chunks)) * 100)
                progress_bar.progress(progress_val, text=f"Processing: {progress_val}%")
            
            progress_bar.empty()
            chunk_status.empty()

            # --- Parse chunk responses & merge into a single structure ---
            post_processing_hr_placeholder = st.empty()
            post_processing_header_placeholder = st.empty()
            post_processing_hr_placeholder.markdown("---")
            post_processing_header_placeholder.subheader("Post-processing chunks")

            parsed_structs = []
            has_parsing_errors = False
            for idx, raw in enumerate(llm_results):
                try:
                    # Attempt to parse even if it was an error message, parse_llm_output should handle it or raise
                    struct = parse_llm_output(raw, output_format)
                    if isinstance(struct, dict) and struct.get("error"): # Heuristic for error objects from LLM
                         st.error(f"Chunk {idx + 1}: Contained an error from LLM - {struct.get('error')}")
                         has_parsing_errors = True
                    parsed_structs.append(struct)
                except Exception as e:
                    st.error(
                        f"Chunk {idx + 1}: parsing failed – {e}\\n"
                        "Tip: If you see this error, try reducing the 'Pages per chunk' number above. "
                        "This helps avoid output truncation by the AI model."
                    )
                    has_parsing_errors = True
            
            if parsed_structs and not has_parsing_errors: # Proceed if some structs were parsed and no critical parsing errors
                try:
                    merged_structure = merge_structures(parsed_structs)
                    st.session_state.app_data["merged_structure"] = merged_structure
                    
                    if output_format.upper() == "JSON":
                        st.session_state.app_data["download_structured_bytes"] = json.dumps(merged_structure, indent=2).encode()
                        st.session_state.app_data["download_structured_mime"] = "application/json"
                        st.session_state.app_data["download_structured_filename"] = "extracted_structure.json"
                    else: # YAML
                        yaml_str = yaml.safe_dump(merged_structure, sort_keys=False, indent=2)
                        st.session_state.app_data["download_structured_bytes"] = yaml_str.encode()
                        st.session_state.app_data["download_structured_mime"] = "text/yaml"
                        st.session_state.app_data["download_structured_filename"] = "extracted_structure.yaml"
                    
                    rows = flatten_requirements(merged_structure)
                    if rows:
                        df = pd.DataFrame(rows)
                        st.session_state.app_data["df_requirements"] = df
                        if not df.empty:
                            st.session_state.app_data["download_csv_bytes"] = df.to_csv(index=False).encode()
                        else:
                            st.session_state.app_data["download_csv_bytes"] = None
                    else:
                        st.session_state.app_data["df_requirements"] = pd.DataFrame()
                        st.session_state.app_data["download_csv_bytes"] = None
                    
                    st.session_state.app_data["show_results"] = True
                    st.session_state.app_data["last_error_message"] = None # Clear any previous non-critical message
                    status_placeholder.empty() # Clear "Extracted text..."
                    post_processing_hr_placeholder.empty()
                    post_processing_header_placeholder.empty()

                except Exception as e:
                    err_msg = f"Failed to merge structured outputs: {e}"
                    st.error(err_msg)
                    st.session_state.app_data["last_error_message"] = err_msg
                    st.session_state.app_data["show_results"] = False
            elif has_parsing_errors:
                err_msg = "Some chunks failed to parse correctly. Cannot merge."
                st.error(err_msg)
                st.session_state.app_data["last_error_message"] = err_msg
                st.session_state.app_data["show_results"] = False
            else: # No parsed_structs
                warn_msg = "No valid structured outputs to merge."
                st.warning(warn_msg)
                if not st.session_state.app_data["last_error_message"]: # Don't overwrite a more specific error
                    st.session_state.app_data["last_error_message"] = warn_msg
                st.session_state.app_data["show_results"] = False
        elif not pages_text_global and not st.session_state.app_data["last_error_message"] :
             # This case is if PDF extraction was 'successful' but yielded no text, and no specific error was set
            no_text_msg = "No text content was extracted from the PDF/selected pages to process."
            status_placeholder.warning(no_text_msg) # Use warning, as it's not an exception
            st.session_state.app_data["last_error_message"] = no_text_msg # Log it
            st.session_state.app_data["show_results"] = False


# Display Results Section (reads from session state)
if st.session_state.app_data.get("show_results"):
    st.markdown("---")
    st.subheader("Processed Results")

    # Display extracted text preview if it exists from a successful run
    if st.session_state.app_data.get("extracted_text_preview"):
        with st.expander("Preview of First Page Extracted Text", expanded=False): # Default to collapsed when showing results
            st.code(st.session_state.app_data["extracted_text_preview"], language="text")


    # Download buttons in a row
    dl_col1, dl_col2 = st.columns(2) # Changed to 2 columns as per original structure, adjust if 3 were intended
    with dl_col1:
        if st.session_state.app_data.get("download_structured_bytes"):
            st.download_button(
                label="📥 Download Structured Output",
                data=st.session_state.app_data["download_structured_bytes"],
                file_name=st.session_state.app_data["download_structured_filename"],
                mime=st.session_state.app_data["download_structured_mime"],
                key="download_structured_btn" # Unique key
            )
    with dl_col2:
        if st.session_state.app_data.get("download_csv_bytes"):
            st.download_button(
                label="📥 Download CSV",
                data=st.session_state.app_data["download_csv_bytes"],
                file_name="requirements.csv",
                mime="text/csv",
                key="download_csv_btn" # Unique key
            )
        elif st.session_state.app_data.get("df_requirements") is not None and \
             st.session_state.app_data["df_requirements"].empty:
            # This condition implies that flatten_requirements ran but produced no rows for the table.
            # We should indicate that CSV download isn't available due to no table data.
            st.caption("No data for CSV download (table is empty).")


    json_tab, yaml_tab, table_tab = st.tabs(["JSON View", "YAML View", "Table View"])
    
    display_merged_structure = st.session_state.app_data.get("merged_structure")
    display_df = st.session_state.app_data.get("df_requirements")

    if display_merged_structure:
        with json_tab:
            with st.expander("Merged JSON Structure", expanded=True):
                st.json(display_merged_structure)
        
        with yaml_tab:
            with st.expander("Merged YAML Structure", expanded=True):
                yaml_str_display = yaml.safe_dump(display_merged_structure, sort_keys=False, indent=2)
                st.code(yaml_str_display, language="yaml")
    
    with table_tab:
        if display_df is not None and not display_df.empty:
            st.subheader("Requirements Table")
            st.info(f"Showing all {len(display_df)} requirements.")
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "requirement_text": st.column_config.TextColumn(
                        "Requirement Text",
                        width="large",
                    ),
                    "keywords": st.column_config.ListColumn(
                        "Keywords"
                    ) if "keywords" in display_df.columns else None
                }
            )
        elif display_df is not None: # Empty DataFrame
            st.info("No requirements found in the merged structure to display in table.")
        else: # df_requirements is None
             if display_merged_structure: # If we have a structure but no df, it means flatten_requirements yielded nothing
                 st.info("No requirements could be flattened into a table from the structure.")
             # else: (no merged structure either, handled by show_results=False)

elif st.session_state.app_data.get("last_error_message") and not process_clicked:
    # If not currently processing (i.e., process_clicked is False), but there was a previous error, show it.
    # This is helpful if the user changes some input after an error, but doesn't re-process.
    # status_placeholder might have been cleared or overwritten by other UI elements if not handled carefully.
    # The explicit check for `not process_clicked` ensures this message doesn't immediately overwrite
    # a new "Processing..." message or a fresh success/warning from a current run.
    status_placeholder.error(f"Last processing attempt resulted in an error: {st.session_state.app_data['last_error_message']}")


# End of script 