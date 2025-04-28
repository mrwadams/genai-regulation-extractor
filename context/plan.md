---

## Coding Plan: GenAI Regulation Extraction Tool (PoC)

**Based on:** Specification Document v1.3 (`genai_regulation_spec_v1`)

**Goal:** Develop a functional Proof of Concept demonstrating the core capabilities outlined in the specification, including handling potential output token limits.

### Phase 1: Setup & Environment

* **Task 1.1: Create Project Directory:**
    * Set up a dedicated folder for the project (e.g., `regulation_extractor_poc`).
* **Task 1.2: Set Up Virtual Environment:**
    * Create and activate a Python virtual environment (e.g., using `venv`).
* **Task 1.3: Install Dependencies:**
    * Install necessary Python libraries: `streamlit`, `google-genai`, `PyMuPDF`.
    * Install optional libraries if needed: `python-dotenv`, `PyYAML`.
    * Create a `requirements.txt` file.
* **Task 1.4: API Key Configuration:**
    * Obtain a Gemini API key.
    * The application will attempt to load the API key automatically from a `.env` file or the `GOOGLE_API_KEY` environment variable if present, reducing the need for manual entry.

### Phase 2: Basic Streamlit UI Structure

* **Task 2.1: Create Main App File:**
    * Create the main Python script (e.g., `app.py`).
* **Task 2.2: Basic Layout & Title:**
    * Add `st.title()` and potentially `st.columns` or `st.container`.
* **Task 2.3: Implement File Uploader:**
    * Add `st.file_uploader()` accepting PDF files.
* **Task 2.4: Add Output Format Selection:**
    * Use `st.radio()` or `st.selectbox()` for "JSON" / "YAML".
* **Task 2.5: Add Processing Trigger:**
    * Include an `st.button()` to start the extraction.
* **Task 2.6: Set Up Placeholders:**
    * Add placeholders (`st.empty()`) for status, preview, and download button.
* **Task 2.7: Add Test Page Range Feature:**
    * Implement a UI input allowing the user to specify a set of pages (e.g., 1-3,5,7-8) to process for testing purposes, so that only those pages are loaded and sent to the LLM. This enables rapid validation and debugging before running on the full document.

### Phase 3: PDF Parsing Module

* **Task 3.1: Create Parsing Function:**
    * Define `extract_text_from_pdf(uploaded_file)`.
* **Task 3.2: Implement Text Extraction:**
    * Use `PyMuPDF` to open the PDF bytes and extract text page by page.
    * Concatenate text. Consider returning text per page or identified section to aid chunking in Phase 4.
* **Task 3.3: Basic Error Handling:**
    * Use `try...except` to handle PDF parsing errors.

### Phase 4: Gemini Integration & Prompting (Handling Output Limits)

* **Task 4.1: Configure Gemini Client:**
    * Import `google.genai as genai` (using the new `google-genai` client library).
    * Configure the client: `genai.configure(api_key=...)`.
    * Instantiate the model: `model = genai.GenerativeModel('gemini-2.0-flash')`.
    * Verify exact function calls against `google-genai` documentation.
* **Task 4.2: Develop Core Prompt:**
    * Create a detailed prompt instructing the model on its role, section/requirement identification, and desired output format (JSON/YAML schema).
* **Task 4.3: Create API Call Function:**
    * Define `get_structured_data_from_llm(text_content, output_format)`.
* **Task 4.4: Implement Iterative Processing Strategy (Handling Output Limit):**
    * **Rationale:** Primary strategy for the 8k output token limit.
    * **Logic:**
        * Define a chunking strategy (e.g., N pages, logical sections).
        * Create a wrapper function or modify `get_structured_data_from_llm` to iterate through text chunks.
        * For each chunk, call the Gemini API (`model.generate_content`) with a prompt requesting structured output *only for that chunk*.
        * Collect the structured responses from each chunk.
    * **Note:** Refine Phase 3 text extraction if needed to support logical chunking.
* **Task 4.5: API Error Handling:**
    * Wrap `model.generate_content` calls (within the chunk processing loop) in `try...except` blocks.
* **Task 4.6: Investigate Streaming (Optional/Alternative):**
    * Check `google-genai` documentation for `generate_content(..., stream=True)` support with `gemini-2.0-flash`.
    * If supported, implement logic to process the response stream incrementally.

### Phase 5: Output Structuring, Display & Download

* **Task 5.1: Parse LLM Response:**
    * Parse the response from each chunk (`json.loads` or `yaml.safe_load`) or process the stream.
* **Task 5.2: Merge Chunked Results:**
    * If using iterative processing (Task 4.4), implement logic to merge the parsed structures from individual chunks into a single, coherent final structure conforming to the defined schema.
* **Task 5.3: Validate Structure:**
    * Perform basic checks on the final merged structure.
* **Task 5.4: Handle Malformed Output:**
    * Use `try...except` during parsing and merging; report errors if structure is invalid.
* **Task 5.5: Display Preview:**
    * Display the final, merged structure using `st.json()` or `st.code()`.
* **Task 5.6: Implement Download Button:**
    * Provide `st.download_button` for the final, merged JSON/YAML data.

### Phase 6: Refinement & Workflow Integration

* **Task 6.1: Connect UI to Logic:**
    * Wire the "Extract Requirements" button to trigger the full workflow (parsing, iterative processing/streaming, merging, display).
* **Task 6.2: Implement Status Updates:**
    * Use `st.spinner()` or `st.progress()` to show activity, indicating progress through chunks if applicable.
* **Task 6.3: Improve Error Display:**
    * Use `st.error()` for failures in parsing, API calls, chunk processing, merging, or output validation.
* **Task 6.4: Initial Testing & Prompt Tuning:**
    * Test with various regulation PDFs, including large ones.
    * Refine prompts, chunking strategy, and merging logic based on results.

### Phase 7: Documentation & Cleanup

* **Task 7.1: Add Code Comments:**
    * Explain functions, classes, and complex logic (especially chunking/merging).
* **Task 7.2: Create README.md:**
    * Include setup, API key configuration, and run instructions.
* **Task 7.3: Code Cleanup:**
    * Remove debugging statements, unused code; ensure formatting consistency.