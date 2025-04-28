## GenAI Regulation Extraction Tool: Specification Document

**Version:** 1.3 Final
**Date:** April 21, 2025

### 1. Introduction

**1.1 Purpose:**
This document outlines the requirements and specifications for a Generative AI (GenAI) tool designed to process regulatory documents (primarily in PDF format) uploaded by users. The tool's primary function is to identify, extract, and structure key information, specifically regulatory sections and requirements, into a machine-readable format (JSON or YAML). This version reflects refinements for an initial Proof of Concept (PoC).

**1.2 Scope:**
The tool will focus on:
* Accepting user-uploaded regulation documents (initial focus on PDF).
* Parsing and understanding the document structure (headings, sections, paragraphs).
* Identifying and extracting distinct regulatory requirements, obligations, or rules.
* Structuring the extracted information hierarchically, reflecting the document's organization.
* Providing the structured output in user-selectable formats (JSON, YAML).

**Out of Scope (for PoC):**
* Interpretation or legal analysis of the requirements.
* Comparison between different regulation documents.
* Direct integration with compliance management systems.
* Handling heavily scanned or image-based PDFs with poor OCR quality (best-effort basis).
* Support for formats other than PDF initially.
* High scalability and enterprise-grade security (addressed in later phases).

**1.3 Goals:**
* **(PoC Focus)** Demonstrate the feasibility of using GenAI to extract requirements from regulations.
* Reduce manual effort and time required to analyze and digitize regulations.
* Improve the accuracy and consistency of extracting regulatory requirements.
* Provide a structured, machine-readable output suitable for downstream processing or analysis.
* Offer a user-friendly interface (using Streamlit) for uploading documents and retrieving results.

**1.4 Target Audience:**
Compliance officers, legal professionals, risk managers, regulatory analysts, consultants, and software developers building compliance solutions. (Initially, internal stakeholders for PoC evaluation).

### 2. Functional Requirements

**2.1 Document Upload:**
* **FR-001:** The system shall provide an interface for users to upload one or more regulation documents.
* **FR-002:** The system shall initially support PDF (.pdf) file format.
* **FR-003:** The system should provide feedback on the upload status (e.g., uploading, processing, complete, error).
* **FR-004:** The system should handle reasonably large file sizes (e.g., up to 50MB).

**2.2 Document Processing & Parsing:**
* **FR-005:** The system shall parse the text content and basic structure (headings, paragraphs, lists) from the uploaded PDF documents.
* **FR-006:** The system shall employ GenAI models to understand the semantic content and hierarchical structure of the regulation.

**2.3 Section & Requirement Extraction:**
* **FR-007:** The system shall identify distinct sections, sub-sections, articles, clauses, etc., based on formatting and numbering patterns.
* **FR-008:** The system shall use GenAI (potentially combined with NLP techniques) to identify sentences or paragraphs that constitute specific requirements, obligations, prohibitions, or definitions.
* **FR-009:** The system shall attempt to capture the relationship between requirements and their parent sections/sub-sections.
* **FR-010:** The system should extract the verbatim text of the requirement.
* **FR-011:** The system should associate extracted requirements with their corresponding section number/identifier within the document.

**2.4 Output Generation:**
* **FR-012:** The system shall structure the extracted sections and requirements hierarchically.
* **FR-013:** The system shall provide the structured output in JSON format.
* **FR-014:** The system shall provide the structured output in YAML format.
* **FR-015:** The system shall allow the user to download the generated JSON or YAML file.
* **FR-016:** The system may offer a preview of the extracted structure within the UI before download.

### 3. Non-Functional Requirements

* **NFR-001 (Accuracy):** The system should strive for high accuracy in identifying section boundaries and extracting relevant requirements. Define target metrics (e.g., Precision, Recall, F1-score) based on a benchmark dataset. *(Accuracy remains important for PoC viability)*.
* **NFR-002 (Performance):** Processing time should be reasonable for typical regulation document sizes. Provide estimated processing times (e.g., < 5 minutes for a 100-page document). *(Acceptable performance needed for usability)*.
* **NFR-003 (Scalability):** The system architecture should support single-user or limited concurrent use typical for a PoC. High scalability is not a primary objective for this phase but should be considered for future versions.
* **NFR-004 (Usability):** The user interface (built with Streamlit) should be intuitive and require minimal training.
* **NFR-005 (Security):** As the input documents are public information and this is a PoC, stringent security measures are not the primary focus. However, basic precautions should be taken (e.g., secure handling of API keys, avoiding unnecessary data persistence). Data encryption and robust access controls are deferred to later phases.
* The application should securely and conveniently load the Gemini API key from a `.env` file or the `GOOGLE_API_KEY` environment variable if available, to improve usability and avoid unnecessary user prompts for the API key.
* **NFR-006 (Reliability):** The system should be robust enough for demonstration purposes and handle potential errors during PDF parsing or AI processing gracefully.

### 4. System Architecture (High-Level)

A potential architecture could include:

1.  **Frontend (UI):** A web-based interface built using Streamlit, providing components for file upload (`st.file_uploader`), status display (`st.progress`, `st.spinner`), preview (`st.json`, `st.code`), and download (`st.download_button`).
2.  **Backend Logic (Integrated with Streamlit):** Python code within the Streamlit application to manage requests, orchestrate processing, and interact with other modules. User authentication is not required for the PoC.
3.  **Document Ingestion & Parsing Module:** Receives uploaded files (via Streamlit), uses libraries (e.g., PyMuPDF, pdfminer.six) to extract text and basic structure. Basic handling for OCR is out of scope for PoC.
4.  **GenAI Processing Module:**
    * Sends cleaned text/structure to the specified GenAI model (`gemini-2.0-flash`) via its API.
    * Uses carefully crafted prompts to instruct the model to identify sections and requirements.
    * **Handles Output Token Limits:** Implements strategies (e.g., iterative processing of document chunks, prompt modification for partial output, or potentially streaming if supported) to manage the model's output token limit (e.g., 8,192 tokens) and ensure complete extraction for large documents.
    * May involve chunking large documents if *input* context limits are exceeded (less likely with 1M input, but relevant for output handling).
5.  **Structuring & Formatting Module:**
    * Receives the raw output (potentially partial or streamed) from the GenAI model.
    * Parses, cleans, and structures it into the defined hierarchical JSON/YAML schema.
    * **Merges Results (if chunking):** If iterative processing is used, this module is responsible for merging the structured outputs from different chunks into a single coherent document structure.
6.  **Output Storage/Delivery:** Generates the final JSON/YAML data structures/strings directly within the Streamlit app for preview and download. No persistent storage is required for the PoC.

*(Diagram could be inserted here in a more formal document)*

### 5. Data Handling

* **Input:** PDF files (publicly available regulations). Consider potential variations in PDF quality and structure.
* **Intermediate:** Extracted text, structural metadata, GenAI model inputs/outputs (handled in memory where possible).
* **Output:** JSON or YAML files conforming to a defined schema (see Section 9), generated on-the-fly.
* **Privacy/Security:** Given the public nature of the input data, data privacy concerns are minimal for the PoC. Focus on securing API credentials and avoiding unnecessary logging or storage of intermediate data. Compliance with data protection regulations like GDPR is less critical for this phase but essential for future production versions. Anonymization is not applicable.

### 6. GenAI Model Details

* **Model Selection:** Utilize the `gemini-2.0-flash` model via the new `google-genai` client library, suitable for rapid iteration and cost-effective PoC development. Prioritize models with good instruction-following and structured data generation capabilities.
* **Prompt Engineering:** Develop robust prompts that clearly instruct the model on:
    * The desired output format (JSON/YAML schema).
    * How to identify section headings and hierarchy.
    * Criteria for identifying a "requirement" (e.g., presence of modal verbs like "shall," "must," specific keywords).
    * How to handle nuances like definitions, exceptions, etc.
* **Handling Output Token Limits:**
    * **Acknowledge Constraint:** The `gemini-2.0-flash` model has a significant output token limit (e.g., 8,192 tokens). The structured JSON/YAML output for large regulations could exceed this limit if the entire document is processed in a single API call.
    * **Potential Strategies:**
        1.  **Iterative Processing (Chunking):** Process the document in logical chunks (e.g., chapters, major sections). Send each chunk's text to the LLM and request the structured output *for that chunk*. Ensure the prompt requests output that fits within the limit for the chunk size. The application backend will then merge the structured results. This is likely the most robust approach for full extraction, and should use the new `google-genai` client for all Gemini API calls.
        2.  **Prompt Modification (Partial Output):** For very large sections, prompts could potentially request the model to return only a subset of the structure or paginate its response, although this adds complexity to the prompt design and backend logic. (Less preferred for PoC unless chunking proves difficult).
        3.  **Streaming:** Investigate if the `google-genai` API and model support streaming responses. If so, the application could process the structured output incrementally as it's received, mitigating the single-response size limit.
* **Fine-tuning (Optional):** Out of scope for the initial PoC. May be considered for future enhancements.
* **Evaluation:** Establish a small "golden dataset" of manually annotated regulations to evaluate the accuracy (Precision, Recall, F1) of the extraction process during PoC development. Regular testing and refinement of prompts are crucial.

### 7. User Interface (UI) / User Experience (UX)

* **Framework:** The UI will be developed using Streamlit.
* **Simplicity:** Leverage standard Streamlit components for a clean, functional interface.
* **Workflow:**
    1.  Use `st.file_uploader` for PDF file selection.
    2.  Use `st.selectbox` or `st.radio` for selecting output format (JSON/YAML).
    3.  Use `st.text_input` to optionally specify a set of pages (e.g., 1-3,5,7-8) to process for testing purposes, so only those pages are loaded and sent to the LLM. This enables rapid validation and debugging before running on the full document.
    4.  Use `st.button` to trigger the extraction process.
    5.  Display status using `st.spinner` or `st.progress`.
    6.  Upon completion:
        * Display a preview using `st.json` or `st.code`.
        * Provide download links using `st.download_button`.
    7.  Display errors using `st.error`.

### 8. Error Handling & Validation

* **Invalid PDFs:** Detect corrupted or password-protected PDFs using parsing library exceptions and inform the user via `st.warning` or `st.error`.
* **Parsing Errors:** Handle failures during text extraction and report errors.
* **GenAI Errors:** Implement basic retries for transient API errors. Handle cases where the model fails to produce valid structured output (e.g., JSON parsing errors) and report clearly.
* **Output Limit Handling:** Add checks to detect potentially truncated output if strategies to manage the output limit fail or are imperfect. Inform the user if the output might be incomplete.
* **Ambiguity:** Recognize that AI extraction may not be perfect. The output should be presented as a best effort. Users should be advised to review the output.
* **Low-Quality Scans:** Inform the user if the PDF quality seems low, potentially impacting results.

### 9. Output Format Examples

**9.1 JSON Example:**
```json
{
  "document_title": "Example Regulation XYZ",
  "source_file": "regulation_xyz.pdf",
  "extraction_timestamp": "2025-04-21T12:00:00Z",
  "sections": [
    {
      "id": "1",
      "title": "Introduction and Scope",
      "text_content": "This document outlines the requirements for...",
      "requirements": [],
      "subsections": [
        {
          "id": "1.1",
          "title": "Applicability",
          "text_content": "These regulations apply to all licensed entities...",
          "requirements": [
            {
              "req_id": "RQ-1.1-001",
              "text": "These regulations apply to all licensed entities operating within the jurisdiction.",
              "keywords": ["apply", "licensed entities"]
            }
          ],
          "subsections": []
        }
      ]
    },
    {
      "id": "2",
      "title": "Operational Requirements",
      "text_content": "Entities must adhere to the following operational standards...",
      "requirements": [],
      "subsections": [
        {
          "id": "2.1",
          "title": "Data Security",
          "text_content": "Entities shall implement robust data security measures.",
          "requirements": [
            {
              "req_id": "RQ-2.1-001",
              "text": "Entities shall implement robust data security measures.",
              "keywords": ["shall implement", "data security"]
            },
            {
              "req_id": "RQ-2.1-002",
              "text": "Security measures must include encryption for data at rest and in transit.",
              "keywords": ["must include", "encryption"]
            }
          ],
          "subsections": []
        }
      ]
    }
    // ... more sections
  ]
}
```

**9.2 YAML Example:**
```yaml
document_title: Example Regulation XYZ
source_file: regulation_xyz.pdf
extraction_timestamp: '2025-04-21T12:00:00Z'
sections:
  - id: '1'
    title: Introduction and Scope
    text_content: This document outlines the requirements for...
    requirements: []
    subsections:
      - id: '1.1'
        title: Applicability
        text_content: These regulations apply to all licensed entities...
        requirements:
          - req_id: RQ-1.1-001
            text: These regulations apply to all licensed entities operating within the jurisdiction.
            keywords:
              - apply
              - licensed entities
        subsections: []
  - id: '2'
    title: Operational Requirements
    text_content: Entities must adhere to the following operational standards...
    requirements: []
    subsections:
      - id: '2.1'
        title: Data Security
        text_content: Entities shall implement robust data security measures.
        requirements:
          - req_id: RQ-2.1-001
            text: Entities shall implement robust data security measures.
            keywords:
              - shall implement
              - data security
          - req_id: RQ-2.1-002
            text: Security measures must include encryption for data at rest and in transit.
            keywords:
              - must include
              - encryption
        subsections: []
# ... more sections
```
*(Note: The exact schema, including fields like `keywords` or `req_id`, can be refined based on specific needs.)*

### 10. Future Enhancements

* Support for other document formats (e.g., DOCX, HTML).
* Ability to handle scanned PDFs using integrated OCR.
* Extraction of definitions, dates, specific entities.
* Requirement summarization.
* Comparison feature to highlight differences between regulation versions.
* Confidence scoring for extracted requirements.
* API for programmatic access.
* Integration with compliance management platforms.
* User feedback mechanism to improve extraction accuracy over time.
* Address Scalability and Security for production.

### 11. Assumptions & Dependencies

* **Assumption:** Regulation documents generally follow a hierarchical structure with identifiable headings/numbering.
* **Assumption:** Key requirements are often explicitly stated using modal verbs or specific phrasing.
* **Dependency:** Access to the `gemini-2.0-flash` model API and associated API keys.
* **Dependency:** Availability of robust Python PDF parsing libraries (e.g., PyMuPDF, pdfminer.six).
* **Dependency:** Development environment with Python and Streamlit installed.
* **Dependency:** User-provided documents are reasonably well-formatted.
* **Assumption:** Strategies like iterative processing or streaming can effectively manage the model's output token limit for expected document sizes.

---


